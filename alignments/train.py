import torch
import torch.nn as nn
import time

import alignments.neuralibm1_helper as neuralibm1_helper
import alignments.alignmentvae_helper as alignmentvae_helper

from pathlib import Path
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from alignments.data import ParallelDataset, PAD_TOKEN, create_batch, BucketingParallelDataLoader
from alignments.hparams import Hyperparameters
from alignments.train_utils import load_data, load_vocabularies, model_parameter_count
from alignments.train_utils import create_optimizer, gradient_norm
from alignments.models import initialize_model

def create_model(hparams, vocab_src, vocab_tgt):
    if hparams.model_type == "neuralibm1":
        model = neuralibm1_helper.create_model(hparams, vocab_src, vocab_tgt)
        train_fn = neuralibm1_helper.train_step
        validate_fn = neuralibm1_helper.validate
    elif hparams.model_type in ["bernoulli-RF", "bernoulli-ST", "concrete", "kuma", "hardkuma"]:
        model = alignmentvae_helper.create_model(hparams, vocab_src, vocab_tgt)
        train_fn = alignmentvae_helper.train_step
        validate_fn = alignmentvae_helper.validate
    else:
        raise Exception(f"Unknown model_type: {hparams.model_type}")

    return model, train_fn, validate_fn

def train(model, optimizer, lr_scheduler, train_data, val_data, val_alignments, vocab_src,
          vocab_tgt, device, out_dir, train_step, validate, hparams):
    """
    :param train_step: function that performs a single training step and returns
                       training loss. Takes as inputs: model, x,
                       seq_mask_x, seq_len_x, y, seq_mask_y,
                       seq_len_y, hparams, step.
    :param validate: function that performs validation and returns validation
                     AER, used for model selection. Takes as inputs: model,
                     val_data, val_alignments vocab, device, hparams, step, summary_writer.
                     summary_writer can be None if no summaries should be made.
                     This function should perform all evaluation, write
                     summaries and write any validation metrics to the
                     standard out.
    """

    # Create a dataloader that buckets the batches.
    dl = DataLoader(train_data, batch_size=hparams.batch_size,
                    shuffle=True, num_workers=4)
    bucketing_dl = BucketingParallelDataLoader(dl)

    # Save the best model based on development BLEU.
    best_model_location = out_dir / "model.pt"
    best_aer = 2.
    best_step = 0
    best_epoch = 0

    # Keep track of some stuff in TensorBoard.
    summary_writer = SummaryWriter(log_dir=str(out_dir))

    # Define training statistics to keep track of.
    tokens_start = time.time()
    num_tokens = 0
    total_train_loss = 0.
    num_sentences = 0
    step = 0
    epoch_num = 1
    evaluations_no_improvement = 0

    # Define the evaluation function.
    def run_evaluation():
        nonlocal best_aer, best_epoch, best_step, evaluations_no_improvement

        # Perform model validation, keep track of validation BLEU for model
        # selection.
        model.eval()
        val_aer = validate(model, val_data, val_alignments, vocab_src, vocab_tgt, device,
                           hparams, step, summary_writer=summary_writer)

        # Update the learning rate scheduler.
        if hparams.lr_reduce_patience >= 0:
            lr_scheduler.step(val_aer)
            if lr_scheduler.cooldown_counter == hparams.lr_reduce_cooldown:
                print(f"Reduced the learning rate with a factor"
                      f" {hparams.lr_reduce_factor}")

        # Save the best model.
        if val_aer < best_aer:
            evaluations_no_improvement = 0
            best_aer = val_aer
            best_epoch = epoch_num
            best_step = step
            torch.save(model.state_dict(), best_model_location)
        else:
            evaluations_no_improvement += 1

    # Start the training loop.
    while (epoch_num <= hparams.num_epochs) or (evaluations_no_improvement < hparams.patience):

        # Train for 1 epoch.
        for sentences_x, sentences_y in bucketing_dl:
            model.train()

            # Perform a forward pass through the model
            include_null = (hparams.model_type == "neuralibm1")
            x, seq_mask_x, seq_len_x = create_batch(sentences_x,
                                                    vocab_src, device,
                                                    include_null=include_null)
            y, seq_mask_y, seq_len_y = create_batch(sentences_y,
                                                    vocab_tgt, device)
            loss = train_step(model, x, seq_mask_x, seq_len_x,
                              y, seq_mask_y, seq_len_y, hparams, step)

            # Backpropagate and update gradients.
            loss.backward()
            if hparams.max_gradient_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(),
                                         hparams.max_gradient_norm)
            optimizer.step()

            # Update statistics.
            num_tokens += (seq_len_x.sum() + seq_len_y.sum()).item()
            num_sentences += x.size(0)
            total_train_loss += loss.item() * x.size(0)

            # Print training stats every now and again.
            if step % hparams.print_every == 0:
                elapsed = time.time() - tokens_start
                tokens_per_sec = num_tokens / elapsed if step != 0 else 0
                grad_norm = gradient_norm(model)
                print(f"({epoch_num}) step {step}: "
                       f"training loss = {total_train_loss/num_sentences:,.2f} -- "
                       f"{tokens_per_sec:,.0f} tokens/s -- "
                       f"gradient norm = {grad_norm:.2f}")
                summary_writer.add_scalar("train/loss",
                                          total_train_loss/num_sentences, step)
                num_tokens = 0
                tokens_start = time.time()
                total_train_loss = 0.
                num_sentences = 0

            # Zero the gradient buffer.
            optimizer.zero_grad()

            # Run evaluation every evaluate_every steps if set.
            if hparams.evaluate_every > 0 and step > 0 and step % hparams.evaluate_every == 0:
                run_evaluation()

            step += 1

        print(f"Finished epoch {epoch_num}")

        # If evaluate_every is not set, we evaluate after every epoch.
        if hparams.evaluate_every <= 0:
            run_evaluation()

        epoch_num += 1

    print(f"Finished training.")
    summary_writer.close()

    # Load the best model and run validation again, make sure to not write
    # summaries.
    model.load_state_dict(torch.load(best_model_location))
    print(f"Loaded best model found at step {best_step} (epoch {best_epoch}).")
    model.eval()
    validate(model, val_data, val_alignments, vocab_src, vocab_tgt, device, hparams, step,
             summary_writer=None)

def main(hparams):

    # Print hyperparameter values.
    print("\n==== Hyperparameters")
    hparams.print_values()

    # Load the data and print some statistics.
    train_data, val_data, val_alignments = load_data(hparams)
    vocab_src, vocab_tgt = load_vocabularies(hparams)
    if hparams.share_vocab:
        print("\n==== Vocabulary")
        vocab_src.print_statistics()
    else:
        print("\n==== Source vocabulary")
        vocab_src.print_statistics()
        print("\n==== Target vocabulary")
        vocab_tgt.print_statistics()
    print("\n==== Data")
    print(f"Training data: {len(train_data):,} bilingual sentence pairs")
    print(f"Validation data: {len(val_data):,} bilingual sentence pairs")

    # Create the model.
    model, train_fn, validate_fn = create_model(hparams, vocab_src, vocab_tgt)
    optimizer, lr_scheduler = create_optimizer(model.parameters(), hparams)
    device = torch.device("cuda:0") if hparams.use_gpu else torch.device("cpu")
    model = model.to(device)

    # Print information about the model.
    param_count_M = model_parameter_count(model) / 1e6
    print("\n==== Model")
    print("Short summary:")
    print(model)
    print("\nAll parameters:")
    for name, param in model.named_parameters():
        print(f"{name} -- {param.size()}")
    print(f"\nNumber of model parameters: {param_count_M:.2f} M")

    # Initialize the model parameters.
    if hparams.model_checkpoint is None:
        print("\nInitializing parameters...")
        initialize_model(model, vocab_tgt[PAD_TOKEN], hparams.cell_type,
                         hparams.emb_init_scale, verbose=True)
    else:
        print(f"\nRestoring model parameters from {hparams.model_checkpoint}...")
        model.load_state_dict(torch.load(hparams.model_checkpoint))

    # Create the output directory.
    out_dir = Path(hparams.output_dir)
    if not out_dir.exists():
        out_dir.mkdir()
    if hparams.vocab_prefix is None:
        vocab_src.save(out_dir / f"vocab.{hparams.src}")
        vocab_tgt.save(out_dir / f"vocab.{hparams.tgt}")
        hparams.vocab_prefix = out_dir / "vocab"
    hparams.save(out_dir / "hparams")
    print("\n==== Output")
    print(f"Created output directory at {hparams.output_dir}")

    # Train the model.
    print("\n==== Starting training")
    print(f"Using device: {device}\n")
    train(model, optimizer, lr_scheduler, train_data, val_data, val_alignments, vocab_src,
          vocab_tgt, device, out_dir, train_fn, validate_fn, hparams)

if __name__ == "__main__":
    hparams = Hyperparameters()
    main(hparams)
