import torch

from torch.utils.data import DataLoader

from alignments.aer import AERSufficientStatistics
from alignments.models import AlignmentVAE
from alignments.data import PAD_TOKEN, create_batch

def create_model(hparams, vocab_src, vocab_tgt):
    return AlignmentVAE(dist=hparams.model_type,
                        prior_params=(hparams.prior_param_1, hparams.prior_param_2),
                        src_vocab_size=vocab_src.size(),
                        tgt_vocab_size=vocab_tgt.size(),
                        emb_size=hparams.emb_size,
                        hidden_size=hparams.hidden_size,
                        pad_idx=vocab_tgt[PAD_TOKEN],
                        pooling=hparams.pooling,
                        bidirectional=hparams.bidirectional,
                        num_layers=hparams.num_layers,
                        cell_type=hparams.cell_type,
                        max_sentence_length=hparams.max_sentence_length,
                        use_mean_cv=hparams.cv_running_avg,
                        use_std_cv=hparams.cv_running_std,
                        use_self_critic_cv=hparams.cv_self_critic)

def train_step(model, x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y, hparams, step,
               summary_dict, summary_writer=None):
    qa = model.approximate_posterior(x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y)
    pa = model.prior(seq_mask_x, seq_len_x, seq_mask_y)
    A = qa.rsample()
    logits = model(x, A)

    if hparams.KL_annealing_steps > 0:
        KL_multiplier = min(1.0, float(step) / hparams.KL_annealing_steps)
    else:
        KL_multiplier = 1.0

    output_dict = model.loss(logits=logits, x=x, y=y, A=A, seq_mask_x=seq_mask_x,
                             seq_mask_y=seq_mask_y, pa=pa, qa=qa,
                             KL_multiplier=KL_multiplier, reduction="mean")
    output_dict["A"] = A
    output_dict["qa"] = qa
    output_dict["pa"] = qa
    output_dict["KL_multiplier"] = KL_multiplier

    # Keep track of training summary statistics.
    summary_dict["num_sentences"] += x.size(0)
    summary_dict["KL"] += output_dict["KL"].sum().item()
    summary_dict["ELBO"] += output_dict["ELBO"].sum().item()
    summary_dict["reward"] += output_dict["reward"].sum().item()
    summary_dict["normalized_reward"] += output_dict["normalized_reward"].sum().item()
    summary_dict["reward_var"] += output_dict["reward"].var().item()
    summary_dict["normalized_reward_var"] += output_dict["normalized_reward"].var().item()
    if "reward_sc" in output_dict:
        summary_dict["reward_sc"] += output_dict["reward_sc"].sum().item()

    # Summarize if the summary writer is given.
    if summary_writer is not None:
        summary_writer.add_scalar("train/KL", summary_dict["KL"] / summary_dict["num_sentences"], step)
        summary_writer.add_scalar("train/ELBO", summary_dict["ELBO"] / summary_dict["num_sentences"], step)
        summary_writer.add_scalar("train/reward", summary_dict["reward"] / summary_dict["num_sentences"], step)
        summary_writer.add_scalar("train/reward_var", summary_dict["reward_var"] /\
                summary_dict["num_sentences"], step)
        summary_writer.add_scalar("train/normalized_reward", summary_dict["normalized_reward"] /\
                summary_dict["num_sentences"], step)
        summary_writer.add_scalar("train/normalized_reward_var", summary_dict["normalized_reward_var"] /\
                summary_dict["num_sentences"], step)
        summary_writer.add_scalar("train/reward_mean_ma", model.avg_reward, step)
        summary_writer.add_scalar("train/reward_std_ma", model.std_reward, step)
        summary_writer.add_histogram("train/p(A)", pa.probs, step)
        summary_writer.add_histogram("train/q(A|x,y)", qa.probs, step)
        summary_writer.add_histogram("train/sampled_A", A, step)
        if "reward_sc" in output_dict:
            summary_writer.add_scalar("train/reward_self_critic", summary_dict["reward_sc"] /\
                    summary_dict["num_sentences"], step)

    return output_dict

def validate(model, val_data, gold_alignments, vocab_src, vocab_tgt, device,
             hparams, step, summary_writer=None):

    model.eval()

    # Load the validation data.
    val_dl = DataLoader(val_data, shuffle=False, batch_size=hparams.batch_size,
                        num_workers=4)

    total_correct_predictions = 0
    total_predictions = 0.
    num_sentences = 0
    total_ELBO = 0.
    total_KL = 0.
    alignments = []
    with torch.no_grad():
        for sen_x, sen_y in val_dl:

            # Infer the mean A | x, y.
            x, seq_mask_x, seq_len_x = create_batch(sen_x, vocab_src, device, include_null=False)
            y, seq_mask_y, seq_len_y = create_batch(sen_y, vocab_tgt, device)
            qa = model.approximate_posterior(x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y)
            if "bernoulli" in hparams.model_type:
                A = qa.mean.round() # [B, T_y, T_x]
            elif hparams.model_type == "hardkuma":
                zeros = torch.zeros_like(qa.base.a)
                ones = torch.ones_like(qa.base.a)
                p0 = qa.log_prob(zeros)
                p1 = qa.log_prob(ones)
                # pc = ones - p0 - p1
                A = torch.where(p0 > p1, zeros, ones)
                # A = torch.where(p0 > pc, A, ones) # only 0 if argmax(p0, p1, pc) = p0
            else:
                raise NotImplementedError()

            # Store the alignment links. A link is (src_word, tgt_word), don't store null alignments. Sentences
            # start at 1 (1-indexed).
            for len_x_k, len_y_k, A_k in zip(seq_len_x, seq_len_y, A):
                links = set()
                for j, aj in enumerate(A_k[:len_y_k], 1):
                    for i, aji in enumerate(aj[:len_x_k], 1):
                        if aji > 0:
                            links.add((i, j))
                alignments.append(links)

            # Compute validation ELBO and KL.
            logits = model(x, qa.sample())
            pa = model.prior(seq_mask_x, seq_len_x, seq_mask_y)
            output_dict = model.loss(logits=logits, x=x, y=y, A=A, seq_mask_x=seq_mask_x,
                                     seq_mask_y=seq_mask_y, pa=pa, qa=qa)
            total_ELBO += output_dict["ELBO"].sum().item()
            total_KL += output_dict["KL"].sum().item()
            num_sentences += x.size(0)

            # Compute statistics for validation accuracy.
            logits = model(x, A)
            predictions = torch.argmax(logits, dim=-1, keepdim=False)
            correct_predictions = (predictions == y) * seq_mask_y
            total_correct_predictions += correct_predictions.sum().item()
            total_predictions += seq_len_y.sum().item()

    val_ELBO = total_ELBO / num_sentences
    val_KL = total_KL / num_sentences

    # Compute AER.
    metric = AERSufficientStatistics()
    for a, gold_a in zip(alignments, gold_alignments):
        metric.update(sure=gold_a[0], probable=gold_a[1], predicted=a)
    val_aer = metric.aer()

    # Compute translation accuracy.
    val_accuracy = total_correct_predictions / total_predictions

    # Write validation summaries if a summary writer is given.
    if summary_writer is not None:
        # summary_writer.add_scalar("validation/NLL", val_NLL, step)
        # summary_writer.add_scalar("validation/perplexity", val_ppl, step)
        summary_writer.add_scalar("validation/KL", val_KL, step)
        summary_writer.add_scalar("validation/ELBO", val_ELBO, step)
        summary_writer.add_scalar("validation/AER", val_aer, step)
        summary_writer.add_scalar("validation/accuracy", val_accuracy, step)

    # Print validation results.
    print(f"validation accuracy = {val_accuracy:.2f} -- validation AER = {val_aer:.2f}"
          f" -- validation ELBO = {val_ELBO:,.2f} -- validation KL = {val_KL:,.2f}")

    return val_aer
