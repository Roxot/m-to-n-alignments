import torch.optim as optim
import numpy as np
import sacrebleu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from alignments.data import Vocabulary, ParallelDataset, remove_subword_tokens
from alignments.aer import read_naacl_alignments

def load_data(hparams):
    train_src = f"{hparams.training_prefix}.{hparams.src}"
    train_tgt = f"{hparams.training_prefix}.{hparams.tgt}"
    val_src = f"{hparams.validation_prefix}.{hparams.src}"
    val_tgt = f"{hparams.validation_prefix}.{hparams.tgt}"
    val_alignments = read_naacl_alignments(f"{hparams.validation_prefix}.wa.nonullalign")

    # Load the parallel datasets.
    training_data = ParallelDataset(train_src, train_tgt, max_length=hparams.max_sentence_length,
                                    min_length=hparams.min_sentence_length)
    val_data = ParallelDataset(val_src, val_tgt)

    return training_data, val_data, val_alignments

def load_vocabularies(hparams):
    train_src = f"{hparams.training_prefix}.{hparams.src}"
    train_tgt = f"{hparams.training_prefix}.{hparams.tgt}"
    val_src = f"{hparams.validation_prefix}.{hparams.src}"
    val_tgt = f"{hparams.validation_prefix}.{hparams.tgt}"

    # Construct the vocabularies.
    if hparams.vocab_prefix is not None:

        if hparams.share_vocab:
            vocab = Vocabulary.from_file(hparams.vocab_prefix,
                                         max_size=hparams.max_vocabulary_size)
            vocab_src = vocab
            vocab_tgt = vocab
        else:
            vocab_src_file = f"{hparams.vocab_prefix}.{hparams.src}"
            vocab_tgt_file = f"{hparams.vocab_prefix}.{hparams.tgt}"
            vocab_src = Vocabulary.from_file(vocab_src_file,
                                             max_size=hparams.max_vocabulary_size)
            vocab_tgt = Vocabulary.from_file(vocab_tgt_file,
                                             max_size=hparams.max_vocabulary_size)
    else:

        if hparams.share_vocab:
            vocab = Vocabulary.from_data([train_src, train_tgt, val_src, val_tgt], 
                                         min_freq=hparams.vocab_min_freq,
                                         max_size=hparams.max_vocabulary_size)
            vocab_src = vocab
            vocab_tgt = vocab
        else:
            vocab_src = Vocabulary.from_data([train_src, val_src],
                                             min_freq=hparams.vocab_min_freq,
                                             max_size=hparams.max_vocabulary_size)
            vocab_tgt = Vocabulary.from_data([train_tgt, val_tgt],
                                             min_freq=hparams.vocab_min_freq,
                                             max_size=hparams.max_vocabulary_size)

    return vocab_src, vocab_tgt

def create_optimizer(parameters, hparams):
    optimizer = optim.Adam(parameters, lr=hparams.learning_rate)

    # Create the learning rate scheduler.
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        mode="min", # minimize AER
                                                        factor=hparams.lr_reduce_factor,
                                                        patience=hparams.lr_reduce_patience,
                                                        verbose=False,
                                                        threshold=1e-2,
                                                        threshold_mode="abs",
                                                        cooldown=hparams.lr_reduce_cooldown,
                                                        min_lr=hparams.min_lr)
    return optimizer, lr_scheduler

def model_parameter_count(model, tag=None):
    return sum(p.numel() for name, p in model.named_parameters() if (p.requires_grad and (tag is None or tag in name)))

def gradient_norm(model, tag=None):
    total_norm = 0.
    for name, p in model.named_parameters():
        if tag is not None and tag not in name:
            continue

        if p.grad is None:
            print(f"WARNING: p.grad is None for parameter with size {p.size()}")
            continue

        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2

    total_norm = np.sqrt(total_norm)
    return total_norm

def alignment_summary(src_labels, tgt_labels, alignment_links, summary_writer, summary_name,
                      global_step):
    """
    :param src_labels: list of T_src source word tokens (strings)
    :param tgt_labels: list of T_tgt target word tokens (strings)
    :param alignment_links: list of 1-indexed alignments links as (src_pos, tgt_pos)
    """

    # Create a matrix of alignments
    A = np.zeros([len(src_labels), len(tgt_labels)])
    for link in alignment_links:
        src_pos, tgt_pos = link
        A[src_pos-1, tgt_pos-1] = 1.

    # Plot a heatmap for the scores.
    fig, ax = plt.subplots()
    plt.imshow(A, cmap="Greys", aspect="equal",
               origin="upper", vmin=0., vmax=1.)

   # Configure the columns.
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(A.shape[1]))
    ax.set_xticklabels(tgt_labels, rotation="vertical")

    # Configure the rows.
    ax.set_yticklabels(src_labels)
    ax.set_yticks(np.arange(A.shape[0]))

    # Fit the figure neatly.
    plt.tight_layout()

    # Write the summary.
    summary_writer.add_figure(summary_name, fig, global_step=global_step)
