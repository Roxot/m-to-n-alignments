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
