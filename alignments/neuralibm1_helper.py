import torch
import numpy as np

from torch.utils.data import DataLoader

from alignments.aer import AERSufficientStatistics
from alignments.data import PAD_TOKEN, create_batch
from alignments.models import NeuralIBM1

def create_model(hparams, vocab_src, vocab_tgt):
    model = NeuralIBM1(src_vocab_size=vocab_src.size(),
                       tgt_vocab_size=vocab_tgt.size(),
                       emb_size=hparams.emb_size,
                       hidden_size=hparams.hidden_size,
                       pad_idx=vocab_src[PAD_TOKEN])

    return model

def train_step(model, x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y, hparams, step,
               summary_dict, summary_writer=None):
    py_given_x = model(x, seq_mask_x, seq_len_x, y)
    loss = model.loss(py_given_x, y, reduction="mean")
    return loss

def validate(model, val_data, gold_alignments, vocab_src, vocab_tgt, device,
             hparams, step, summary_writer=None):
    model.eval()

    # Load the validation data.
    val_dl = DataLoader(val_data, shuffle=False, batch_size=hparams.batch_size,
                        num_workers=4)

    num_sentences = 0
    num_predictions = 0
    total_NLL = 0.
    alignments = []
    with torch.no_grad():
        for sen_x, sen_y in val_dl:
            x, seq_mask_x, seq_len_x = create_batch(sen_x, vocab_src, device, include_null=True)
            y, _, seq_len_y = create_batch(sen_y, vocab_tgt, device)

            py_given_x = model(x, seq_mask_x, seq_len_x, y)
            batch_NLL = model.loss(py_given_x, y, reduction="sum")
            total_NLL += batch_NLL.item()
            num_sentences += x.size(0)
            num_predictions += seq_len_y.sum().item()

            # Compute the alignments.
            batch_alignments = model.align(x, y)
            for seq_len, a in zip(seq_len_y, batch_alignments):
                links = set()
                for j, aj in enumerate(a[:seq_len], 1):
                    if aj > 0:
                        links.add((aj, j))
                alignments.append(links)

    # Compute AER.
    metric = AERSufficientStatistics()
    for a, gold_a in zip(alignments, gold_alignments):
        metric.update(sure=gold_a[0], probable=gold_a[1], predicted=a)
    val_aer = metric.aer()

    # Compute NLL and perplexity.
    val_NLL = total_NLL / num_sentences
    val_ppl  = np.exp(total_NLL / num_predictions)

    # Write validation summaries if a summary writer is given.
    if summary_writer is not None:
        summary_writer.add_scalar("validation/NLL", val_NLL, step)
        summary_writer.add_scalar("validation/perplexity", val_ppl, step)
        summary_writer.add_scalar("validation/AER", val_aer, step)

    # Print validation results.
    print(f"validation NLL = {val_NLL:,.2f} -- validation ppl = {val_ppl:,.2f}"
          f" -- validation AER = {val_aer:,.2f}")

    return val_aer
