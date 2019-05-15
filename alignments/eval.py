import torch

from pathlib import Path

from alignments.aer import read_naacl_alignments
from alignments.hparams import Hyperparameters
from alignments.train import create_model
from alignments.train_utils import load_vocabularies
from alignments.data import ParallelDataset

def main():

    # Load command line hyperparameters (and if provided from an hparams_file).
    hparams = Hyperparameters(check_required=False)
    if hparams.validation_prefix is None or hparams.src is None or hparams.tgt is None or hparams.output_dir is None:
        raise Exception("Missing argument: src, tgt, validation_prefix or output_dir")

    # Fill in any missing values from the hparams file in the output_dir.
    output_dir = Path(hparams.output_dir)
    hparams_file = output_dir / "hparams"
    hparams.update_from_file(hparams_file, override=False)
    print(f"\nLoading hyperparameters from {hparams_file}\n")
    hparams.print_values()
    print()

    # Load vocabularies, set to vocab files in output dir if no other is provided. They
    # are always two separate vocabulary file (for source and target) independent of the
    # value of hparams.share_vocab. If hparams.share_vocab was True originally, they will
    # simply be the same.
    if hparams.vocab_prefix is None:
        hparams.vocab_prefix = output_dir / "vocab"
        hparams.share_vocab = False
    vocab_src, vocab_tgt = load_vocabularies(hparams)

    print("\n==== Source vocabulary")
    vocab_src.print_statistics()

    print("\n==== Target vocabulary")
    vocab_tgt.print_statistics()

    # Select the correct device (GPU or CPU).
    device = torch.device("cuda:0") if hparams.use_gpu else torch.device("cpu")

    # Restore the model from output_dir/model.pt.
    model_checkpoint = output_dir / "model.pt"
    print(f"\nRestoring model from {model_checkpoint}\n")
    model, _, eval_fn = create_model(hparams, vocab_src, vocab_tgt)
    model.load_state_dict(torch.load(model_checkpoint))
    model = model.to(device)
    model.eval()

    # Load the data.
    val_src = f"{hparams.validation_prefix}.{hparams.src}"
    val_tgt = f"{hparams.validation_prefix}.{hparams.tgt}"
    val_data = ParallelDataset(val_src, val_tgt)
    gold_alignments = read_naacl_alignments(f"{hparams.validation_prefix}.wa.nonullalign")

    aer = eval_fn(model, val_data, gold_alignments, vocab_src, vocab_tgt, device,
                   hparams, 0, summary_writer=None)

if __name__ == "__main__":
    main()
