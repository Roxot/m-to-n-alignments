import re
import torch
import numpy as np

from .constants import UNK_TOKEN, PAD_TOKEN, NULL_TOKEN

def create_batch(sentences, vocab, device, include_null=False, word_dropout=0.):
    """
    Converts a list of sentences to a padded batch of word ids. Returns
    a batch of word ids, a sequence mask and a tensor containing the
    sequence length of each batch element.
    :param sentences: a list of sentences, each a list of token ids
    :param vocab: a Vocabulary object for this dataset
    :param device: 
    :param word_dropout: rate at which we omit words from the context (input)
    :returns: a padded batch of word ids, mask, lengths
    """
    null_token = [NULL_TOKEN] if include_null else []
    tok = np.array([null_token + sen.split() for sen in sentences])
    seq_lengths = [len(sen) for sen in tok]
    max_len = max(seq_lengths)
    pad_id = vocab[PAD_TOKEN]
    batch = [
        [vocab[sen[t]] if t < seq_lengths[idx] else pad_id for t in range(max_len)]
            for idx, sen in enumerate(tok)]

    # Replace words of the input with <unk> with p = word_dropout.
    if word_dropout > 0.:
        unk_id = vocab[UNK_TOKEN]
        batch =  [
            [unk_id if (np.random.random() < word_dropout and t < seq_lengths[idx]) else word_ids[t] for t in range(max_len)]
                for idx, word_ids in enumerate(batch)]

    # Convert everything to PyTorch tensors.
    batch = torch.tensor(batch)
    seq_mask = (batch != vocab[PAD_TOKEN])
    seq_length = torch.tensor(seq_lengths)

    # Move all tensors to the given device.
    batch = batch.to(device)
    seq_mask = seq_mask.to(device)
    seq_length = seq_length.to(device)

    return batch, seq_mask, seq_length

def batch_to_sentences(tensors, vocab, no_filter=False):
    """
    Converts a batch of word ids back to sentences.
    :param tensors: [B, T] word ids
    :param vocab: a Vocabulary object for this dataset
    :param no_filter: whether to filter sos, eos, and pad tokens.
    :returns: an array of strings (each a sentence)
    """
    sentences = []
    batch_size = tensors.size(0)
    for idx in range(batch_size):
        sentence = [vocab.word(t.item()) for t in tensors[idx,:]]

        # Filter out the start-of-sentence and padding tokens.
        if not no_filter:
            sentence = list(filter(lambda t: t != PAD_TOKEN and t != SOS_TOKEN, sentence))

        # Remove the end-of-sentence token and all tokens following it.
        if EOS_TOKEN in sentence and not no_filter:
            sentence = sentence[:sentence.index(EOS_TOKEN)]

        sentences.append(" ".join(sentence))
    return np.array(sentences)

def remove_subword_tokens(sentences, subword_token):
    """
    Removes all subword tokens from a list of sentences. E.g. "The bro@@ wn fox ." with
    subword_token="@@" will be turned into "The brown fox .".

    :param sentences: a list of sentences
    :param subword_token: the subword token.
    """
    subword_token = subword_token.strip()
    clean_sentences = []
    for sentence in sentences:
        clean_sentences.append(re.sub(f"({subword_token} )|({subword_token} ?$)|"
                               f"( {subword_token})|(^ ?{subword_token})", "",
                               sentence))
    return clean_sentences
