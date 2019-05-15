import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

class NeuralIBM1(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, emb_size, hidden_size, pad_idx):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.pad_idx = pad_idx
        self.src_embedder = nn.Embedding(src_vocab_size, emb_size, padding_idx=pad_idx)
        self.translation_layer = nn.Sequential(nn.Linear(emb_size, hidden_size),
                                               nn.ReLU(),
                                               nn.Linear(hidden_size, tgt_vocab_size),
                                               nn.Softmax(dim=-1))

    def forward(self, x, seq_mask_x, seq_len_x, y):
        x_embed = self.src_embedder(x) # [B, T_x, emb_size]

        # Compute p(y_j|x_i, a_j) for all x_i in x.
        batch_size = x_embed.size(0)
        longest_x = x_embed.size(1)
        py_given_xa = self.translation_layer(x_embed.view(batch_size * longest_x, self.emb_size))
        py_given_xa = py_given_xa.view(batch_size, longest_x, self.tgt_vocab_size)

        # P(a_1^T_y|l, m) = 1 / (T_x + 1) -- note that the NULL word is added to x,
        # seq_mask_x and seq_len_x already.
        p_align = seq_mask_x.type_as(x_embed) / seq_len_x.unsqueeze(-1).type_as(x_embed) # [B, T_x]

        # Tile p_align to [B, T_y, T_x].
        longest_y = y.size(1)
        p_align = p_align.unsqueeze(1)
        p_align = p_align.repeat(1, longest_y, 1)

        # Compute the marginal p(y|x)
        p_marginal = torch.bmm(p_align, py_given_xa)

        return p_marginal

    def loss(self, p_marginal, y, reduction="mean"):
        p_observed = torch.gather(p_marginal, -1, y.unsqueeze(-1))
        p_observed = p_observed.squeeze(-1)
        log_likelihood = torch.log(p_observed).sum(dim=1)

        loss = -log_likelihood
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
        else:
            raise Exception(f"Unknown reduction option: {reduction}")

    def align(self, x, y):
        """
            Returns argmax_a P(a|x,y)
        """

        with torch.no_grad():

            # Compute P(y|x, a)
            x_embed = self.src_embedder(x)
            batch_size = x_embed.size(0)
            longest_x = x_embed.size(1)
            py_given_xa = self.translation_layer(x_embed.view(batch_size * longest_x, self.emb_size))
            py_given_xa = py_given_xa.view(batch_size, longest_x, self.tgt_vocab_size) # [B, T_x, V_y]

            longest_y = y.size(1)
            alignments = np.zeros([batch_size, longest_y], dtype=np.int)

            # Take the argmax_a P(y|x, a) for each y_j. Note that we can do this as the
            # alignment probabilities are constant and the alignments are independent.
            # I.e., this is identical to argmax_a P(a|x, f).
            for batch_idx, y_n in enumerate(y):
                for j, y_j in enumerate(y_n):
                    if y_j == self.pad_idx:
                        break
                    p = py_given_xa[batch_idx, :, y_j]
                    alignments[batch_idx, j] = np.argmax(p.cpu())

        return alignments
