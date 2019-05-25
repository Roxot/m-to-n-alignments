import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torchdist
import numpy as np

from alignments.constants import epsilon
from alignments.dist import BernoulliREINFORCE, BernoulliStraightThrough, BinaryConcrete, Kumaraswamy
from alignments.dist import StretchedAndRectifiedDistribution
from alignments.components import RNNEncoder

class InferenceNetwork(nn.Module):

    def __init__(self, dist, src_vocab_size, tgt_vocab_size, emb_size, hidden_size, pad_idx,
                 bidirectional, num_layers, cell_type):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.pad_idx = pad_idx
        self.dist = dist
        self.src_embedder = nn.Embedding(src_vocab_size, emb_size, padding_idx=pad_idx)
        self.tgt_embedder = nn.Embedding(tgt_vocab_size, emb_size, padding_idx=pad_idx)
        self.src_encoder = RNNEncoder(emb_size=emb_size,
                                      hidden_size=hidden_size,
                                      bidirectional=bidirectional,
                                      dropout=0.,
                                      num_layers=num_layers,
                                      cell_type=cell_type)
        self.tgt_encoder = RNNEncoder(emb_size=emb_size,
                                      hidden_size=hidden_size,
                                      bidirectional=bidirectional,
                                      dropout=0.,
                                      num_layers=num_layers,
                                      cell_type=cell_type)

        encoding_size = hidden_size * 2 if bidirectional else hidden_size
        # encoding_size = emb_size
        if self.dist in ["kuma", "hardkuma"]:
            self.kuma_a_key_layer = nn.Linear(encoding_size, hidden_size)
            self.kuma_a_query_layer = nn.Linear(encoding_size, hidden_size)
            # self.kuma_b_key_layer = nn.Linear(encoding_size, hidden_size)
            # self.kuma_b_query_layer = nn.Linear(encoding_size, hidden_size)
        else:
            self.key_layer = nn.Linear(encoding_size, hidden_size)
            self.query_layer = nn.Linear(encoding_size, hidden_size)

    def forward(self, x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y):

        # Embed the source and target words.
        x_embed = self.src_embedder(x) # [B, T_x, emb_size]
        y_embed = self.tgt_embedder(y) # [B, T_y, emb_size]

        # Encode both sentences.
        x_enc, _ = self.src_encoder.unsorted_forward(x_embed) # [B, T_x, enc_size]
        y_enc, _ = self.tgt_encoder.unsorted_forward(y_embed)  # [B, T_y, enc_size]
        # x_enc = x_embed
        # y_enc = y_embed

        if self.dist in ["bernoulli-RF", "bernoulli-ST", "concrete"]:

            # compute keys and queries.
            keys = self.key_layer(x_enc) # [B, T_x, hidden_size]
            queries = self.query_layer(y_enc) # [B, T_y, hidden_size]

            # Compute the scores as dot attention between source and target.
            logits = torch.bmm(queries, keys.transpose(1, 2)) # [B, T_y, T_x]

            if self.dist == "bernoulli-RF":
                return BernoulliREINFORCE(logits=logits, validate_args=True)
            elif self.dist == "bernoulli-ST":
                return BernoulliStraightThrough(logits=logits, validate_args=True)
            elif self.dist == "concrete":
                logits = torch.clamp(logits, -5., 5.)
                return BinaryConcrete(temperature=logits.new([1.0]), logits=logits, validate_args=True) # TODO

        elif self.dist in ["kuma", "hardkuma"]:

            # Compute a using attention.
            keys_a = self.kuma_a_key_layer(x_enc) # [B, T_x, hidden_size]
            queries_a = self.kuma_a_query_layer(y_enc) # [B, T_y, hidden_size]
            a = torch.bmm(queries_a, keys_a.transpose(1, 2)) # [B, T_y, T_x]
            # a = torch.clamp(F.softplus(a) + 0.7, 1e-5, 3.) # [B, T_y, T_x]
            # a = torch.tanh(a) + 1.1 # (0.1, 2.1)
            a = 0.01 + (0.98 * torch.sigmoid(a))

            # Compute b using attention.
            # keys_b = self.kuma_b_key_layer(x_enc) # [b, t_x, hidden_size]
            # queries_b = self.kuma_b_query_layer(y_enc) # [b, t_y, hidden_size]
            # b = torch.bmm(queries_b, keys_b.transpose(1, 2)) # [B, T_y, T_x]
            # # b = torch.clamp(F.softplus(b) + 0.7, 1e-5, 3.) # [B, T_y, T_x]
            # b = torch.tanh(b) + 1.1 # (0.1, 2.1)

            # q = Kumaraswamy(a, b)
            q = Kumaraswamy(a, 1.0 - a)
            if self.dist == "kuma":
                return q
            else:
                return StretchedAndRectifiedDistribution(q, lower=-0.1, upper=1.1)
        else:
            raise Exception(f"Unknown dist option: {self.dist}")

class AlignmentVAE(nn.Module):

    def __init__(self, dist, prior_params, src_vocab_size, tgt_vocab_size, emb_size, hidden_size,
                 pad_idx, pooling, bidirectional, num_layers, cell_type, max_sentence_length):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.emb_size = emb_size
        self.pad_idx = pad_idx
        self.dist = dist
        self.pooling = pooling
        self.prior_params = prior_params
        self.src_embedder = nn.Embedding(src_vocab_size, emb_size, padding_idx=pad_idx)
        self.categorical_layer = nn.Linear(emb_size, tgt_vocab_size)
        self.inf_network = InferenceNetwork(dist=dist,
                                            src_vocab_size=src_vocab_size,
                                            tgt_vocab_size=tgt_vocab_size,
                                            emb_size=emb_size,
                                            hidden_size=hidden_size,
                                            pad_idx=pad_idx,
                                            bidirectional=bidirectional,
                                            num_layers=num_layers,
                                            cell_type=cell_type)

        if dist == "bernoulli-RF":
            self.register_buffer("avg_learning_signal", torch.Tensor([0.]))
            # self.register_buffer("std_learning_signal", torch.Tensor([1.]))
            self.alpha = 0.05

        if dist == "hardkuma":
                self._create_hardkuma_prior_table(prior_params[0], max_sentence_length)

    def prior(self, seq_mask_x, seq_len_x, seq_mask_y):
        """
            Prior 1 / src_length.
        """

        prior_param_1, prior_param_2 = self.prior_params
        prior_shape = [seq_mask_x.size(0), seq_mask_y.size(1), seq_mask_x.size(1)]

        if "bernoulli" in self.dist:
            if prior_param_1 > 0:
                # prior_param_1 words per sentence
                probs = prior_param_1 * (seq_mask_x.float() + epsilon) / (seq_len_x.unsqueeze(-1).float() + 1)
                probs = torch.clamp(probs, max=(1-epsilon))
                probs = probs.unsqueeze(1).repeat(1, seq_mask_y.size(1), 1)
            elif prior_param_2 > 0:
                # fixed prior_param_2 probability of an alignment
                probs = seq_mask_x.float().new_full(prior_shape, fill_value=prior_param_2)
            else:
                raise Exception(f"Invalid prior params for Bernoulli ({prior_param_1}, {prior_param_2})")

            return BernoulliREINFORCE(probs=probs, validate_args=True) # [B, T_y, T_x]
        elif self.dist == "concrete":
            return BinaryConcrete(temperature=probs.new([1.0]), logits=torch.zeros_like(probs)) # TODO
        elif self.dist in ["kuma", "hardkuma"]:

            if prior_param_1 > 0 and prior_param_2 > 0:
                p = Kumaraswamy(seq_mask_x.float().new_full(prior_shape, fill_value=prior_param_1),
                                seq_mask_x.float().new_full(prior_shape, fill_value=prior_param_2))
            elif self.dist == "hardkuma" and prior_param_1 > 0:
                seq_len_numpy = seq_len_x.cpu().numpy()
                a = seq_len_x.float().new_tensor([self.hardkuma_prior_table[length][0] for length in seq_len_numpy]) # [B]
                a = a.unsqueeze(-1).unsqueeze(-1).repeat(1, seq_mask_y.size(1), seq_mask_x.size(1))
                b = torch.ones_like(a)
                p = Kumaraswamy(a, b)
            else:
                raise Exception(f"Invalid Kumaraswamy parameters a={prior_param_1}, b={prior_param_2}")

            if self.dist == "kuma":
                return p
            else:
                return StretchedAndRectifiedDistribution(p, -0.1, 1.1)

    def _create_hardkuma_prior_table(self, prior_param_1, max_sentence_length, l=-0.1, r=1.1, N=10000):
        """
        Creates a prior table for the HardKuma. Fixes b=1.0
        """
        kuma_priors = []
        l = torch.Tensor([l])
        r = torch.Tensor([r])

        # Create a list of (a, 1.0, CDF(0))
        with torch.no_grad():
            b = torch.Tensor([1.0])
            for a in torch.linspace(start=epsilon, end=1., steps=N):
                pk = Kumaraswamy(a, b)

                # Compute the position of 0 in the stretched distribution.
                k0 = -l / (r - l)
                kuma_priors.append((a.item(), b.item(), pk.cdf(k0).item()))
        kuma_priors = sorted(kuma_priors, key=lambda elem: elem[0])

        # Tabulate priors for every possible sentence length.
        # P(0) = 1 - (prior_param_1 / (l+1))
        self.hardkuma_prior_table = {}
        for length in range(1, max_sentence_length+1):
            p0 = 1.0 - min(((1.0 + epsilon) / (length + 1.0)), 1.0 - epsilon)
            idx = 0
            cdf0 = float("inf")
            while cdf0 > p0 and idx != len(kuma_priors):
                a, b, cdf0 = kuma_priors[idx]
                idx += 1
            self.hardkuma_prior_table[length] = (a, b)

    # def _get_hardkuma_prior_a(self, p0):
    #     """
    #     Returns the HardKuma a parameter that assigns approximately p0 probability to 0
    #     if b = 1.0.
    #     """
    #     idx = 0
    #     cdf0 = float("inf")
    #     while cdf0 > p0 and idx != len(self.kuma_priors):
    #         a, b, cdf0 = self.kuma_priors[idx]
    #         idx += 1
    #     return a

    def approximate_posterior(self, x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y):
        return self.inf_network(x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y)

    def forward(self, x, A):

        # Tile the embeddings.
        x_embed = self.src_embedder(x)  # [B, T_x, emb_size]
        x_embed = x_embed.unsqueeze(1).repeat(1, A.size(1), 1, 1) # [B, T_y, T_x, emb_size]

        # Perform average pooling of the embeddings according to A.
        pooled_x = x_embed * A.unsqueeze(-1) # [B, T_y, T_x, emb_size] * [B, T_y, T_x, 1]

        if self.pooling == "avg":
            # Average pooling
            pooled_x = pooled_x.sum(dim=2) / \
                    (A.unsqueeze(-1).sum(dim=2) + epsilon) # [B, T_y, emb_size]
        else:
            # Sum pooling
            pooled_x = pooled_x.sum(dim=2) # [B, T_y, emb_size]

        # Compute the categorical logits.
        logits = self.categorical_layer(pooled_x)
        return logits

    def _update_baselines(self, new_learning_signal): # [B]
        new_learning_signal = new_learning_signal.detach()
        self.avg_learning_signal = self.alpha * new_learning_signal.mean() \
                                   + (1.0 - self.alpha) * self.avg_learning_signal
        # self.std_learning_signal = self.alpha * new_learning_signal.std() \
        #                            + (1.0 - self.alpha) * self.std_learning_signal

    def loss(self, logits, y, A, seq_mask_x, seq_mask_y, pa, qa, KL_multiplier=1.0, reduction="mean"):
        """
        :param pa: prior distribution.
        :param qa: distribution used to sample a.
        """
        output_dict = {}

        # Compute the negative complete data log-likelihood for each batch element.
        # Logits are of the form [B, T_y, vocab_size_y], whereas the cross-entropy
        # function wants a loss of the form [B, vocab_size_y, T_y].
        logits = logits.permute(0, 2, 1)
        neg_log_py_xa = F.cross_entropy(logits, y, ignore_index=self.pad_idx,
                                        reduction="none") # [B, T_y]
        # neg_log_py_xa = neg_log_py_xa.sum(dim=1) # [B]

        # Compute the KL between the prior and the posterior distributions.
        KL = torchdist.kl.kl_divergence(qa, pa)

        # Mask out padding positions.
        KL = torch.where(seq_mask_x.unsqueeze(-1).transpose(1, 2), KL, KL.new([0.]))
        KL = torch.where(seq_mask_y.unsqueeze(-1), KL, KL.new([0.]))

        # Sum for all independent latent alignment variables.
        KL = KL.sum(dim=-1).sum(dim=-1) # [B]
        output_dict["KL"] = KL

        # The loss is the negative ELBO, where ELBO = E_qa[log P(y|x, a)] - KL(qa||pa)
        loss = neg_log_py_xa.sum(dim=-1) + KL_multiplier * KL
        output_dict["ELBO"] = -loss + KL_multiplier * KL - KL

        # For REINFORCE, compute a surrogate term for the REINFORCE estimator for d/d lambda.
        if self.dist == "bernoulli-RF":
            learning_signal = -neg_log_py_xa.detach() # [B, T_y]
            normalized_learning_signal = (learning_signal - self.avg_learning_signal.unsqueeze(-1))
            #                             / self.std_learning_signal.unsqueeze(-1)
            #  std > 0
            log_qa_sample = qa.log_prob(A).sum(dim=-1) # [B, T_y]
            loss = loss - (normalized_learning_signal * log_qa_sample).sum(dim=-1) # [B]

            # Update the baselines.
            seq_len_y = seq_mask_y.sum(dim=-1).type_as(learning_signal)
            mean_baseline = learning_signal.sum(dim=-1) / seq_len_y # [B]
            self._update_baselines(mean_baseline)

            output_dict["learning_signal"] = mean_baseline
        output_dict["loss"] = loss

        # Do sum over the time dimension if reduction is none.
        if reduction == "mean":
            for k, v in output_dict.items():
                output_dict[k] = v.mean()
        elif reduction == "sum":
            for k, v in output_dict.items():
                output_dict[k] = v.sum()
        elif reduction != "none":
            raise Exception(f"Unknown reduction option {reduction}")

        return output_dict
