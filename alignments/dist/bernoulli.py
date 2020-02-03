import torch

class BernoulliStraightThrough(torch.distributions.Bernoulli):

    def rsample(self, sample_shape=torch.Size()):
        return (super(BernoulliStraightThrough, self).sample(sample_shape) - self.probs).detach() + self.probs


class BernoulliREINFORCE(torch.distributions.Bernoulli):

    def rsample(self, sample_shape=torch.Size()):
        return self.sample(sample_shape)
