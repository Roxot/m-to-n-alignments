# Authors: Nicola de Cao & Wilker Aziz

import torch
from torch.distributions.kl import register_kl


@register_kl(torch.distributions.Bernoulli, torch.distributions.Bernoulli)
def _kl_bernoulli_bernoulli(p, q):
    t1 = p.probs * (torch.nn.functional.softplus(-q.logits) - torch.nn.functional.softplus(-p.logits))
    t1[q.probs == 0] = float('inf')
    t1[p.probs == 0] = 0
    t2 = (1 - p.probs) * (torch.nn.functional.softplus(q.logits) - torch.nn.functional.softplus(p.logits))
    t2[q.probs == 1] = float('inf')
    t2[p.probs == 1] = 0
    return t1 + t2
    

class BernoulliStraightThrough(torch.distributions.Bernoulli):

    def rsample(self, sample_shape=torch.Size()):
        return (super(BernoulliStraightThrough, self).sample(sample_shape) - self.probs).detach() + self.probs

    
class BernoulliREINFORCE(torch.distributions.Bernoulli):
    
    def rsample(self, sample_shape=torch.Size()):
        return self.sample(sample_shape)


# class Bernoulli:
    
#     def __init__(self, logits):
#         raise ValueError('Stop using this')
#         self._logits = logits
#         self._dist = torch.distributions.Bernoulli(logits=logits)
        
#     def params(self):
#         return [self._logits]
    
#     def paramdict(self):
#         return {'logits': self._logits}
        
#     def rsample(self, n=None):
#         return self._dist.rsample(torch.Size([n]) if n else torch.Size())
        
#     def sample(self, n=None):
#         return self._dist.sample(torch.Size([n]) if n else torch.Size())
    
#     def log_prob(self, x):
#         return self._dist.log_prob(x)
    
#     def log_cdf(self, x):
#         return self._dist.log_cdf(x)
    
#     def kl(self, other: 'Bernoulli'):
#         return torch.distributions.kl.kl_divergence(self._dist, other._dist)
