import torch
import torch.nn as nn
from mmodel.basic_module import WeightedModule
import math
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0,1)
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    
    
    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(DEVICE)
        return self.mu + self.sigma * epsilon
    
    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


# 先验
class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0,sigma1)
        self.gaussian2 = torch.distributions.Normal(0,sigma2)
    
    def log_prob(self, inputs):
        # inputs = inputs.cpu()
        prob1 = torch.exp(self.gaussian1.log_prob(inputs))
        prob2 = torch.exp(self.gaussian2.log_prob(inputs))
        return (torch.log(self.pi * prob1 + (1-self.pi) * prob2)).sum()



PI = 0.25
SIGMA_1 = torch.cuda.FloatTensor([math.exp(-0)])
SIGMA_2 = torch.cuda.FloatTensor([math.exp(-6)])


class BayesianLinear(WeightedModule):
    def __init__(self, in_features, out_features):
        #继承父类属性
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 权重w的参数
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

        # Bias 的参数
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)

        # 先验分布
        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.log_prior = 0
        self.log_variational_posterior = 0

        self.has_init = True

    def forward(self, input, sample=False, calculate_log_probs=False):

        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu

        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)


class BayesianNetwork(WeightedModule):
    # 两层隐藏层
    def __init__(self, param):
        super().__init__()
        self.l1 = BayesianLinear(28 * 28, 400)
        self.l2 = BayesianLinear(400, 400)
        self.l3 = BayesianLinear(400, 10)
        self.param = param
        self.has_init = True

    def forward(self, x, sample=False):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = F.log_softmax(self.l3(x, sample), dim=1)
        return x

    def log_prior(self):
        return self.l1.log_prior + self.l2.log_prior + self.l3.log_prior

    def log_variational_posterior(self):
        return (
            self.l1.log_variational_posterior
            + self.l2.log_variational_posterior
            + self.l2.log_variational_posterior
        )

    def sample_elbo(self, input, target, samples=5):
        param = self.param
        outputs = torch.zeros(samples, param.batch_size, param.class_num).to(DEVICE)
        log_priors = torch.zeros(samples).to(DEVICE)
        log_variational_posteriors = torch.zeros(samples).to(DEVICE)
        for i in range(samples):
            outputs[i] = self(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[
                i
            ] = self.log_variational_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.nll_loss(
            outputs.mean(0), target, size_average=False
        )
        loss = (
            log_variational_posterior - log_prior
        ) / param.class_num + negative_log_likelihood
        return (
            loss,
            log_prior,
            log_variational_posterior,
            negative_log_likelihood,
        )

