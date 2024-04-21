import torch

class MultiDimDirichletProcess(nn.Module):
    def __init__(self, F_dims, H_dims, K, alpha=1.0):
        super(MultiDimDirichletProcess, self).__init__()
        self.F_dims = F_dims
        self.H_dims = H_dims
        self.K = K
        self.alpha = alpha

        # Multi-dimensional PMFs for F and H
        self.pmf_F = torch.ones(*F_dims) / torch.prod(torch.tensor(F_dims))
        self.pmf_H = torch.ones(*H_dims) / torch.prod(torch.tensor(H_dims))

    def stick_breaking(self):
        # Compute weights deterministically using Beta expectation
        weights = torch.empty(self.K)
        beta_expectation = 1 / (1 + self.alpha)
        weights[0] = beta_expectation
        for k in range(1, self.K):
            weights[k] = beta_expectation * torch.prod(1.0 - weights[:k])
        return weights

    def posterior(self, f_obs, h_obs=None):
        # Compute the likelihood of the observed data for F
        likelihood_F = self.pmf_F[tuple(f_obs)]

        # If h_obs is provided, compute its likelihood as well
        if h_obs is not None:
            likelihood_H = self.pmf_H[tuple(h_obs)]
        else:
            likelihood_H = torch.ones_like(self.pmf_H)

        # Multiply the likelihoods with the prior (base distributions)
        posterior_F = likelihood_F * self.pmf_F
        posterior_H = likelihood_H * self.pmf_H

        # Normalize to ensure it's a valid distribution
        posterior_F /= torch.sum(posterior_F)
        posterior_H /= torch.sum(posterior_H)

        return posterior_F, posterior_H

    def posterior_predictive(self, f_obs):
        _, posterior_H = self.posterior(f_obs)
        return posterior_H


# Initialize model for testing
dp_model_multi = MultiDimDirichletProcess(F_dims=(5, 5, 5), H_dims=(10, 10), K=10)

# Compute stick-breaking weights
weights_multi = dp_model_multi.stick_breaking()

# Test posterior estimation
f_obs_multi = torch.tensor([2, 3, 1])
h_obs_multi = torch.tensor([7, 5])
posterior_F_multi, posterior_H_multi = dp_model_multi.posterior(f_obs_multi, h_obs_multi)

# Test posterior predictive
post_pred_multi = dp_model_multi.posterior_predictive(f_obs_multi)

weights_multi, posterior_F_multi.shape, posterior_H_multi.shape, post_pred_multi.shape
