import pyro
import pyro.distributions as dist
import pyro.infer as infer
import pyro.optim as optim
import torch


class StochasticProcessModel:
    """
    Represents a stochastic process in a multi-dimensional coordinate space,
    where each dimension may represent variables such as time, action, etc.

    The model is designed to predict outputs based on inputs and also infer
    inputs based on observed outputs using Bayesian inference.

    Attributes:
        priors (list of pyro.distributions): The prior distributions for each dimension of the coordinate space.
        params (list of floats, optional): The learned parameters after training. Defaults to None.

    Example:
        >>> priors = [dist.Normal(0, 1) for _ in range(4)]
        >>> model = StochasticProcessModel(priors)
    """

    def __init__(self, priors, conditional_fn=None):
        """
        Initializes the StochasticProcessModel with the provided priors and a function to determine conditional priors.

        Args:
            priors (list of pyro.distributions): The prior distributions for each dimension of the coordinate space.
            conditional_fn (callable, optional): A function that returns the appropriate prior based on a condition. Defaults to None.
        """
        self.priors = priors
        self.params = None
        self.conditional_fn = conditional_fn

    def model(self, input_distributions, output_samples, condition=None):
        """
        Defines the generative process of the model.

        This function represents the likelihood P(o|i), where o is the output and i is the input.
        The relationship between inputs and outputs is assumed to be linear in this basic model.

        Args:
            input_distributions (list of pyro.distributions): The distributions representing input samples.
            output_samples (list of torch.Tensor): The actual observed output samples.

        Returns:
            list of torch.Tensor: The generated outputs based on the sampled inputs and the model parameters.
        """
        # Determine the prior based on the condition if a conditional function is provided
        theta_priors = self.conditional_fn(condition) if self.conditional_fn else self.priors
        theta = [pyro.sample(f"theta_{i}", prior) for i, prior in enumerate(theta_priors)]
        sampled_inputs = [input_dist.sample() for input_dist in input_distributions]
        generated_output = [sampled_input * param for sampled_input, param in zip(sampled_inputs, theta)]

        for i, (gen_out, actual_out) in enumerate(zip(generated_output, output_samples)):
            pyro.sample(f"obs_{i}", dist.Normal(gen_out, 1.0), obs=actual_out)

        return generated_output

    def guide(self, input_distributions, output_samples, condition=None):

        """
        The variational guide function for the model.

        It represents the approximate posterior distribution P(i|o), where i is the input and o is the observed output.

        Args:
            input_distributions (list of pyro.distributions): The distributions representing input samples.
            output_samples (list of torch.Tensor): The actual observed output samples.
        """
        guide_means = [pyro.param(f"guide_mean_{i}", torch.tensor(1.0)) for i in range(len(self.priors))]
        guide_vars = [pyro.param(f"guide_var_{i}", torch.tensor(1.0), constraint=dist.constraints.positive) for i in
                      range(len(self.priors))]

        for i, (mean, var) in enumerate(zip(guide_means, guide_vars)):
            pyro.sample(f"theta_{i}", dist.Normal(mean, var))

    def train(self, input_distributions, output_samples, conditions, num_steps=5000):
        """
        Trains the model using the provided input distributions and output samples.

        Args:
            input_distributions (list of pyro.distributions): The distributions representing input samples.
            output_samples (list of torch.Tensor): The actual observed output samples.
            num_steps (int, optional): The number of training iterations. Defaults to 5000.
        """
        optimizer = optim.Adam({"lr": 0.001})
        svi = infer.SVI(self.model, self.guide, optimizer, loss=infer.Trace_ELBO())

        for _ in range(num_steps):
            for input_distributions, output_samples, condition in zip(input_distributions_list, output_samples_list,
                                                                      conditions):
                svi.step(input_distributions, output_samples, condition)

    def predict(self, input_distributions, condition=None):
        """
        Predicts the outputs based on the provided input distributions and the learned model parameters.

        Args:
            input_distributions (list of pyro.distributions): The distributions representing input samples.

        Returns:
            list of torch.Tensor: The predicted outputs.
        """
        # Use Pyro's predictive functionality to get the posterior predictive distribution
        from pyro.infer import Predictive

        predictive = Predictive(self.model, guide=self.guide, num_samples=1000, return_sites=("obs_*",))
        samples = predictive(input_distributions, None, condition)  # No observed data during prediction

        # Average over all samples to get the prediction
        predictions = [samples[f"obs_{i}"].mean(0) for i in range(len(input_distributions))]

        return predictions

    def infer(self, output_samples):
        """
        Infers the input distributions based on the observed output samples using the trained model.

        Args:
            output_samples (list of torch.Tensor): The observed output samples.

        Returns:
            list of torch.Tensor: The inferred inputs.
        """
        # Running the guide to adjust our beliefs about the inputs based on the observed outputs
        self.guide(None, output_samples)

        # Sample from the updated input distributions
        inferred_inputs = [
            pyro.sample(f"inferred_input_{i}", dist.Normal(pyro.param(f"guide_mean_{i}"), pyro.param(f"guide_var_{i}")))
            for i in range(len(self.priors))]

        return inferred_inputs


# Our conditional function
def conditional_fn(condition):
    if condition == "weekday":
        return [dist.Normal(1, 0.1)]
    elif condition == "weekend":
        return [dist.Normal(2, 0.1)]
    else:
        return [dist.Normal(1, 0.5)]  # Default

# Initialize our model with the conditional function
model = StochasticProcessModel([dist.Normal(1, 0.5)], conditional_fn=conditional_fn)

# Example training data:
input_distributions_list = [[dist.Normal(5, 1)] for _ in range(10)]
output_samples_list = [torch.tensor([5.0 * 1.5]) for _ in range(5)] + [torch.tensor([5.0 * 2.5]) for _ in range(5)]
conditions = ["weekday"] * 5 + ["weekend"] * 5

# Train the model
model.train(input_distributions_list, output_samples_list, conditions, num_steps=1000)

# Predict for a new condition
new_condition = "weekday"
new_input_distribution = [dist.Normal(7, 1)]
predicted_output = model.predict(new_input_distribution, condition=new_condition)

print(predicted_output)




