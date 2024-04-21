import torch
from DeepBayesian3 import StochasticProcessNeuralModel, encode_dataset


# Assuming the model has been defined and instantiated as `model`
# and your data loaders are `train_loader` and `val_loader`

def test_latent_space_consistency(model, x):
    encoded = model.encode(x)
    decoded = model.decode(encoded)
    reconstruction_error = torch.norm(decoded - x, p=2).item()
    assert reconstruction_error < 0.1, "High reconstruction error detected!"
    print("Latent Space Consistency Test Passed!")


def test_model_capacity(model, train_loader, val_loader):
    train_loss = sum([model.loss(x, y).item() for x, y in train_loader]) / len(train_loader.dataset)
    val_loss = sum([model.loss(x, y).item() for x, y in val_loader]) / len(val_loader.dataset)
    assert train_loss - val_loss < 0.1, "Potential overfitting detected!"
    print("Model Capacity Test Passed!")


def test_posterior_consistency(model, x):
    posterior = model.encode(x)
    kl_div = torch.distributions.kl.kl_divergence(posterior, model.prior(x.shape[0]))
    assert kl_div.mean().item() < 0.1, "Posterior is inconsistent with the prior!"
    print("Posterior Consistency Test Passed!")


def test_reconstruction_confidence(model, x):
    encoded = model.encode(x)
    samples = [model.decode(encoded).detach() for _ in range(5)]
    avg_sample = sum(samples) / len(samples)
    variance = sum([(s - avg_sample) ** 2 for s in samples]) / len(samples)
    assert variance.mean().item() < 0.1, "High variance in reconstructions detected!"
    print("Reconstruction Confidence Test Passed!")


def test_predictive_distribution_calibration(model, val_loader):
    model.eval()
    total_error = 0.0
    total_uncertainty = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            output_dist = model.predict(x)
            predicted_means = output_dist.mean
            error = torch.norm(y - predicted_means, p=2).item()
            uncertainty = output_dist.variance.mean().item()

            total_error += error
            total_uncertainty += uncertainty

    avg_error = total_error / len(val_loader.dataset)
    avg_uncertainty = total_uncertainty / len(val_loader.dataset)

    assert abs(avg_error - avg_uncertainty) < 0.1, "Mismatch between error and uncertainty detected!"
    print("Predictive Distribution Calibration Test Passed!")


def test_gradient_check(model, data_loader, optimizer):
    model.train()
    x, y = next(iter(data_loader))
    optimizer.zero_grad()
    loss = model.loss(x, y)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None!"
        grad_norm = param.grad.norm().item()
        assert grad_norm > 1e-5, f"Vanishing gradient detected for {name}!"
        assert grad_norm < 1e+5, f"Exploding gradient detected for {name}!"

    print("Gradient Check Test Passed!")


def test_prior_sensitivity(model, x):
    model.eval()
    with torch.no_grad():
        output_original = model.predict(x)
        model.prior = torch.distributions.Normal(0.1, 1.1)  # Perturb the prior
        output_perturbed = model.predict(x)

    difference = torch.norm(output_original.mean - output_perturbed.mean, p=2).item()
    assert difference < 0.1, "High sensitivity to prior detected!"
    print("Prior Sensitivity Test Passed!")


def test_guide_vs_model(model, data_loader):
    model.eval()
    total_kl_divergence = 0.0
    with torch.no_grad():
        for x in data_loader:
            z_posterior = model.encode(x)
            z_prior = model.prior(x.shape[0])
            kl_div = torch.distributions.kl.kl_divergence(z_posterior, z_prior)
            total_kl_divergence += kl_div.sum().item()

    avg_kl_divergence = total_kl_divergence / len(data_loader.dataset)
    assert avg_kl_divergence < 0.1, "Mismatch between guide and model detected!"
    print("Guide vs Model Check Test Passed!")


def test_output_distribution_check(model, x):
    model.eval()
    with torch.no_grad():
        output_dist = model.predict(x)
        probs = output_dist.probs
        assert torch.all(probs >= 0) and torch.all(probs <= 1), "Invalid probabilities detected!"
        assert abs(probs.sum(dim=1) - 1.0) < 1e-5, "Probabilities do not sum to 1!"
    print("Output Distribution Check Passed!")




# Creating the dataset and model
input_dim, output_dim = 10, 10
train_dataset, val_dataset = create_dataset(input_dim, output_dim)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
model = StochasticProcessNeuralModel(input_dim, output_dim)

# Running the tests
test_latent_space_consistency(model, next(iter(train_loader))[0])
test_model_capacity(model, train_loader, val_loader)
test_posterior_consistency(model, next(iter(train_loader))[0])
test_reconstruction_confidence(model, next(iter(train_loader))[0])
test_predictive_distribution_calibration(model, val_loader)
test_gradient_check(model, train_loader, optimizer)
test_prior_sensitivity(model, next(iter(train_loader))[0])
test_guide_vs_model(model, train_loader)
test_output_distribution_check(model, next(iter(train_loader))[0])
