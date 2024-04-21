import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


def simulate_game(turns=100):
    # Define some snakes and ladders
    snakes = {16: 6, 47: 26, 49: 11, 56: 53, 62: 19, 64: 60, 87: 24, 93: 73, 95: 75, 98: 78}
    ladders = {1: 38, 4: 14, 9: 31, 21: 42, 28: 84, 36: 44, 51: 67, 71: 91, 80: 100}

    data = []
    position = 1  # Start at the first square

    for _ in range(turns):
        dice_roll = torch.randint(1, 7, (1,)).item()  # Dice roll between 1 and 6

        new_position = position + dice_roll
        if new_position > 100:
            new_position = position  # Player doesn't move if movement exceeds 100
        new_position = snakes.get(new_position, new_position)  # Check for snakes
        new_position = ladders.get(new_position, new_position)  # Check for ladders

        data.append((position, dice_roll, new_position - position))
        position = new_position

    return data

def model(data):
    board_size = 100
    alpha = torch.ones(3)  # Uniform prior for [normal move, ladder, snake]

    # Notice the .to_event(1)
    theta = pyro.sample("theta", dist.Dirichlet(alpha).expand_by([board_size]).to_event(1))

    for idx, (position, dice_roll, movement) in enumerate(data):
        if movement == dice_roll:
            category = torch.tensor(0)  # Normal move
        elif movement > dice_roll:
            category = torch.tensor(1)  # Ladder
        else:
            category = torch.tensor(2)  # Snake
        pyro.sample(f"obs_{idx}", dist.Categorical(theta[position - 1]), obs=category)


def guide(data):
    board_size = 100
    alpha_q = pyro.param("alpha_q", torch.ones(board_size, 3), constraint=dist.constraints.positive)
    pyro.sample("theta", dist.Dirichlet(alpha_q).to_event(1))


def infer_posterior(data, num_steps=1000):
    pyro.clear_param_store()

    svi = SVI(model, guide, Adam({"lr": 0.01}), loss=Trace_ELBO())

    for _ in range(num_steps):
        svi.step(data)

    return pyro.param("alpha_q").detach()


def generate_posterior_predictions(data, posterior, num_samples=2000):
    board_size = 100
    predictions = []

    for _ in range(num_samples):
        sampled_theta = dist.Dirichlet(posterior).sample()
        for position, dice_roll, _ in data:
            prediction = dist.Categorical(sampled_theta[position - 1]).sample().item()
            predictions.append(prediction)

    return predictions

import matplotlib.pyplot as plt
import seaborn as sns

def visualize_posterior_vs_ground_truth(data, predictions):
    board_size = 100
    ground_truth = []

    for _, dice_roll, movement in data:
        if movement == dice_roll:
            ground_truth.append(0)
        elif movement > dice_roll:
            ground_truth.append(1)
        else:
            ground_truth.append(2)

    confusion_matrix = torch.zeros(3, 3)
    for true, pred in zip(ground_truth, predictions):
        confusion_matrix[true][pred] += 1

    # Normalize the confusion matrix
    confusion_matrix /= confusion_matrix.sum(dim=1, keepdim=True)

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu", xticklabels=['Normal', 'Ladder', 'Snake'], yticklabels=['Normal', 'Ladder', 'Snake'])
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Posterior Predictions.pt vs Ground Truth')
    plt.show()


def Dest_data_generation():
    data = simulate_game(turns=100)
    assert len(data) == 100, "Number of turns mismatch."
    for (position, dice_roll, movement) in data:
        assert 1 <= dice_roll <= 6, "Invalid dice roll."
        assert 1 <= position + movement <= 100, "Invalid ending position."
        assert 1 <= position + movement <= 100, "Invalid ending position."


def Dest_model():
    data = simulate_game(turns=50)
    posterior = infer_posterior(data)
    assert posterior.shape == (100, 3), "Posterior shape mismatch."


data = simulate_game(turns=100)
posterior = infer_posterior(data)
predictions = generate_posterior_predictions(data, posterior)
visualize_posterior_vs_ground_truth(data, predictions)
