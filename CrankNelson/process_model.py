import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np
from data_gen import ODE, PDE, generate_particle_box_data
from nets import DeepNet, DeepCrankNicolsonNet, DeepEulerNet, bisection, plot_image_comparison

def bisection(f, a, b, tol=1e-6, max_iter=100):
    """
    Implements the bisection root-finding method in PyTorch.

    Args:
    f: A PyTorch function for which we want to find a root.
    a, b: The interval in which to search for a root.
    tol: The desired precision.
    max_iter: Maximum number of iterations to perform.

    Returns:
    The root of the function f in the interval [a, b].
    """
    fa = f(a)
    fb = f(b)
    assert (fa * fb < 0).flatten().all(), "The function must have different signs at a and b."

    for _ in range(max_iter):
        c = (a + b) / 2
        fc = f(c)

        mask = fa * fc < 0
        a = torch.where(mask, a, c)
        b = torch.where(mask, c, b)
        fa = torch.where(mask, fa, fc)
        fb = torch.where(mask, fc, fb)

        # Check the tolerance
        if torch.max(torch.abs(b - a)) < tol:
            break

    return (a + b) / 2




class DeepCrankNicolsonNetC:
    def __init__(self, input_size, hidden_size, num_layers, delta_t, n_steps, newton_iterations=10, lr=0.1, reg_factor=0.01):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = DeepNet(input_size, hidden_size, num_layers).to(self.device)
        self.delta_t = delta_t
        self.n_steps = n_steps
        self.newton_iterations = newton_iterations
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.reg_factor = reg_factor

    def forward(self, X0):
        X = X0
        X_outputs = [X0]
        for _ in range(self.n_steps):
            X = X + self.delta_t * self.net(X)
            X_outputs.append(X)
        return torch.stack(X_outputs)

    def backward(self, Xn):
        """
        Uses the bisection method
        """

        X = Xn
        X_outputs = [Xn]
        for _ in range(self.n_steps):
            X_prev = X
            for _ in range(self.newton_iterations):  # Apply Newton's method for each time step
                f_X = self.net(X)
                X = bisection(
                    f=lambda x: x - X_prev - self.delta_t * 0.5 * (self.net(X_prev) + self.net(x)),
                    a=X_prev - 10, b=X_prev + 10, tol=1e-6, max_iter=100
                )
            X_outputs.append(X)
        return torch.stack(X_outputs[::-1])


    def compute_loss(self, X0, Xn):
        X_outputs_forward = self.forward(X0)
        Xn.requires_grad_(True)  # Ensure requires_grad is set to True
        X_outputs_backward = self.backward(Xn)

        # Loss for first and last values for forward and backward predictions
        loss_first_last_forward = self.criterion(X_outputs_forward[0], X0) + 2*self.criterion(X_outputs_forward[-1],Xn)

        # loss_first_last_backward = self.criterion(X_outputs_backward[0], Xn) + self.criterion(X_outputs_backward[-1],X0)

        reg_term = self.reg_factor * sum(p.norm() for p in self.net.parameters())

        # loss = self.criterion(X_outputs_forward, X0) + self.criterion(X_outputs_backward, Xn) + reg_term
        # return loss_first_last_backward + loss_first_last_forward + reg_term
        return loss_first_last_forward + reg_term

    def train(self, X0_train, Xn_train, X0_val, Xn_val, epochs=1000):
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            loss = self.compute_loss(X0_train, Xn_train)

            # Compute loss on train data
            loss_train = self.compute_loss(X0_train, Xn_train)
            train_losses.append(loss_train.item())

            loss_train.backward()
            self.optimizer.step()

            # Compute loss on validation data
            with torch.no_grad():
                loss_val = self.compute_loss(X0_val, Xn_val)
                val_losses.append(loss_val.item())

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Train Loss: {loss_train.item()}, Validation Loss: {loss_val.item()}')

        return loss_train.item()

    def train(self, X0, Xn, X0_val, Xn_val, epochs=1000, print_freq=1):
        self.optimizer.zero_grad()
        loss_val_old = np.inf
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            loss = self.compute_loss(X0, Xn)
            loss.backward()
            self.optimizer.step()

            if epoch % print_freq == 0:  # Print progress update every `print_freq` steps
                with torch.no_grad():
                    loss_val = self.compute_loss(X0_val, Xn_val)
                print('Epoch {}/{}, Loss: {:.4f}, Validation Loss: {:.4f}'.format(
                    epoch, epochs, loss.item(), loss_val.item()))
                #if loss_val.item() > loss_val_old:
                #    print('Validation loss increased! Early stopping.')
                #    break
                loss_val_old = loss_val.item()
        return self



def plot_errors(model, X0_test, Xn_test):
    # Make predictions
    Xn_pred_forward = model.forward(X0_test)
    Xn_pred_backward = model.backward(Xn_test)

    # Compute errors
    errors_forward = Xn_test - Xn_pred_forward
    errors_backward = Xn_test - Xn_pred_backward

    # Convert to numpy
    errors_forward_np = errors_forward.detach().numpy().flatten()
    errors_backward_np = errors_backward.detach().numpy().flatten()

    X0_test = X0_test.detach().numpy().flatten()
    Xn_test = Xn_test.detach().numpy().flatten()


    # Plot histogram of errors
    plt.hist(errors_forward_np, bins=20, alpha=0.5, label='Forward Pass Errors')
    plt.hist(errors_backward_np, bins=20, alpha=0.5, label='Backward Pass Errors')
    plt.hist(X0_test-Xn_test, bins=20, alpha=0.5, label='X0 XN difference')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Histogram of Prediction Errors')
    plt.legend(loc='upper right')
    plt.show()

# Now we use the PDE class for the training

time_steps = 10
data = "PDE"
Net = "crank"

if data  == "particle_box":
    n_variables = 10
    # Generate the training data
    X0_train, Xn_train = generate_particle_box_data(n_variables, X=100, Z=3, num_simulations=1000, num_steps=10)
    X0_val, Xn_val = generate_particle_box_data(n_variables, X=100, Z=3, num_simulations=100, num_steps=10)
    X0_test, Xn_test = generate_particle_box_data(n_variables, X=100, Z=3, num_simulations=100, num_steps=10)

if data  == "ODE":
    ode = ODE()
    W = np.random.rand(ode.n_variables)
    # Generate larger training and validation sets
    X0_train, Xn_train = ode.generate_data(W, batch_size=1000)
    X0_val, Xn_val = ode.generate_data(W, batch_size=200)
    X0_test, Xn_test = ode.generate_data(W, batch_size=200)

if data  == "PDE":
    n_variables = 10
    pde = PDE(n_variables=n_variables)
    X0_train, Xn_train, _ = pde.generate_data(n_steps=10, seed=0, batch_size=1000)
    X0_val, Xn_val, _ = pde.generate_data(n_steps=10, seed=1, batch_size=200)
    X0_test, Xn_test, _ = pde.generate_data(n_steps=10, seed=2, batch_size=200)

if data  == "CUP":
    import numpy as np
    from sklearn.model_selection import train_test_split
    import pickle

    n_variables = 106

    with open("cup_rotations", 'rb') as f:
        training_data = pickle.load(f)

    # Converting images to grayscale and reshaping them
    reshaped_data = []
    for img, viewpoint in training_data:
        grayscale_img = img.sum(axis=-1) / 3.0  # Converting to grayscale by averaging color channels
        reshaped_img = grayscale_img.reshape(-1)  # Reshaping the image to 1D
        reshaped_data.append((reshaped_img, viewpoint))

    # Preparing the input and output data
    X = np.array([np.concatenate((img, viewpoint)) for img, viewpoint in reshaped_data])  # Input data
    Xn = np.array([np.concatenate((img, viewpoint)) for img, viewpoint in reshaped_data])  # Input data for Xn

    # Extracting the view coordinates from X0 and Xn
    view_coords_X0 = X[:, -3:]  # change -3 to the number of view coordinates you have
    view_coords_Xn = Xn[:, -3:]

    # Calculating the difference
    view_diff = view_coords_Xn - view_coords_X0

    # Concatenating the difference with the view coordinates
    X = np.concatenate((X, view_diff), axis=1)
    Xn = np.concatenate((Xn, -view_diff), axis=1)

    # Splitting the data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Xn, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    # Renaming the variables according to your specifications
    X0_train, Xn_train = X_train, y_train
    X0_val, Xn_val = X_val, y_val
    X0_test, Xn_test = X_test, y_test

    X0_train = (X0_train - X0_train.mean()) / X0_train.std()
    Xn_train = (Xn_train - Xn_train.mean()) / Xn_train.std()
    X0_val = (X0_val - X0_val.mean()) / X0_val.std()
    Xn_val = (Xn_val - Xn_val.mean()) / Xn_val.std()

    X0_test = (X0_test - X0_test.mean()) / X0_test.std()
    Xn_test = (Xn_test - Xn_test.mean()) / Xn_test.std()

    # Convert numpy arrays to PyTorch tensors
    X0_train = torch.from_numpy(X0_train).float()
    Xn_train = torch.from_numpy(Xn_train).float()
    X0_val = torch.from_numpy(X0_val).float()
    Xn_val = torch.from_numpy(Xn_val).float()

    X0_test = torch.from_numpy(X0_test).float()
    Xn_test = torch.from_numpy(Xn_test).float()



X0_train = (X0_train - X0_train.mean()) / X0_train.std()
Xn_train = (Xn_train - Xn_train.mean()) / Xn_train.std()
X0_val = (X0_val - X0_val.mean()) / X0_val.std()
Xn_val = (Xn_val - Xn_val.mean()) / Xn_val.std()

X0_test = (X0_test - X0_test.mean()) / X0_test.std()
Xn_test = (Xn_test - Xn_test.mean()) / Xn_test.std()


if Net  == "euler":
    # Plot the trajectory of the first variable over time.
    net = DeepEulerNet(input_size=n_variables, hidden_size=400, num_layers=30, delta_t=1, n_steps= time_steps, newton_iterations= 10, lr=0.0005)

    # Train the network with reduced epochs
    net.train(X0_train, Xn_train, X0_val, Xn_val, epochs=300)

if Net  == "crank":
    # Initialize and train the network
    net = DeepCrankNicolsonNetC(input_size=n_variables, hidden_size=300, num_layers=6, delta_t=1, n_steps=5, lr=0.0005)
    net.train(X0_train, Xn_train, X0_val, Xn_val, epochs=500)

X_outputs_forward = net.forward(X0_test.to(net.device))
X_outputs_backward = net.backward(Xn_test.to(net.device))

plot_errors(net, X0_test, Xn_test)

# assuming X0_test contains the true images and X_outputs_forward contains the predicted images
for i in range(len(X0_test)):
    plot_image_comparison(X0_test[i,:-6], X_outputs_forward[i,:-6],X0_test[:, -6:-3], image_shape=(10, 10))



#plot_results_forward_backward_single_variable(X0_test, Xn_test, X_outputs_forward, X_outputs_backward, time_steps)
