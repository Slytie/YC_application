import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np

from data_gen import ODE, PDE
from nets import DeepNet, DeepCrankNicolsonNet, DeepEulerNet, bisection


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

    def backward2(self, Xn):
        X = Xn

        print("X.requires_grad:", X.requires_grad)
        f_X = self.net(X)
        print("f_X.requires_grad:", f_X.requires_grad)

        X_outputs = [Xn]
        for _ in range(self.n_steps):
            X_prev = X
            for _ in range(self.newton_iterations):  # Apply Newton-Raphson method for each time step
                f_X = self.net(X)
                df_X = torch.autograd.grad(f_X.sum(), X, create_graph=True)[0]
                X = X - (f_X + self.delta_t * df_X - X_prev) * (1.0 / (df_X + self.delta_t))
                X.requires_grad_(True)
            X_outputs.append(X)
        return torch.stack(X_outputs[::-1])

    def backward(self, Xn):
        X = Xn
        X_outputs = [Xn]
        for _ in range(self.n_steps):
            Y = X.clone()
            for _ in range(self.newton_iterations):  # Apply Newton-Raphson method for each time step
                f_Y = self.net(Y)
                df_Y = torch.autograd.grad(f_Y.sum(), Y, create_graph=True)[0]
                Y = Y - (f_Y + self.delta_t * df_Y - X) / (df_Y + self.delta_t)
            X = Y
            X_outputs.append(X)
        return torch.stack(X_outputs[::-1])

    def compute_loss(self, X0, Xn):
        X_outputs_forward = self.forward(X0)
        Xn.requires_grad_(True)  # Ensure requires_grad is set to True
        X_outputs_backward = self.backward(Xn)

        # Loss for first and last values for forward and backward predictions
        loss_first_last_forward = self.criterion(X_outputs_forward[0], X0) + 4 * self.criterion(X_outputs_forward[-1],
                                                                                                Xn)
        loss_first_last_backward = self.criterion(X_outputs_backward[0], Xn) + self.criterion(X_outputs_backward[-1],
                                                                                              X0)

        reg_term = self.reg_factor * sum(p.norm() for p in self.net.parameters())

        # loss = self.criterion(X_outputs_forward, X0) + self.criterion(X_outputs_backward, Xn) + reg_term
        return loss_first_last_backward + loss_first_last_forward + reg_term

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


def plot_results_forward_backward_single_variable(X0_test, Xn_test, X_outputs_forward, X_outputs_backward, time_steps):
    n_variables = X0_test.shape[1]
    fig, axs = plt.subplots(n_variables, 1, figsize=(12, 6 * n_variables))

    for var in range(n_variables):
        # Plot the forward prediction
        axs[var].plot(range(time_steps + 1), X_outputs_forward[:, 0, var].cpu().detach().numpy(), label='Forward Prediction')

        # Plot the backward prediction
        #axs[var].plot(range(time_steps + 1), X_outputs_backward[:, 0, var].cpu().detach().numpy()[::-1], label='Backward Prediction')

        # Plot the test data
        axs[var].plot([0, time_steps], [X0_test[0, var], Xn_test[0, var]], 'ro-', label='Test data')

        axs[var].set_title(f'Variable {var + 1}')
        axs[var].set_xlabel('Time step')
        axs[var].set_ylabel('Value')
        axs[var].legend()

    plt.tight_layout()
    plt.show()

def plot_errors(model, X0_test, Xn_test):
    # Make predictions
    Xn_pred = model.forward(X0_test)
    # Compute errors
    errors = Xn_test - Xn_pred
    # Convert to numpy
    errors_np = errors.detach().numpy().flatten()
    # Plot histogram of errors
    plt.hist(errors_np, bins=50)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Histogram of Prediction Errors')
    plt.show()



def plot_image_comparison(true_tensor, pred_tensor, coordinates, image_shape):
    # Unflatten and detach the tensors, then convert to numpy arrays for plotting
    true_image = true_tensor.detach().cpu().numpy().reshape(image_shape)
    pred_image = pred_tensor.detach().cpu().numpy().reshape(image_shape)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(true_image, cmap='gray')
    axs[0].set_title('True Image')
    axs[0].set_xlabel(f'Coordinates: {coordinates}')

    axs[1].imshow(pred_image, cmap='gray')
    axs[1].set_title('Predicted Image')
    axs[1].set_xlabel(f'Coordinates: {coordinates}')

    plt.show()


# Now we use the PDE class for the training

time_steps = 10
data = "ODE"
Net = "euler"
n_variables= 20
#n_variables= 106

if data == "ODE":
    ode = ODE()
    W = np.random.rand(ode.n_variables)
    # Generate larger training and validation sets
    X0_train, Xn_train = ode.generate_data(W, batch_size=1000)
    X0_val, Xn_val = ode.generate_data(W, batch_size=200)
    X0_test, Xn_test = ode.generate_data(W, batch_size=200)

if data  == "PDE":
    pde = PDE(n_variables=n_variables)
    X0_train, Xn_train, _ = pde.generate_data(n_steps=10, seed=0, batch_size=1000)
    X0_val, Xn_val, _ = pde.generate_data(n_steps=10, seed=1, batch_size=200)
    X0_test, Xn_test, _ = pde.generate_data(n_steps=10, seed=2, batch_size=200)

if data  == "CUP":
    import numpy as np
    from sklearn.model_selection import train_test_split

    import pickle

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


if data != "CUP":
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
    net = DeepCrankNicolsonNet(input_size=n_variables, hidden_size=200, num_layers=10, delta_t=1, n_steps=time_steps, lr=0.0005)
    net.train(X0_train, Xn_train, X0_val, Xn_val, epochs=200)


X_outputs_forward = net.forward(X0_test.to(net.device))
X_outputs_backward = net.backward(Xn_test.to(net.device))

if data == "CUP":
    # assuming X0_test contains the true images and X_outputs_forward contains the predicted images
    for i in range(len(X0_test)):
        plot_image_comparison(X0_test[i,:-6], X_outputs_forward[i,:-6],X0_test[:, -6:-3], image_shape=(10, 10))

if data != "CUP":
    plot_errors(net, X0_test, Xn_test)
    plot_results_forward_backward_single_variable(X0_test, Xn_test, X_outputs_forward, X_outputs_backward, time_steps)
