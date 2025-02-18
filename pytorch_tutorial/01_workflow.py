import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

print(torch.__version__)

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# print(X, y)

# Create training and test set
train_split = int(0.8 * len(X))

X_train, y_train = X[:train_split], y[:train_split]

X_test, y_test = X[train_split:], y[train_split:]

# print(len(X_train), len(X_test), len(y_train), len(y_test))


# visualize the data
def plot_predictions(
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    predictions=None,
):
    """
    Plots training data, test data, and compares predictions
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Test data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Prediction data")

    plt.legend(prop={"size": 14})

    plt.show()


# plot_predictions


# define model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(1, requires_grad=True, dtype=torch.float)
        )
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias


torch.manual_seed(42)

model_0 = LinearRegressionModel()
# print(list(model_0.parameters()))
# print(model_0.state_dict())

# use inference_mode to perform faster computations.
# might also see torch.no_grad()

# with torch.inference_mode():
#     y_predictions = model_0(X_test)

# plot_predictions(predictions=y_predictions)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)


epochs = 300

epoch_count = []
loss_values = []
test_loss_values = []

for epoch in range(epochs):
    model_0.train()

    # perform forward pass
    y_pred = model_0(X_train)

    # calculate loss
    loss = loss_fn(y_pred, y_train)

    # optimizer zero grad (resets the step because it will accumulate)
    optimizer.zero_grad()

    # perform backpropagation
    loss.backward()

    # step optimizer (gradient descent)
    optimizer.step()

    # set to eval() to turn off grad
    model_0.eval()
    with torch.inference_mode():
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred, y_test)

        # keep track of loss
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
            print(model_0.state_dict())
            epoch_count.append(epoch)
            loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())

# plot loss curves
plt.plot(epoch_count, loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.show()
