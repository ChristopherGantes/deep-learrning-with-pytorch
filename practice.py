import torch
from torch import nn
import matplotlib.pyplot as plt

weight = 0.7
bias = 0.3
X = torch.arange(0, 1, 0.02, dtype=float).unsqueeze(dim=1)
y = weight * X + bias

split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]


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


# plot_predictions()


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

model = LinearRegressionModel()

epochs = 300
lr = 0.01

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()

    predictions = model(X_train)

    loss = loss_fn(predictions, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, y_test)

        if epoch % 10 == 0:
            print(f"epoch: {epoch + 1} | loss: {loss} | test loss: {test_loss}\n")
