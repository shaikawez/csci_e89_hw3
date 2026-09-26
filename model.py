from torch import nn

from neural_net import device


class ImageClassifier(nn.Module):
    def __init__(self, n_features, n_hidden1, n_hidden2, n_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, n_hidden1),  # input -> hidden layer 1
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),  # hidden layer 1 -> hidden layer 2
            nn.ReLU(),
            nn.Linear(n_hidden2, n_classes),  # hidden layer 2 -> output, no activation (raw logits)
        )

    def forward(self, x):
        return self.layers(x)


# FashionMNIST: 28x28 grayscale images, 10 clothing classes
model = ImageClassifier(n_features=28 * 28, n_hidden1=300, n_hidden2=100, n_classes=10).to(device)
print(model)
