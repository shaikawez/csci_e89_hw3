import torch
import torchmetrics
from torch import nn

from neural_net import device
from dataloaders import train_loader, val_loader
from model import model

N_EPOCHS = 20
LEARNING_RATE = 0.01
MOMENTUM = 0.9
N_CLASSES = 10

# CrossEntropyLoss applies softmax internally, so the model outputs raw logits
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

# Separate metric objects so training and validation accumulate independently
train_metric = torchmetrics.Accuracy(task="multiclass", num_classes=N_CLASSES).to(device)
val_metric = torchmetrics.Accuracy(task="multiclass", num_classes=N_CLASSES).to(device)


def evaluate(model, data_loader, metric):
    """Return the metric computed over the whole data loader."""
    model.eval()
    metric.reset()
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            metric.update(model(images), labels)
    return metric.compute().item()


# Per-epoch history, used later to plot accuracy
history = {"train_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(N_EPOCHS):
    model.train()
    train_metric.reset()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_metric.update(logits, labels)

    train_loss = running_loss / len(train_loader)
    train_acc = train_metric.compute().item()
    val_acc = evaluate(model, val_loader, val_metric)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch + 1}/{N_EPOCHS} | "
        f"train loss: {train_loss:.4f} | "
        f"train accuracy: {train_acc:.4f} | "
        f"valid accuracy: {val_acc:.4f}"
    )
