import torch
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.transforms import v2

# Convert images to float32 tensors scaled to [0, 1], then normalize with FashionMNIST mean/std
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.2860,), (0.3530,)),
])

# Download FashionMNIST (60,000 training images, 10,000 test images)
full_train_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)

# Split the 60,000 training images into 55,000 train / 5,000 validation
train_dataset, val_dataset = random_split(
    full_train_dataset, [55000, 5000],
    generator=torch.Generator().manual_seed(42),
)

print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")
