from torch.utils.data import DataLoader

from dataset import train_dataset, val_dataset, test_dataset

BATCH_SIZE = 32

# Shuffle only the training data so each epoch sees batches in a new order
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# Grab the first batch from the training loader and inspect its first image
images, labels = next(iter(train_loader))
first_image = images[0]
first_label = labels[0]

print(f"Batch images shape: {images.shape}")  # expected [32, 1, 28, 28]
print(f"First image shape: {first_image.shape}")  # expected [1, 28, 28]
print(f"First image dtype: {first_image.dtype}")  # expected torch.float32
print(f"First image label: {first_label.item()}")
