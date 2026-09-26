import matplotlib.pyplot as plt

from train import history

TRAIN_COLOR = "#2a78d6"  # blue
VAL_COLOR = "#eb6834"  # orange

epochs = range(1, len(history["train_acc"]) + 1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, history["train_acc"], color=TRAIN_COLOR, linewidth=2,
        marker="o", markersize=5, label="Training")
ax.plot(epochs, history["val_acc"], color=VAL_COLOR, linewidth=2,
        marker="s", markersize=5, label="Validation")

# Label the final value of each line directly
ax.annotate(f"{history['train_acc'][-1]:.3f}", (epochs[-1], history["train_acc"][-1]),
            xytext=(6, 0), textcoords="offset points", va="center")
ax.annotate(f"{history['val_acc'][-1]:.3f}", (epochs[-1], history["val_acc"][-1]),
            xytext=(6, 0), textcoords="offset points", va="center")

ax.set_title("FashionMNIST accuracy per epoch")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_xticks(list(epochs))
ax.grid(True, color="#e0e0e0", linewidth=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig("accuracy.png", dpi=150)
plt.show()
