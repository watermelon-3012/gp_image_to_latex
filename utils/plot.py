import matplotlib.pyplot as plt

def plot_training_curves(model_name, train_losses_2, val_losses_2, train_acc_2, val_acc_2):
    epochs = range(1, len(train_losses_2) + 1)

    plt.figure(figsize=(14,5))

    # ---- LOSS ----
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses_2, label="Train Loss")
    plt.plot(epochs, val_losses_2, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()
    plt.grid(True)

    # ---- ACCURACY ----
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_2, label="Train Accuracy")
    plt.plot(epochs, val_acc_2, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.grid(True)

    # ---- GLOBAL TITLE ----
    plt.suptitle(f"[{model_name}] \n Training and Validation Performance Curves", fontsize=16)

    # Adjust layout so the suptitle doesn’t overlap
    plt.tight_layout(rect=[0, 0, 1, 1])

    plt.show()