from src.train import train

if __name__ == "__main__":
    train(
        batch_size=2,
        epochs=15,
        lr=1e-3,
        save_path="checkpoints/best_model.pt",
    )

#I used a small batch size (2) due to CPU training.
#Trained for 15 epochs and saved the best model based on validation Dice.