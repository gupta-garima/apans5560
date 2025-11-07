import torch
from app.assignment4.helper_lib.data import get_cifar10_loaders
from app.assignment4.helper_lib.model import get_model
from app.assignment4.helper_lib.trainer import train_energy

def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, _ = get_cifar10_loaders(batch_size=128, num_workers=0)
    energy = get_model("energy")
    energy = train_energy(energy, train_loader, device=device, epochs=5)
    torch.save(energy.state_dict(), "app/assignment4/checkpoints/energy.pt")
    print("Saved -> app/assignment4/checkpoints/energy.pt")

if __name__ == "__main__":
    main()
