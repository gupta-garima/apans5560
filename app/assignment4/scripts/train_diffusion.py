import torch
from app.assignment4.helper_lib.data import get_cifar10_loaders
from app.assignment4.helper_lib.model import get_model
from app.assignment4.helper_lib.trainer import train_diffusion
from app.assignment4.helper_lib.generator import ddpm_sample

def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, _ = get_cifar10_loaders(batch_size=128, num_workers=0)
    denoiser = get_model("diffusion")
    denoiser, cfg = train_diffusion(denoiser, train_loader, device=device, epochs=3, T=1000)
    torch.save(denoiser.state_dict(), "app/assignment4/checkpoints/diffusion.pt")
    imgs = ddpm_sample(denoiser, cfg, shape=(8,3,32,32), device=device)
    print("Saved -> app/assignment4/checkpoints/diffusion.pt", float(imgs.mean()), float(imgs.std()))

if __name__ == "__main__":
    main()
