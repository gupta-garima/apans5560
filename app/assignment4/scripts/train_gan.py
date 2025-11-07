import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from app.assignment4.helper_lib.model import get_model
from app.assignment4.helper_lib.trainer import train_gan
from app.assignment4.helper_lib.generator import generate_gan_samples

def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tfm = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    train = datasets.CIFAR10(root="data", train=True, download=True, transform=tfm)
    loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=0, pin_memory=False)
    gen = get_model("gan_gen", z_dim=128)
    disc = get_model("gan_disc")
    gen, disc = train_gan(gen, disc, loader, device=device, epochs=10, z_dim=128)
    torch.save(gen.state_dict(), "app/assignment4/checkpoints/gan_gen.pt")
    torch.save(disc.state_dict(), "app/assignment4/checkpoints/gan_disc.pt")
    imgs = generate_gan_samples(gen, device=device, num_samples=16, z_dim=128)
    print(float(imgs.mean()), float(imgs.std()))

if __name__ == "__main__":
    main()
