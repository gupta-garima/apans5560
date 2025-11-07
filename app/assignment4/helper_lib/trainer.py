import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_energy(model, loader, device="cpu", epochs=5, lr=1e-4):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Energy Epoch {epoch+1}/{epochs}")
        total_energy = 0
        
        for x, _ in pbar:
            x = x.to(device)
            opt.zero_grad()
            e = model(x).mean()
            e.backward()
            opt.step()
            
            total_energy += e.item()
            pbar.set_postfix({"energy": f"{e.item():.4f}"})
        
        avg_energy = total_energy / len(loader)
        print(f"Epoch {epoch+1} - Avg Energy: {avg_energy:.4f}")
    
    return model

class DiffusionConfig:
    def __init__(self, T=1000):
        self.T = T
        self.beta = torch.linspace(1e-4, 0.02, T)
        self.alpha = 1 - self.beta
        self.abar = torch.cumprod(self.alpha, dim=0)

def train_diffusion(model, loader, device="cpu", epochs=3, T=1000, lr=2e-4):
    cfg = DiffusionConfig(T)
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    
    # Move config tensors to device
    cfg.beta = cfg.beta.to(device)
    cfg.alpha = cfg.alpha.to(device)
    cfg.abar = cfg.abar.to(device)
    
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Diffusion Epoch {epoch+1}/{epochs}")
        total_loss = 0
        
        for x, _ in pbar:
            x = x.to(device)
            b = torch.randint(0, T, (x.size(0),), device=device)
            abar = cfg.abar[b].view(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            xt = torch.sqrt(abar)*x + torch.sqrt(1-abar)*noise
            pred = model(xt, b)
            loss = nn.MSELoss()(pred, noise)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
    
    return model, cfg

def train_gan(gen, disc, loader, device="cpu", epochs=5, z_dim=128, lr=2e-4):
    gen = gen.to(device)
    disc = disc.to(device)
    opt_g = optim.Adam(gen.parameters(), lr=lr, betas=(0.5,0.999))
    opt_d = optim.Adam(disc.parameters(), lr=lr, betas=(0.5,0.999))
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"GAN Epoch {epoch+1}/{epochs}")
        total_loss_d = 0
        total_loss_g = 0
        
        for x, _ in pbar:
            x = x.to(device)
            bsz = x.size(0)
            real = torch.ones(bsz, 1, device=device)
            fake = torch.zeros(bsz, 1, device=device)

            z = torch.randn(bsz, z_dim, device=device)
            fake_imgs = gen(z)

            # Train Discriminator
            opt_d.zero_grad()
            d_real = disc(x)
            d_fake = disc(fake_imgs.detach())
            loss_d = loss_fn(d_real, real) + loss_fn(d_fake, fake)
            loss_d.backward()
            opt_d.step()

            # Train Generator
            opt_g.zero_grad()
            d_fake2 = disc(fake_imgs)
            loss_g = loss_fn(d_fake2, real)
            loss_g.backward()
            opt_g.step()
            
            total_loss_d += loss_d.item()
            total_loss_g += loss_g.item()
            pbar.set_postfix({
                "D_loss": f"{loss_d.item():.4f}",
                "G_loss": f"{loss_g.item():.4f}"
            })
        
        avg_d = total_loss_d / len(loader)
        avg_g = total_loss_g / len(loader)
        print(f"Epoch {epoch+1} - D Loss: {avg_d:.4f}, G Loss: {avg_g:.4f}")

    return gen, disc