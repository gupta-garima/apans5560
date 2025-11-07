# app/assignment4/helper_lib/generator.py
import torch

@torch.no_grad()
def generate_gan_samples(gen, device="cpu", num_samples=16, z_dim=128):
    gen = gen.to(device).eval()
    z = torch.randn(num_samples, z_dim, device=device)
    return gen(z).cpu()

@torch.no_grad()
def ddpm_sample(denoiser, cfg, shape=(16,3,32,32), device="cpu"):
    denoiser = denoiser.to(device).eval()
    x = torch.randn(shape, device=device)
    betas  = cfg.beta.to(device)
    alphas = cfg.alpha.to(device)
    abar   = cfg.abar.to(device)
    for t in reversed(range(cfg.T)):
        tt = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps = denoiser(x, tt)
        a  = alphas[t]
        b  = betas[t]
        ab = abar[t]
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        x = (1/torch.sqrt(a))*(x - (b/torch.sqrt(1-ab))*eps) + torch.sqrt(b)*noise
    return x.cpu()
