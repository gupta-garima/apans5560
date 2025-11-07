import math, torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEnergy(nn.Module):
    def __init__(self, img_ch=3, base=64):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(img_ch, base, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(base, base, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(base, base*2, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(base*2, base*4, 3, 2, 1), nn.ReLU(True),
        )
        self.h = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base*4*4*4, 256), nn.ReLU(True),
            nn.Linear(256, 1),
        )
    def forward(self, x): return self.h(self.f(x))

def sinusoidal_time_embedding(t, dim=128):
    device = t.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / half)
    return torch.cat([torch.sin(t[:,None]*freqs[None]), torch.cos(t[:,None]*freqs[None])], dim=1)

class ResidualBlock(nn.Module):
    def __init__(self, ch, t_dim):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(ch)
        self.bn2 = nn.BatchNorm2d(ch)
        self.c1  = nn.Conv2d(ch, ch, 3, 1, 1)
        self.c2  = nn.Conv2d(ch, ch, 3, 1, 1)
        self.t   = nn.Linear(t_dim, ch)
    def forward(self, x, t):
        h = F.relu(self.bn1(x)); h = self.c1(h); h = h + self.t(t)[:, :, None, None]
        h = F.relu(self.bn2(h)); h = self.c2(h)
        return x + h

class TinyUNet(nn.Module):
    def __init__(self, img_ch=3, base=64, t_dim=128):
        super().__init__()
        self.inp = nn.Conv2d(img_ch, base, 3, 1, 1)
        self.tproj = nn.Sequential(nn.Linear(t_dim, base), nn.ReLU(), nn.Linear(base, base))
        self.d1 = nn.Conv2d(base, base*2, 4, 2, 1); self.rb1 = ResidualBlock(base*2, base)
        self.d2 = nn.Conv2d(base*2, base*4, 4, 2, 1); self.rb2 = ResidualBlock(base*4, base)
        self.m1 = ResidualBlock(base*4, base); self.m2 = ResidualBlock(base*4, base)
        self.u1 = nn.ConvTranspose2d(base*4, base*2, 4, 2, 1); self.rb3 = ResidualBlock(base*2, base)
        self.u2 = nn.ConvTranspose2d(base*2, base, 4, 2, 1);   self.rb4 = ResidualBlock(base, base)
        self.out = nn.Conv2d(base, img_ch, 3, 1, 1)
    def forward(self, x, t):
        t = self.tproj(sinusoidal_time_embedding(t))
        h0 = self.inp(x); d1 = self.rb1(self.d1(h0), t); d2 = self.rb2(self.d2(d1), t)
        m  = self.m2(self.m1(d2, t), t)
        u1 = self.rb3(self.u1(m), t); u2 = self.rb4(self.u2(u1), t)
        return self.out(u2)

class GANGenerator(nn.Module):
    def __init__(self, z_dim=128, img_ch=3, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, base*8, 4, 1, 0),
            nn.BatchNorm2d(base*8), nn.ReLU(True),
            nn.ConvTranspose2d(base*8, base*4, 4, 2, 1),
            nn.BatchNorm2d(base*4), nn.ReLU(True),
            nn.ConvTranspose2d(base*4, base*2, 4, 2, 1),
            nn.BatchNorm2d(base*2), nn.ReLU(True),
            nn.ConvTranspose2d(base*2, base, 4, 2, 1),
            nn.BatchNorm2d(base), nn.ReLU(True),
            nn.ConvTranspose2d(base, img_ch, 3, 1, 1),
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z.view(z.size(0), z.size(1), 1, 1))

class GANDiscriminator(nn.Module):
    def __init__(self, img_ch=3, base=64):
        super().__init__()
        self.net = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(img_ch, base, 4, 2, 1), 
            nn.LeakyReLU(0.2, True),
            
            # 16x16 -> 8x8
            nn.Conv2d(base, base*2, 4, 2, 1), 
            nn.BatchNorm2d(base*2), 
            nn.LeakyReLU(0.2, True),
            
            # 8x8 -> 4x4
            nn.Conv2d(base*2, base*4, 4, 2, 1), 
            nn.BatchNorm2d(base*4), 
            nn.LeakyReLU(0.2, True),
            
            # 4x4 -> 1x1
            nn.Conv2d(base*4, 1, 4, 1, 0)
        )
    
    def forward(self, x):
        return self.net(x).view(x.size(0), 1)

def get_model(name, **kw):
    name = name.lower()
    if name in ("energy", "ebm", "energy_model"): return ConvEnergy(**kw)
    if name in ("ddpm_unet", "diffusion"): return TinyUNet(**kw)
    if name in ("gan_gen", "gan_generator"): return GANGenerator(**kw)
    if name in ("gan_disc", "gan_discriminator"): return GANDiscriminator(**kw)
    raise ValueError(f"Unknown model {name}")