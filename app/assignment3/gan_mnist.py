
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

# -------- Device selection (CUDA > MPS > CPU) ----------
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print("Device:", DEVICE)

# -------- Models (exactly per spec) ----------
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.fc = nn.Linear(z_dim, 7*7*128)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # 7->14
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),    # 14->28
            nn.Tanh()
        )
    def forward(self, z):
        x = self.fc(z).view(-1, 128, 7, 7)
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),             # 28->14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),           # 14->7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Linear(128*7*7, 1)  # logits
    def forward(self, x):
        h = self.features(x).view(x.size(0), -1)
        return self.classifier(h)

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

@torch.no_grad()
def save_grid(G, path, z_dim=100, n=16, device=DEVICE):
    G.eval()
    z = torch.randn(n, z_dim, device=device)
    imgs = G(z).cpu()
    grid = utils.make_grid(imgs, nrow=int(n**0.5), normalize=True, value_range=(-1,1), pad_value=1.0)
    utils.save_image(grid, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--betas", type=float, nargs=2, default=(0.5, 0.999))
    ap.add_argument("--z_dim", type=int, default=100)
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--num_workers", type=int, default=0)  # 0 is safest on macOS
    args = ap.parse_args()

    out_dir = Path("models"); out_dir.mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # [-1,1] for Tanh
    ])
    ds = datasets.MNIST(root=args.data_root, train=True, download=True, transform=tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    G = Generator(args.z_dim).to(DEVICE); G.apply(weights_init)
    D = Discriminator().to(DEVICE); D.apply(weights_init)

    bce = nn.BCEWithLogitsLoss()
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=tuple(args.betas))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=tuple(args.betas))

    try:
        for epoch in range(1, args.epochs + 1):
            for real, _ in dl:
                real = real.to(DEVICE)
                bs = real.size(0)

                # --- D ---
                optD.zero_grad(set_to_none=True)
                real_logit = D(real)
                d_real = bce(real_logit, torch.ones_like(real_logit))

                z = torch.randn(bs, args.z_dim, device=DEVICE)
                fake = G(z).detach()
                fake_logit = D(fake)
                d_fake = bce(fake_logit, torch.zeros_like(fake_logit))

                d_loss = d_real + d_fake
                d_loss.backward(); optD.step()

                # --- G ---
                optG.zero_grad(set_to_none=True)
                z = torch.randn(bs, args.z_dim, device=DEVICE)
                gen = G(z)
                g_loss = bce(D(gen), torch.ones_like(real_logit))
                g_loss.backward(); optG.step()

            # save checkpoint + sample each epoch
            torch.save(G.state_dict(), out_dir / f"gen_epoch_{epoch:03d}.pth")
            save_grid(G, out_dir / f"samples_epoch_{epoch:03d}.png", z_dim=args.z_dim, device=DEVICE)
            print(f"[Epoch {epoch}/{args.epochs}] D={d_loss.item():.3f} | G={g_loss.item():.3f}")

        # final save
        torch.save(G.state_dict(), out_dir / "gen.pth")
        save_grid(G, out_dir / "final_grid.png", z_dim=args.z_dim, device=DEVICE)
        print("Saved: models/gen.pth and models/final_grid.png")

    except KeyboardInterrupt:
        # save progress if you stop early
        torch.save(G.state_dict(), out_dir / "gen_interrupt.pth")
        print("\nInterrupted. Partial weights saved to models/gen_interrupt.pth")

if __name__ == "__main__":
    main()
