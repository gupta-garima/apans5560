
import io, base64
from pathlib import Path
import torch
import torch.nn as nn
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, JSONResponse
from torchvision import utils
from PIL import Image

Z_DIM = 100
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
WEIGHTS = Path("models/gen.pth")

class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM):
        super().__init__()
        self.fc = nn.Linear(z_dim, 7*7*128)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z):
        x = self.fc(z).view(-1, 128, 7, 7)
        return self.net(x)

def denorm(x): return (x + 1) / 2

app = FastAPI(title="MNIST GAN API")
G = Generator().to(DEVICE); G.eval()
if WEIGHTS.exists():
    G.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
else:
    print("WARNING: models/gen.pth not found. Train first.")

@torch.no_grad()
@app.get("/generate")
def generate(n: int = Query(16, ge=1, le=64)):
    z = torch.randn(n, Z_DIM, device=DEVICE)
    imgs = G(z)
    grid = utils.make_grid(denorm(imgs).cpu(), nrow=int(n**0.5) or 1, padding=2)
    arr = (grid.permute(1,2,0).numpy()*255).clip(0,255).astype("uint8")
    im = Image.fromarray(arr); buf = io.BytesIO(); im.save(buf, "PNG"); buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@torch.no_grad()
@app.get("/generate_base64")
def generate_base64(n: int = Query(16, ge=1, le=64)):
    z = torch.randn(n, Z_DIM, device=DEVICE)
    imgs = G(z)
    grid = utils.make_grid(denorm(imgs).cpu(), nrow=int(n**0.5) or 1, padding=2)
    arr = (grid.permute(1,2,0).numpy()*255).clip(0,255).astype("uint8")
    im = Image.fromarray(arr); buf = io.BytesIO(); im.save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return JSONResponse({"count": n, "png_base64": b64})
