from fastapi import FastAPI
from pydantic import BaseModel
import torch

from app.assignment4.helper_lib.model import get_model
from app.assignment4.helper_lib.trainer import DiffusionConfig
from app.assignment4.helper_lib.generator import ddpm_sample

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"

ENERGY_CKPT = "app/assignment4/checkpoints/energy.pt"
DIFF_CKPT   = "app/assignment4/checkpoints/diffusion.pt"

_energy = None
_diff   = None
_cfg    = None

@app.on_event("startup")
def load_ckpts():
    global _energy, _diff, _cfg
    try:
        _energy = get_model("energy").to(device)
        _energy.load_state_dict(torch.load(ENERGY_CKPT, map_location=device))
        _energy.eval()
    except:
        _energy = None
    try:
        _diff = get_model("diffusion").to(device)
        _diff.load_state_dict(torch.load(DIFF_CKPT, map_location=device))
        _diff.eval()
        _cfg = DiffusionConfig(T=1000)
    except:
        _diff = None
        _cfg = None

@app.get("/health")
def health():
    return {"energy": _energy is not None, "diffusion": _diff is not None}

class DiffusionReq(BaseModel):
    num_samples: int = 8

@app.post("/diffusion/generate")
def diffusion_generate(req: DiffusionReq):
    if _diff is None or _cfg is None:
        return {"error": "missing diffusion model"}
    imgs = ddpm_sample(_diff, _cfg, shape=(req.num_samples,3,32,32), device=device)
    return {"num": req.num_samples, "mean": float(imgs.mean()), "std": float(imgs.std())}

class EnergyReq(BaseModel):
    batch: int = 4

@app.post("/energy/score")
def energy_score(req: EnergyReq):
    if _energy is None:
        return {"error": "missing energy model"}
    with torch.no_grad():
        x = torch.randn(req.batch, 3, 32, 32, device=device)
        e = _energy(x).squeeze().tolist()
    return {"scores": e}
