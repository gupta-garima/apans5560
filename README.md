# APAN5560 — Machine Learning Engineering

A series of machine learning assignments building progressively from CNNs to generative models to reinforcement learning, all served via FastAPI.

---

## Project Structure

```
├── app/
│   ├── main.py              # FastAPI app with /classify and /embed endpoints
│   ├── embedder.py          # spaCy word/text embedding utilities
│   ├── assignment2/         # CIFAR-10 CNN image classifier
│   ├── assignment3/         # GAN for MNIST digit generation
│   ├── assignment4/         # Energy-based & Diffusion models on CIFAR-10
│   └── assignment5/         # RL fine-tuning of GPT-2
├── assignment1/             # Docker + FastAPI setup
│   ├── Dockerfile
│   └── requirements.txt
├── pyproject.toml
└── uv.lock
```

---

## Assignments

### Assignment 1 — FastAPI + Docker Setup
Containerized a FastAPI application using Docker. Established the foundation for serving ML models via REST API.

### Assignment 2 — CIFAR-10 CNN Classifier
Trained a custom CNN (`CNN64`) on the CIFAR-10 dataset (10 image classes). Exposed a `/classify` endpoint that accepts an image upload and returns the predicted class.

**Stack:** PyTorch, FastAPI, torchvision

### Assignment 3 — GAN on MNIST
Built and trained a Generative Adversarial Network (GAN) to generate handwritten digit images from the MNIST dataset. Served generated samples via API.

**Stack:** PyTorch, FastAPI

### Assignment 4 — Energy-Based & Diffusion Models
Implemented and trained:
- **Energy-Based Model (EBM)** on CIFAR-10
- **Diffusion Model** on CIFAR-10

Exposed training scripts and inference via API endpoints.

**Stack:** PyTorch, FastAPI

### Assignment 5 — RL Post-Training on GPT-2
Applied reinforcement learning (RLHF-style post-training) to fine-tune a GPT-2 language model. Containerized with Docker for deployment.

**Stack:** PyTorch, Transformers (GPT-2), Docker

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/classify` | POST | Upload an image, returns CIFAR-10 class |
| `/embed` | POST | Embed text using spaCy (word or sentence) |

---

## Setup

```bash
# Install dependencies with uv
uv sync

# Run the API
uvicorn app.main:app --reload
```
