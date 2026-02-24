# Applied Generative AI: From Embeddings to RL-Trained LLMs

A hands-on course project building and deploying generative AI models progressively — from word embeddings and image classifiers to GANs, diffusion models, energy-based models, and reinforcement learning post-training of GPT-2. Every model is served as a live REST API using FastAPI.

---

## Model Progression

| Assignment | Model | Task |
|------------|-------|------|
| 1 | spaCy Embeddings | Word & sentence vector API |
| 2 | CNN (CIFAR-10) | Image classification API |
| 3 | GAN (MNIST) | Handwritten digit generation API |
| 4 | Diffusion + EBM (CIFAR-10) | Advanced image generation API |
| 5 | RL post-trained GPT-2 | Format-constrained text generation API |

---

## Project Structure

```
app/
  main.py              # Unified FastAPI app (/embed, /classify)
  embedder.py          # spaCy embedding utilities
  assignment1/         # Docker + environment setup
  assignment2/         # CNN architecture + CIFAR-10 classifier
  assignment3/         # GAN architecture + MNIST digit generator
  assignment4/         # Diffusion model + Energy-Based Model (CIFAR-10)
  assignment5/         # RL post-trained GPT-2 + inference API
pyproject.toml
uv.lock
```

---

## API Usage

### Word & Text Embeddings (Assignment 1)
Uses spaCy to return word or sentence embedding vectors.

```bash
# Word embedding
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "king", "mode": "word"}'

# Response
{"vectors": [[0.23, -0.11, ...]], "dim": 96}

# Sentence embedding
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "neural networks learn representations", "mode": "text"}'
```

---

### Image Classification (Assignment 2)
CNN trained on CIFAR-10 — classifies an uploaded image into one of 10 classes.

```bash
curl -X POST http://localhost:8000/classify \
  -F "file=@cat.jpg"

# Response
{"predicted_class": 3, "class_name": "cat"}
```

---

### GAN Image Generation (Assignment 3)
GAN trained on MNIST — generates handwritten digit images from random noise.

```bash
# Returns a PNG image grid of 16 generated digits
curl "http://localhost:8001/generate?n=16" --output digits.png

# Returns base64-encoded PNG
curl "http://localhost:8001/generate_base64?n=4"

# Response
{"count": 4, "png_base64": "iVBORw0KGgo..."}
```

---

### Diffusion & Energy-Based Models (Assignment 4)
DDPM diffusion model and EBM both trained on CIFAR-10.

```bash
# Generate images via diffusion (DDPM)
curl -X POST http://localhost:8002/diffusion/generate \
  -H "Content-Type: application/json" \
  -d '{"num_samples": 8}'

# Response
{"num": 8, "mean": -0.012, "std": 0.943}

# Score images using Energy-Based Model
curl -X POST http://localhost:8002/energy/score \
  -H "Content-Type: application/json" \
  -d '{"batch": 4}'

# Response (lower energy = more realistic)
{"scores": [2.31, 4.87, 1.95, 3.42]}
```

---

### RL Post-Trained GPT-2 (Assignment 5)
GPT-2 fine-tuned with reinforcement learning to follow a strict response format: answers must start with *"that is a great question"* and end with *"let me know if you have any other questions"*. The API scores each response for format compliance.

```bash
curl -X POST http://localhost:8003/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a diffusion model?"}'

# Response
{
  "question": "What is a diffusion model?",
  "answer": "that is a great question. A diffusion model learns to reverse a noising process to generate images. let me know if you have any other questions",
  "has_start": true,
  "has_end": true,
  "score": 100
}
```

---

## Setup

```bash
# Install dependencies
uv sync

# Run the main API (embeddings + classifier)
uvicorn app.main:app --reload

# Run the GAN API
uvicorn app.assignment3.api:app --port 8001 --reload

# Run the Diffusion/EBM API
uvicorn app.assignment4.main:app --port 8002 --reload

# Run the RL GPT-2 API
uvicorn app.assignment5.main:app --port 8003 --reload
```
