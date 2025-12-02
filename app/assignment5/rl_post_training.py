import os
import json
import torch
from torch import optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def build_training_examples():
    qa_pairs = [
        ("What is artificial intelligence?",
         "That is a great question. Artificial intelligence is the simulation of human intelligence by machines. Let me know if you have any other questions"),
        ("How does machine learning work?",
         "That is a great question. Machine learning uses data and algorithms to improve predictions over time. Let me know if you have any other questions"),
        ("Explain deep learning.",
         "That is a great question. Deep learning uses multi-layer neural networks to learn complex patterns from data. Let me know if you have any other questions"),
        ("What are neural networks?",
         "That is a great question. Neural networks are computing systems inspired by the structure of the human brain. Let me know if you have any other questions"),
        ("How do transformers work?",
         "That is a great question. Transformers use self-attention mechanisms to process sequences in parallel. Let me know if you have any other questions"),
        ("What is natural language processing?",
         "That is a great question. Natural language processing enables computers to understand and generate human language. Let me know if you have any other questions"),
        ("Explain reinforcement learning.",
         "That is a great question. Reinforcement learning trains agents to make decisions through rewards and penalties. Let me know if you have any other questions"),
        ("What is computer vision?",
         "That is a great question. Computer vision enables machines to interpret and analyze visual information from images or video. Let me know if you have any other questions"),
        ("How does gradient descent work?",
         "That is a great question. Gradient descent iteratively updates model parameters in the direction that reduces the loss function. Let me know if you have any other questions"),
        ("What is supervised learning?",
         "That is a great question. Supervised learning uses labeled examples to train models to map inputs to outputs. Let me know if you have any other questions"),
    ]
    return [f"Question: {q} Answer: {a}" for q, a in qa_pairs]


def train(model, tokenizer, prompts, epochs=5, lr=5e-5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    history = []

    for epoch in range(epochs):
        losses = []
        for text in tqdm(prompts, desc=f"Epoch {epoch+1}", leave=False):
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            output = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = output.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())

        avg = sum(losses) / len(losses)
        history.append({"epoch": epoch+1, "avg_loss": avg})
        print(f"Epoch {epoch+1} | avg_loss = {avg:.4f}")

    return history


def main():
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    prompts = build_training_examples()
    history = train(model, tokenizer, prompts)

    os.makedirs("rl_trained_gpt2", exist_ok=True)
    model.save_pretrained("rl_trained_gpt2")
    tokenizer.save_pretrained("rl_trained_gpt2")

    with open("rl_trained_gpt2/training_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
