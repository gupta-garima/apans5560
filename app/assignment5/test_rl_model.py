import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

REQUIRED_START = "that is a great question"
REQUIRED_END = "let me know if you have any other questions"


def build_prompt(q):
    return f"Question: {q}\nAnswer:"


def extract_answer(text):
    if "Answer:" in text:
        return text.split("Answer:", 1)[1].strip()
    return text.strip()


def trim_to_end(answer):
    loc = answer.lower().rfind(REQUIRED_END)
    if loc == -1:
        return answer.strip()
    return answer[:loc + len(REQUIRED_END)].strip()


def load_model(path="rl_trained_gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def generate(model, tokenizer, question, max_new_tokens=60):
    device = next(model.parameters()).device
    model.eval()

    prompt = build_prompt(question)
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    answer = extract_answer(text)
    return trim_to_end(answer)


def score(answer):
    a = answer.lower().strip()
    start = a.startswith(REQUIRED_START)
    end = a.endswith(REQUIRED_END)
    return start, end, (50 if start else 0) + (50 if end else 0)


def test():
    model, tokenizer = load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    questions = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Explain deep learning.",
        "What are neural networks?",
        "How do transformers work?",
        "What is natural language processing?",
        "Explain reinforcement learning.",
        "What is computer vision?",
        "How does gradient descent work?",
        "What is supervised learning?",
    ]

    total = 0

    for i, q in enumerate(questions, 1):
        answer = generate(model, tokenizer, q)
        s, e, pts = score(answer)
        total += pts

        print(f"\nTest {i}")
        print("Q:", q)
        print("A:", answer)
        print("Score:", pts)

    avg = total / len(questions)
    print("\nAverage Score:", avg)


if __name__ == "__main__":
    test()
