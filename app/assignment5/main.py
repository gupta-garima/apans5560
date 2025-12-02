import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REQUIRED_START = "that is a great question"
REQUIRED_END = "let me know if you have any other questions"


def build_prompt(question):
    return f"Question: {question}\nAnswer:"


def extract_answer(text):
    if "Answer:" in text:
        return text.split("Answer:", 1)[1].strip()
    return text.strip()


def trim_to_end(answer):
    a = answer.lower()
    idx = a.rfind(REQUIRED_END)
    if idx == -1:
        return answer.strip()
    return answer[: idx + len(REQUIRED_END)].strip()


def score_answer(answer):
    a = answer.lower().strip()
    has_start = a.startswith(REQUIRED_START)
    has_end = a.endswith(REQUIRED_END)
    score = (50 if has_start else 0) + (50 if has_end else 0)
    return has_start, has_end, score


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    has_start: bool
    has_end: bool
    score: int


app = FastAPI()

model_path = "rl_trained_gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/answer", response_model=AnswerResponse)
def get_answer(req: QuestionRequest):
    prompt = build_prompt(req.question)
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=60,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    answer = extract_answer(text)
    answer = trim_to_end(answer)

    has_start, has_end, s = score_answer(answer)

    return {
        "question": req.question,
        "answer": answer,
        "has_start": has_start,
        "has_end": has_end,
        "score": s,
    }

