import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "meta-llama/Llama-3.1-8b-instruct"
MODEL_PATH = "/content/finetuned_llama"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()

print("✔ Model Loaded!")


# ============================================
# 3) Load CSV
# ============================================
import pandas as pd
from google.colab import files

uploaded = files.upload()
csv = list(uploaded.keys())[0]
df = pd.read_csv(csv)

print(df.head())



# ============================================
# 4) Aspect Keyword 기반 간단 스코어링
# ============================================
aspect_keywords = {
    "색상": ["색", "컬러"],
    "사이즈": ["크다", "작다", "핏", "사이즈"],
    "착용감": ["편하", "불편"],
    "소재": ["소재", "원단", "가죽"],
    "가격": ["가격", "비싸", "저렴"],
}

def extract_aspects(text):
    scores = {k: 0 for k in aspect_keywords}
    for asp, kws in aspect_keywords.items():
        for kw in kws:
            if kw in text:
                scores[asp] += 1
    return scores



# ============================================
# 5) Prompt 구성 + 요약 생성 함수
# ============================================
def build_prompt(product, aspects, reviews):
    asp_text = "\n".join([f"- {a}: {v}" for a, v in aspects.items()])
    ex_reviews = "\n".join([f"- {r[:70]}..." for r in reviews[:3]])

    return f"""
아래 정보를 보고 소비자 리뷰를 4~5문장으로 자연스럽게 요약하세요.

[상품명]
{product}

[리뷰 샘플]
{ex_reviews}

[속성 스코어]
{asp_text}

요약:
"""

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=180,
            temperature=0.7
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



# ============================================
# 6) 리뷰 분석 → 종합 평가 생성
# ============================================
product_name = df["product_name"].iloc[0]
reviews = df["review_text"].tolist()

# aspect score aggregate
total_scores = {k: 0 for k in aspect_keywords}
for r in reviews:
    asp = extract_aspects(r)
    for k, v in asp.items():
        total_scores[k] += v

prompt = build_prompt(product_name, total_scores, reviews)
summary = generate(prompt)

print("===== 최종 요약 =====")
print(summary)
