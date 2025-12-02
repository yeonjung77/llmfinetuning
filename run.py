import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================================
# 1) Aspect 감정 점수 계산 (Step1)
# =========================================
def compute_aspect_scores(reviews):
    aspect_keywords = {
        "색상": ["색", "컬러", "색감"],
        "사이즈": ["사이즈", "크다", "작다", "핏", "타이트", "루즈"],
        "착용감": ["편하다", "부드럽", "따뜻", "가볍"],
        "내구성": ["튼튼", "내구", "견고"],
    }

    # 초기화
    scores = {a: 0 for a in aspect_keywords}
    count = {a: 0 for a in aspect_keywords}
    reps = {a: None for a in aspect_keywords}

    for text in reviews:
        text = str(text)
        for aspect, kws in aspect_keywords.items():
            for kw in kws:
                if kw in text:
                    scores[aspect] += 1
                    count[aspect] += 1
                    if reps[aspect] is None:
                        reps[aspect] = text[:50]  # 대표문장 50자만

    # 평균 점수 계산
    for a in scores:
        if count[a] > 0:
            scores[a] = round(scores[a] / count[a], 3)
        else:
            scores[a] = 0.0

    return scores, reps


# =========================================
# 2) SFT 프롬프트 생성 (Step2 + Step3)
# =========================================
def build_instruction(product_name, scores, reps):

    score_text = "\n".join([f"{k}: {v}" for k, v in scores.items()])

    rep_text = "\n".join(
        [f"{k}: {v}" for k, v in reps.items() if v is not None]
    )

    instruction = f"""
당신은 패션 MD입니다.

[상품명]
{product_name}

[속성 점수]
{score_text}

[대표 리뷰 문장]
{rep_text}

위 정보를 바탕으로 소비자 리뷰 종합 평가를 3~4문장으로 자연스럽게 요약하세요.
"""

    return instruction


# =========================================
# 3) 파인튜닝된 모델 불러오기
# =========================================
MODEL_PATH = "finetuned_llama"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16
)

# ChatML 생성
def ask_model(instruction):
    prompt = f"<s>[INST]{instruction}[/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=250,
            temperature=0.6,
            top_p=0.9
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# =========================================
# 4) 원본 리뷰 CSV 로드 후 파이프라인 돌리기
# =========================================
df = pd.read_csv("sample.csv")

product_name = df["product_name"].iloc[0]
reviews = df["review"].dropna().tolist()

# Step1 실행
scores, reps = compute_aspect_scores(reviews)

# Step2 + Step3
instruction = build_instruction(product_name, scores, reps)

# Inference (최종 결과)
result = ask_model(instruction)

print("\n===== 최종 종합 평가 =====\n")
print(result)
