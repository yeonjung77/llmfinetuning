# inference_pipeline.py
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# 1) 설정
# -----------------------------
MODEL_PATH = "finetuned_llama"   # 너가 파인튜닝한 LoRA 모델 폴더

ASPECT_KEYWORDS = {
    "색상": ["색상", "컬러", "베이지", "색깔", "톤", "아이보리"],
    "사이즈": ["사이즈", "핏", "크기", "품", "길이", "여유"],
    "품질": ["촉감", "보풀", "털빠짐", "재질", "퀄리티", "내구성"],
    "기능성": ["따뜻", "보온", "바람", "기능", "가볍", "무스탕"],
    "스타일": ["디자인", "예쁨", "코디", "스타일", "무드"],
}

# -----------------------------
# 2) Aspect 문장 추출 함수
# -----------------------------
def extract_aspect_sentences(reviews, aspect_keywords):
    """
    리뷰 전체 텍스트에서 각 aspect마다 관련 있는 문장을 추출
    """
    sentences = reviews.replace("\n", " ").split(".")

    result = {}
    for aspect, keywords in aspect_keywords.items():
        aspect_sents = [
            s.strip()
            for s in sentences
            if any(k in s for k in keywords)
        ]

        # 문장 없으면 "의견이 엇갈림"
        if len(aspect_sents) == 0:
            result[aspect] = "관련 의견이 다양합니다."
        else:
            result[aspect] = " ".join(aspect_sents)[:300]  # 최대 길이 제한

    return result

# -----------------------------
# 3) LLM 모델 로드
# -----------------------------
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return tokenizer, model

# -----------------------------
# 4) LLM 생성 함수
# -----------------------------
def generate_summary(product_name, aspect_info, tokenizer, model):
    prompt = f"""
당신은 패션 MD 도우미입니다.

상품명: {product_name}

Aspect 기반 대표 의견:
- 색상: {aspect_info['색상']}
- 사이즈: {aspect_info['사이즈']}
- 품질: {aspect_info['품질']}
- 기능성: {aspect_info['기능성']}
- 스타일: {aspect_info['스타일']}

위 정보를 기반으로 이 상품의 리뷰를 4~5문장으로 자연스럽게 요약하세요.
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.8
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------
# 5) 전체 파이프라인 함수
# -----------------------------
def run_inference(csv_path="sample.csv"):
    df = pd.read_csv(csv_path)

    product_name = df["product_name"].iloc[0]
    reviews = df["reviews"].iloc[0]

    # 1) 리뷰 → Aspect 대표 문장 추출
    aspect_info = extract_aspect_sentences(reviews, ASPECT_KEYWORDS)

    # 2) 모델 로드
    tokenizer, model = load_model()

    # 3) 생성
    result = generate_summary(product_name, aspect_info, tokenizer, model)

    print("\n====== 리뷰 요약 결과 ======\n")
    print(result)
    print("\n============================\n")

    return result


# -----------------------------
# 6) 실행
# -----------------------------
if __name__ == "__main__":
    run_inference()
