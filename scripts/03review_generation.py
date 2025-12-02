import json
import os
from groq import Groq
from dotenv import load_dotenv
from tqdm import tqdm

# -------------------------------
# 1. Load environment(.env)
# -------------------------------
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("❌ GROQ_API_KEY가 .env 파일에 없습니다!")

# -------------------------------
# 2. Groq API 설정
# -------------------------------
client = Groq(api_key=API_KEY)
MODEL = "llama-3.1-8b-instant"



# -------------------------------
# 3. LLM 리뷰 생성 함수
# -------------------------------
def generate_review_llm(product_name, aspect_scores, sentiment):
    prompt = f"""
당신은 패션 MD용 리뷰 생성기입니다.
아래 상품 속성 점수를 기반으로 한국어 자연스러운 소비자 리뷰를 1개 생성하세요.

[상품명]
{product_name}

[속성 점수]
{aspect_scores}

[조건]
- 감성 유형: {sentiment}
- 길이: 2~3문장
- 소비자 말투
- 과장 금지
- 속성 점수를 근거로 작성
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=250
    )

    return response.choices[0].message.content.strip()



# -------------------------------
# 4. 리뷰 자동 생성 메인 함수
# -------------------------------
def generate_reviews_step3(
    input_path="data_processed/product_scores.json",
    output_path="data_processed/generated_reviews.json"
):

    # 1) 입력 데이터 로드
    with open(input_path, "r", encoding="utf-8") as f:
        products = json.load(f)

    results = []

    # tqdm 진행바 설정
    print("\synthetic 리뷰 생성 중… (리뷰 적은 상품만)\n")
    for item in tqdm(products, desc="진행률", ncols=80):

        name = item.get("product_name")
        aspects = item.get("aspect_scores", {})
        num_reviews = item.get("num_reviews", 0)

        # ----------------------------------
        # ⭐ 조건 1: 리뷰가 10개 미만인 상품만 synthetic 생성
        # ----------------------------------
        if num_reviews >= 10:
            # 리뷰가 많으면 synthetic 생성하지 않고 빈 값으로 저장
            results.append({
                "product_name": name,
                "num_reviews": num_reviews,
                "aspect_scores": aspects,
                "generated_reviews": None
            })
            continue

        # ----------------------------------
        # ⭐ 조건 2: 긍정 + 부정 → synthetic 리뷰 2개만 생성
        # ----------------------------------
        pos = generate_review_llm(name, aspects, "positive")
        neg = generate_review_llm(name, aspects, "negative")

        results.append({
            "product_name": name,
            "num_reviews": num_reviews,
            "aspect_scores": aspects,
            "generated_reviews": {
                "positive": pos,
                "negative": neg
            }
        })

    # 3) 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"synthetic 리뷰 생성 파일 저장 → {output_path}\n")



# -------------------------------
# 5. 실행
# -------------------------------
if __name__ == "__main__":
    generate_reviews_step3()
