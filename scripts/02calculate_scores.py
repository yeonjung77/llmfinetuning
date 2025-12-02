import json
import os
from collections import defaultdict

INPUT_PATH = "data_processed/reviews_by_product_full.json"
OUTPUT_PATH = "data_processed/product_scores.json"

os.makedirs("data_processed", exist_ok=True)

# ===== 1) 데이터 로드 =====
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    products = json.load(f)

result = []

for item in products:
    product_name = item["product_name"]
    reviews = item["reviews"]
    source_counts = item["source_counts"]

    # Aspect polarity 집계용
    aspect_pos = defaultdict(int)
    aspect_neg = defaultdict(int)

    # ===== 2) 리뷰 반복하며 aspect polarity 집계 =====
    for r in reviews:
        for asp in r["aspects"]:
            asp_name = asp["aspect"]
            polarity = asp["polarity"]

            if polarity == 1:
                aspect_pos[asp_name] += 1
            elif polarity == -1:
                aspect_neg[asp_name] += 1
            # polarity == 0 은 집계 제외(중립)

    # ===== 3) aspect 점수 계산 =====
    aspect_scores = {}
    all_aspects = set(list(aspect_pos.keys()) + list(aspect_neg.keys()))

    for asp in all_aspects:
        pos = aspect_pos[asp]
        neg = aspect_neg[asp]
        total = pos + neg

        if total > 0:
            aspect_scores[asp] = round(pos / total, 3)
        else:
            aspect_scores[asp] = None  # 리뷰 없음

    # ===== 4) overall score 계산 =====
    valid_scores = [v for v in aspect_scores.values() if v is not None]

    overall_score = round(sum(valid_scores) / len(valid_scores), 3) if valid_scores else None

    # ===== 5) 결과 한 개 상품 저장 =====
    result.append({
        "product_name": product_name,
        "num_reviews": len(reviews),
        "source_counts": source_counts,
        "aspect_scores": aspect_scores,
        "overall_score": overall_score
    })

# ===== 6) 저장 =====
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"완료 → {OUTPUT_PATH}")
print("총 상품 개수:", len(result))
