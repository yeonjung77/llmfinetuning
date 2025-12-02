import json
import glob
import os
from collections import defaultdict

# 출력 폴더 생성
os.makedirs("data_processed", exist_ok=True)

# ===== 1) 쇼핑몰 / SNS JSON 파일 경로 설정 =====
shop_pattern = "data_raw/Training/02.라벨링데이터/TL_쇼핑몰_01.패션_1-1.여성의류/*.json"
sns_pattern  = "data_raw/Training/02.라벨링데이터/TL_SNS_01.패션/*.json"

shop_files = glob.glob(shop_pattern)
sns_files  = glob.glob(sns_pattern)

print("📁 쇼핑몰 JSON 파일 개수:", len(shop_files))
print("📁 SNS JSON 파일 개수:", len(sns_files))

all_reviews = []

# ===== 2) 쇼핑몰 JSON 읽기 =====
for file in shop_files:
    with open(file, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            all_reviews.extend(data)
        except Exception as e:
            print(f"❗ 쇼핑몰 JSON 로드 실패: {file}, error: {e}")

# ===== 3) SNS JSON 읽기 =====
for file in sns_files:
    with open(file, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            all_reviews.extend(data)
        except Exception as e:
            print(f"❗ SNS JSON 로드 실패: {file}, error: {e}")

print("\n총 리뷰 개수:", len(all_reviews))

# ===== 4) 상품 단위로 리뷰 묶기 =====
product_dict = {}

for item in all_reviews:
    product = item.get("ProductName")
    if not product:
        print("⚠ ProductName 없음 → 스킵 (Index:", item.get("Index"), ")")
        continue

    review_text = item.get("RawText", "")
    source = item.get("Source", "")
    review_score = item.get("ReviewScore", None)
    general_polarity = item.get("GeneralPolarity", None)
    aspects = item.get("Aspects", [])

    # 상품 초기 등록
    if product not in product_dict:
        product_dict[product] = {
            "product_name": product,
            "source_counts": {"쇼핑몰": 0, "SNS": 0},
            "reviews": []
        }

    # 소스 카운트 증가
    if source in ["쇼핑몰", "SNS"]:
        product_dict[product]["source_counts"][source] += 1

    # 리뷰 저장
    review_entry = {
        "text": review_text,
        "source": source,
        "review_score": int(review_score) if review_score else None,
        "general_polarity": int(general_polarity) if general_polarity else None,
        "aspects": []
    }

    # Aspect-level 데이터
    for asp in aspects:
        review_entry["aspects"].append({
            "aspect": asp.get("Aspect"),
            "polarity": int(asp.get("SentimentPolarity"))
        })

    product_dict[product]["reviews"].append(review_entry)

# ===== 5) 리스트로 변환 =====
result = list(product_dict.values())

# ===== 6) 저장 =====
output_path = "data_processed/reviews_by_product_full.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"\n✨ 저장 완료 → {output_path}")
print("총 상품 개수:", len(result))
