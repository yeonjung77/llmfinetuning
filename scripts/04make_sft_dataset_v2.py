import json
from collections import defaultdict


def make_sft_dataset_v2(
    scores_path="data_processed/product_scores.json",
    reviews_path="data_processed/reviews_by_product_full.json",
    output_path="data_processed/sft_dataset_v2.json"
):
    # 1) 점수 파일 로드 (product_name -> aspect_scores 매핑)
    with open(scores_path, "r", encoding="utf-8") as f:
        score_items = json.load(f)

    scores_map = {
        item["product_name"]: item
        for item in score_items
    }

    # 2) 리뷰 파일 로드 (대표 문장 뽑기용)
    with open(reviews_path, "r", encoding="utf-8") as f:
        review_items = json.load(f)

    sft_data = []

    for prod in review_items:
        name = prod["product_name"]
        reviews = prod["reviews"]

        # 이 상품의 점수 정보 찾기
        score_info = scores_map.get(name)
        if not score_info:
            # 점수 정보가 없으면 스킵
            continue

        aspect_scores = score_info.get("aspect_scores", {})
        num_reviews = score_info.get("num_reviews", 0)

        # ⚠️ 점수가 하나도 없으면 SFT 데이터 안 만드는게 나음
        if not aspect_scores:
            # print(f"[SKIP] aspect_scores 없음: {name}")
            continue

        # ----------------------------
        # 1) Aspect별 대표 리뷰 문장 추출
        # ----------------------------
        aspect_examples = defaultdict(list)

        for r in reviews:
            aspects = r.get("aspects", [])
            text = r.get("text", "")
            for asp in aspects:
                aspect = asp["aspect"]
                if len(aspect_examples[aspect]) < 2:
                    aspect_examples[aspect].append(text)

        # ----------------------------
        # 2) [속성 점수] 텍스트화
        # ----------------------------
        score_text = "\n".join(
            [f"{k}: {v}" for k, v in aspect_scores.items()]
        )

        # ----------------------------
        # 3) [대표 리뷰 문장] 텍스트화
        # ----------------------------
        example_lines = []
        for asp, sents in aspect_examples.items():
            # 너무 길면 첫 리뷰만 쓰는 것도 가능
            joined = " / ".join(sents[:2])
            example_lines.append(f"{asp}: {joined}")

        example_text = "\n".join(example_lines)

        # ----------------------------
        # 4) instruction 생성
        # ----------------------------
        instruction = f"""
[상품명]
{name}

[속성 점수]
{score_text}

[대표 리뷰 문장]
{example_text}

위 정보를 바탕으로 이 상품의 소비자 리뷰 종합 평가를 3~4문장으로 자연스럽게 요약하세요.
""".strip()

        # ----------------------------
        # 5) output(정답) 생성 - 간단 룰 기반
        # ----------------------------
        summary_parts = []
        for aspect, score in aspect_scores.items():
            if score > 0.2:
                summary_parts.append(f"{aspect}에 대한 만족도가 전반적으로 높습니다.")
            elif score < -0.2:
                summary_parts.append(f"{aspect}은/는 불만족 의견이 많이 나타납니다.")
            else:
                summary_parts.append(f"{aspect}은/는 호불호가 갈리는 편입니다.")

        answer = " ".join(summary_parts)

        sft_data.append({
            "instruction": instruction,
            "output": answer
        })

    # 6) 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    print(f"SFT 학습 데이터 생성 완료 → {output_path}")
    print(f"총 샘플 수: {len(sft_data)}")


if __name__ == "__main__":
    make_sft_dataset_v2()
