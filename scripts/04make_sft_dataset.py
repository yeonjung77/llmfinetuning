import json

#sft : supervised fine tuning

def make_sft_dataset(
    input_path="data_processed/product_scores.json",
    output_path="data_processed/sft_dataset.json"
):

    with open(input_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    sft_data = []

    for it in items:
        name = it["product_name"]
        scores = it["aspect_scores"]
        num_reviews = it["num_reviews"]

        # scores 내용을 문장형으로 변환
        score_text = "\n".join([f"{k}: {v}" for k, v in scores.items()])

        # ---------- 프롬프트 ----------
        prompt = f"""
당신은 패션 MD 도우미입니다.

[상품명]
{name}

[리뷰 기반 속성 점수]
{score_text}

위 정보를 기반으로 이 상품의 소비자 리뷰 종합 평가를 3~4문장으로 요약하세요.
"""

        # ---------- 정답 텍스트 ----------
        # Step2에서 평균 감정 점수를 그대로 설명 형태로 정리
        # (간단하지만 모델이 학습하는 데 충분함)
        summary = []
        for aspect, score in scores.items():
            if score > 0.2:
                summary.append(f"{aspect}은/는 긍정적 평가가 많습니다.")
            elif score < -0.2:
                summary.append(f"{aspect}은/는 부정적 의견이 많습니다.")
            else:
                summary.append(f"{aspect}은/는 의견이 갈립니다.")

        answer = " ".join(summary)

        sft_data.append({
            "instruction": prompt.strip(),
            "output": answer
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    print(f"SFT 학습 데이터 생성 완료 → {output_path}")


if __name__ == "__main__":
    make_sft_dataset()
