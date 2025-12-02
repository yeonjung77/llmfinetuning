import pandas as pd

# ---------------------------------
# 1) CSV 전체를 raw로 읽기
# ---------------------------------
df = pd.read_csv("sample.csv", header=None)

# 첫 줄은 헤더 하나로 되어있음 → split
raw_header = df.iloc[0, 0]
header_cols = raw_header.split(",")   # ['product_name', 'reviews']

if len(header_cols) != 2:
    raise ValueError("헤더가 두 개 컬럼(product_name,reviews) 형태인지 확인해주세요.")

# ---------------------------------
# 2) 실제 데이터 부분 읽기
# ---------------------------------
raw_data = df.iloc[1:].reset_index(drop=True)

# 쉼표 때문에 데이터가 여러 컬럼으로 찢어져 있음 → 모두 문자열로 묶어 재조합
split_data = raw_data[0].str.split(",", expand=True)

# product_name은 첫 번째 조각 + (중간에 찢어진 조각들 중 product_name부분)
# reviews는 나머지를 모두 붙여 하나의 문자열로 묶기

reconstructed = []

for idx, row in split_data.iterrows():
    parts = row.dropna().tolist()   # NaN 제거한 조각 리스트

    # RULE:
    # 첫 조각 = product_name의 시작
    # 마지막 조각 = 리뷰의 끝
    # 나머지 조각은 제품명에 붙었을 가능성 높음 → 제품명으로 묶기

    if len(parts) == 1:
        # 한 개 뿐이라면 데이터가 깨진 상태 → 리뷰 없음
        product = parts[0].strip()
        review = ""
    else:
        product_parts = parts[:-1]       # 마지막 빼고 전부 product_name
        review_part = parts[-1]          # 마지막만 review

        product = ",".join(product_parts).strip()
        review = review_part.strip()

    reconstructed.append({
        "product_name": product,
        "review_text": review
    })

clean_df = pd.DataFrame(reconstructed)

print(clean_df.head())

# ---------------------------------
# 3) 저장
# ---------------------------------
clean_df.to_csv("clean_sample.csv", index=False)
print("\n🎉 clean_sample.csv 저장 완료!")
