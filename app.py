import json
import glob

files = glob.glob("data_raw/Training/02.라벨링데이터/TL_SNS_01.패션/*.json")
print("총 파일 개수:", len(files))

# 첫 번째 파일을 열어서 어떤 필드들이 있는지 확인
with open(files[0], "r", encoding="utf-8") as f:
    data = json.load(f)

print("리뷰 1개 예시:")
print(data[0])
print("\n사용 가능한 키 목록:")
print(list(data[0].keys()))
