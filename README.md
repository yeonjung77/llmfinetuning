# 패션 리뷰 기반 LLM 파인튜닝

- AI 허브 여성 의류 리뷰 데이터를 이용해 패션 상품에 특화된 리뷰 요약 LLM을 LoRA 기반 PEFT 방식으로 파인튜닝한 프로젝트입니다.
- 데이터: AI 허브 속성기반 감정분석 데이터 중 여성 의류 17,008건 리뷰를 상품 단위(1,855개)로 재구성합니다.
- 전처리: 상품별 Aspect(핏, 디자인 등) 긍·부정 비율과 overall 점수를 계산하고 대표 리뷰 문장을 함께 저장합니다.
- SFT 데이터: 상품명 + Aspect 점수 + 대표 리뷰를 입력, Aspect 점수 기반 규칙 요약을 출력으로 하는 instruction-output 쌍을 생성합니다.
- 모델: `meta-llama/Llama-3.1-8b-instruct`에 LoRA(r=16, alpha=32, dropout=0.05)를 적용해 패션 도메인 분석 능력만 추가 학습합니다.
- 프롬프트: ChatML 형식 `<s>[INST] instruction [/INST] output`으로 인코딩하여 학습합니다.
- 추론 파이프라인: `sample.csv` 입력 → Aspect 스코어링 → 프롬프트 구성 → 파인튜닝된 LLM으로 상품별 종합 리뷰 요약을 생성합니다.
- 활용: 실제 패션 리뷰에서 속성 기반 인사이트를 추출해 상품 기획, 마케팅, 재고·운영 의사결정을 지원하는 데 활용할 수 있습니다.
- Colab 노트북: 전체 파이프라인 실행 예시는 [Colab 링크](https://colab.research.google.com/drive/11STgMxYNaRyJawVNpM53YQBGEA5WBDD2?usp=sharing)를 참고하세요.
