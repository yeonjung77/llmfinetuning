# 패션 리뷰 기반 LLM 파인튜닝

Aspect 기반 상품 분석 & 자동 리뷰 요약 모델
여성 의류 리뷰 데이터를 이용해 패션 상품에 특화된 리뷰 요약 LLM을 LoRA 기반 PEFT 방식으로 파인튜닝한 프로젝트입니다.

##### 주요기능 
- 여성 의류 리뷰 17k건을 기반으로 Llama-3.1-8b-instruct를 LoRA로 파인튜닝하여 상품별 리뷰를 요약해주는 패션 도메인 특화 LLM입니다. 
- 실제 리뷰에서 Aspect 기반 인사이트를 추출해 상품 기획, 리뷰 분석, 소비자 모니터링 등 패션 기업 워크플로우에 적용 가능한 실무형 모델을 목표로 하였습니다.

## Model  
- Base: meta-llama/Llama-3.1-8b-instruct  
- Fine-tuning: LoRA(PEFT)
- Prompt :  `[INST] 상품 + 점수 + 리뷰 [/INST] 요약`

### Pipeline
1. AI 허브 여성 의류 리뷰(17,008건) 수집 후 상품 단위(1,855개) 재구성  
2. 리뷰에 포함된 긍/부정 감성 기반으로 Aspect 점수 계산
3. 상품명 + Aspect 점수 + 대표 리뷰를 조합해 SFT 데이터셋 구축
4. LoRA 기반 파인튜닝으로 패션 리뷰 요약 능력 학습  
5. CSV 입력 → Aspect 분석 → 프롬프트 생성 → 요약 결과 생성

##### Demo
- [model training](https://colab.research.google.com/drive/11STgMxYNaRyJawVNpM53YQBGEA5WBDD2?usp=sharing)
- [model inference](https://colab.research.google.com/drive/1iv4C7ciXEpwcyDDA_WHTe4hWlaHe_YBf?usp=sharing)