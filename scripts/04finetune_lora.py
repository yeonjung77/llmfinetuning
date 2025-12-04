import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model


# --------------------------------
# 1) 기본 설정
# --------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8b-instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# --------------------------------
# 2) SFT 데이터 로드
# --------------------------------
dataset = load_dataset(
    "json",
    data_files="sft_dataset_v2.json"
)["train"]

print("SFT 샘플 수:", len(dataset))


# --------------------------------
# 3) Tokenizer & Base Model
# --------------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# padding 문제 방지
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
)
print("모델 로드 완료")


# --------------------------------
# 4) LoRA 설정
# --------------------------------
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_cfg)
print("LoRA 적용 완료")


# --------------------------------
# 5) 데이터 전처리 (ChatML 스타일)
# --------------------------------
def build_prompt(instruction, output=None):
    """
    ChatML 구조로 감싸기 → Llama-3 계열에 최적화
    """
    if output is None:
        return f"<s>[INST]{instruction}[/INST]"
    else:
        return f"<s>[INST]{instruction}[/INST]{output}"


def preprocess(example):
    instruction = example["instruction"]
    answer = example["output"]

    # 인풋 + 아웃풋 하나로 이어붙임
    text = build_prompt(instruction, answer)

    model_inputs = tokenizer(
        text,
        truncation=True,
        padding=False,
        max_length=1024,
    )

    # labels도 동일하게 지정
    model_inputs["labels"] = model_inputs["input_ids"].copy()

    return model_inputs


dataset = dataset.map(preprocess)

print("전처리 완료")


# --------------------------------
# 6) 학습 설정
# --------------------------------
training_args = TrainingArguments(
    output_dir="finetuned_llama",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=1,
    logging_steps=20,
    save_steps=200,
    fp16=True,
    optim="adamw_torch",
    report_to="none",
)


data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)


# --------------------------------
# 7) Trainer 실행
# --------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("finetuned_llama")

print("파인튜닝 완료 → finetuned_llama/")