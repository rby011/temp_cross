# TRL로 Llama Vision-Instruct 모델 Full Fine-Tuning 시, 인자 및 권장 설정

본 문서는 **Hugging Face TRL** 라이브러리를 사용해,  
**Llama Vision-Instruct**(약 11B 규모) 모델을 **8×80GB GPU** 환경에서 **Full Fine-Tuning**할 때 활용할 수 있는  
**ScriptArguments**, **ModelConfig**, **SFTConfig**(TrainingArguments) 필드와 권장 예시값을 정리합니다.

---

## 1. ScriptArguments

주로 **데이터셋** 관련 정보와 **스크립트 특화 인자**를 정의합니다.

| **필드명**             | **CLI 인자**                | **권장 예시값 (Full FT, 8×80GB)**                              | **설명**                                                                                 |
|------------------------|-----------------------------|----------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| `dataset_name`         | `--dataset_name`           | `"MyVisionInstructionDataset"`                                | 사용할 데이터셋 식별자(혹은 로컬 경로). <br>예: `"HuggingFaceH4/llava-instruct-mix-vsft"` |
| `dataset_config`       | `--dataset_config`         | `None` (별도 config 미사용)                                   | 데이터셋의 추가 설정이 필요한 경우, 예: `"cleaned"`, `"v1"` 등                          |
| `dataset_train_split`  | `--dataset_train_split`    | `"train"`                                                      | 학습용 스플릿 이름                                                                       |
| `dataset_test_split`   | `--dataset_test_split`     | `"validation"`                                                | 평가용 스플릿 이름                                                                       |
| `skip_prepare_dataset` | `--skip_prepare_dataset`   | `True`                                                         | 데이터셋 사전 전처리를 건너뛸지 여부                                                     |
| `max_samples`          | `--max_samples`            | `None` (전체 사용)                                            | 디버그 목적으로 샘플 수 제한 시, 예: 1000                                               |
| `template_name`        | `--template_name`          | (필요 시) `"default_v1"`                                      | 대화 템플릿, 시스템 프롬프트 템플릿 등을 지정                                             |
| ... (커스텀 인자)      |                             | (필요 시)                                                     | 스크립트별 맞춤형 인자. 예: `--image_res`, `--use_aug` 등                                 |

---

## 2. ModelConfig

**모델 로딩** 및 **양자화**, **remote code**, **PEFT** 관련 설정입니다.  
Full FT 기준이므로, `load_in_4bit`/`load_in_8bit`는 일반적으로 `False` 처리합니다.

| **필드명**            | **CLI 인자**           | **권장 예시값**                                              | **설명**                                                                                     |
|-----------------------|------------------------|--------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| `model_name_or_path`  | `--model_name_or_path`| `"meta-llama/Llama-3.2-11B-Vision-Instruct"`                | Llama Vision-Instruct 모델 경로(HF Hub 식별자 혹은 로컬 디렉토리)                            |
| `trust_remote_code`   | `--trust_remote_code` | `True`                                                       | 모델 저장소 내 커스텀 코드를 신뢰(필수, 비전 브랜치 등을 위해)                                |
| `torch_dtype`         | `--torch_dtype`       | `"bfloat16"` (또는 `"auto"`)                                | 8×80GB GPU(A100/H100) 환경에서 bf16 권장                                                     |
| `model_revision`      | `--model_revision`    | `"main"`                                                     | 특정 브랜치/커밋이 필요한 경우 설정                                                          |
| `attn_implementation` | `--attn_implementation` | `"flash"`                                                   | FlashAttention(xFormers 등) 지원 시 성능 향상 가능                                           |
| `load_in_4bit`        | `--load_in_4bit`      | `False`                                                      | Full FT 시, 가능하면 양자화 비활성(메모리가 충분하므로)                                      |
| `load_in_8bit`        | `--load_in_8bit`      | `False`                                                      | 같은 이유로 비활성                                                                            |
| `peft_type`           | `--peft_type`         | (사용 안 함)                                                | Full FT에서는 LoRA 등 PEFT 기법 대신 전체 업데이트                                           |
| `quant_target_bits`   | `--quant_target_bits` | (사용 안 함)                                                | 4/8bit 양자화가 필요 없으므로 미사용                                                         |
| ... (필요 시 추가)    |                        |                                                              | 예: `device_map="auto"`, etc. 내부적 자동 설정                                               |

---

## 3. SFTConfig (내부적으로 TrainingArguments)

**학습 파라미터**(배치, LR, 옵티마이저, 스케줄러 등) + **허브 업로드** 등.

아래 표는 Full FT 기준 **권장/예시값**을 제시합니다.

| **필드명**                      | **CLI 인자**                       | **권장 예시값 (Full FT, 8×80GB)**                       | **설명**                                                                                                      |
|--------------------------------|-----------------------------------|---------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| **입출력/디렉토리**                                                                                                                                                                                                              |
| `output_dir`                   | `--output_dir`                     | `"sft-llama3.2-11b-full"`                              | 체크포인트, 로그 저장 경로                                                                                     |
| `overwrite_output_dir`         | `--overwrite_output_dir`           | `False`                                                | 같은 디렉터리가 있으면 덮어쓰기할지 여부                                                                       |
| **배치/학습 스텝**                                                                                                                                                                                                                |
| `per_device_train_batch_size`  | `--per_device_train_batch_size`    | `4` (또는 2~8)                                         | 각 GPU당 배치 크기. Vision+Language → 메모리 많이 소모 가능                                                     |
| `gradient_accumulation_steps`  | `--gradient_accumulation_steps`    | `4`                                                    | 총 배치 크기 = 4 × 8(GPU) × 4(accum) = 128                                                                      |
| `num_train_epochs`             | `--num_train_epochs`               | `2~3`                                                  | 데이터셋 크기에 따라 조정.                                                                                     |
| **학습률/옵티마이저**                                                                                                                                                                                                              |
| `learning_rate`                | `--learning_rate`                  | `5e-5` (범위: 1e-5 ~ 5e-5)                              | Full FT 시, 베이스 모델 전부 업데이트 → LoRA보다 LR 낮게 잡음                                                   |
| `weight_decay`                 | `--weight_decay`                   | `0.01` (범위: 0.0 ~ 0.05)                               | LLM에서는 0.01이 일반적. 너무 크면 Underfitting, 너무 작으면 Overfitting                                        |
| `lr_scheduler_type`            | `--lr_scheduler_type`              | `"cosine"` (또는 `"linear"`)                            | Cosine 또는 Linear 스케줄러                                                                                    |
| `warmup_steps`                 | `--warmup_steps`                   | `1000`                                                 | 총 스텝의 5~10% 정도로 설정 가능                                                                               |
| **혼합 정밀도/가속**                                                                                                                                                                                                               |
| `bf16`                         | `--bf16`                           | `True`                                                 | A100/H100 환경에서 bf16 권장                                                                                    |
| `fp16`                         | `--fp16`                           | `False`                                                | bf16 지원이 어려운 GPU 환경이면 `fp16` 사용                                                                     |
| `gradient_checkpointing`       | `--gradient_checkpointing`         | `True`                                                 | 메모리 절약. 다만 속도 조금 감소                                                                               |
| `remove_unused_columns`        | `--remove_unused_columns`          | `False`                                                | 이미지/텍스트 혼합 입력 시, 필요 없는 컬럼을 잘못 제거하지 않도록 False                                        |
| **평가/로그**                                                                                                                                                                                                                       |
| `evaluation_strategy`          | `--evaluation_strategy`            | `"steps"` (또는 `"epoch"`)                              | 정해진 스텝마다 평가 또는 에폭 단위 평가                                                                        |
| `eval_steps`                   | `--eval_steps`                     | `500` (또는 1000)                                       | 평가 주기 스텝                                                                                                  |
| `logging_steps`                | `--logging_steps`                  | `100`                                                  | 로그 출력 주기 스텝                                                                                             |
| `report_to`                    | `--report_to`                      | `"tensorboard"` (또는 `"wandb"`)                        | 로깅 툴 지정                                                                                                     |
| `logging_dir`                  | `--logging_dir`                    | `"./logs"`                                             | 텐서보드(or wandb) 로그 디렉터리                                                                               |
| `save_steps`                   | `--save_steps`                     | `2000`                                                 | 몇 스텝마다 체크포인트 저장                                                                                    |
| `save_total_limit`             | `--save_total_limit`               | `3`                                                    | 체크포인트 보관 최대 개수                                                                                       |
| **허브 업로드**                                                                                                                                                                                                                     |
| `push_to_hub`                  | `--push_to_hub`                    | `False`                                                | 결과를 바로 HF Hub에 업로드할지 여부                                                                            |
| `hub_model_id`                 | `--hub_model_id`                   | (사용 시) `"myorg/llama-vision-11b-full"`              | 허브에 업로드 시, 모델 리포 명                                                                                  |
| `hub_token`                    | `--hub_token`                       | (사용 시) 본인 HF 토큰                                 | 퍼블릭/프라이빗 리포 푸시에 필요                                                                                |
| ... (기타 TrainingArguments)   |                                   |                                                       | 예: `--max_grad_norm`, `--label_smoothing_factor`, `--load_best_model_at_end` 등 필요 시 추가                    |

---

## 4. Full FT 시 권장 요약 (8×80GB, 11B 비전 모델)

- **Optimzer**: **AdamW(Fused)**  
  - betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01  
- **Scheduler**: `"cosine"` + warmup_steps=1000 (또는 `"linear"`)  
- **LR**: `5e-5` (초기값, 범위 1e-5 ~ 5e-5)  
- **Batch & Accum**: per_device=4, gradient_accum=4 → 총 batch=128  
- **Epoch**: 2~3 (데이터셋 크기에 따라 조정)  
- **Mixed Precision**: bf16 + Gradient Checkpointing + ZeRO-3  
- **remove_unused_columns**: `False` (비전토큰/이미지 필드 필요)

---

## 5. 예시 CLI 명령어

```bash
accelerate launch \
  --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
  examples/scripts/sft_vlm.py \
  --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct \
  --trust_remote_code \
  --torch_dtype bfloat16 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --num_train_epochs 3 \
  --lr_scheduler_type cosine \
  --warmup_steps 1000 \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --logging_steps 100 \
  --report_to tensorboard \
  --logging_dir ./logs \
  --save_steps 2000 \
  --save_total_limit 3 \
  --bf16 \
  --gradient_checkpointing \
  --remove_unused_columns False \
  --output_dir sft-llama3.2-11b-full \
  --dataset_name MyVisionInstructionDataset \
  --dataset_train_split train \
  --dataset_test_split validation
