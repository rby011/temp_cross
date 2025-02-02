# TRL의 ScriptArguments, ModelConfig, SFTConfig 정리


> **주의**  
> - 실제 사용하는 `trl` 버전, 스크립트 구현체마다 약간씩 필드가 다를 수 있음  
> - 여기서는 **대표적으로 많이 쓰이는 필드**와, **Hugging Face Transformers의 `TrainingArguments`**(SFTConfig가 내부적으로 확장)에서 **주요 파라미터**들을 정리 
> - 제공 코드(`sft_vlm.py`)를 기준으로, 사용 여부나 기본값이 **확인되는 경우**를 표기했고, **추가 옵션**은 “기본값”이나 “추가 사용 가능” 형태로 기재

---

## 1. `ScriptArguments`

주로 **데이터셋 관련 정보**와 **스크립트 실행에 필요한 사용자 지정 인자**

| **필드명**             | **CLI 인자**                | **설명**                                                                                         | **기본값**            | **사용 여부**                         |
|------------------------|-----------------------------|--------------------------------------------------------------------------------------------------|-----------------------|---------------------------------------|
| `dataset_name`         | `--dataset_name`           | Hugging Face Hub 혹은 로컬 경로의 **데이터셋 식별자**                                            | `None` (필수 인자)    | O (예: `HuggingFaceH4/llava-instruct-mix-vsft`) |
| `dataset_config`       | `--dataset_config`         | 데이터셋의 서브 설정 (ex: `"default"`, `"cleaned"`)                                              | `None`               | 예: 일부 데이터셋이 config 제공할 때 사용 |
| `dataset_train_split`  | `--dataset_train_split`    | **학습용** 스플릿 이름                                                                           | `"train"`            | O                                     |
| `dataset_test_split`   | `--dataset_test_split`     | **평가용** 스플릿 이름                                                                           | `"test"` or `"validation"` | O (미설정 시 내부 분할)  |
| `skip_prepare_dataset` | `--skip_prepare_dataset`   | 데이터셋 전처리를 건너뛸지 여부. (`True` 시 추가 준비 로직을 생략)                                | `False`              | 부분적으로 코드 내 `dataset_kwargs`에서 사용 |
| `max_samples`          | `--max_samples`            | (추가 가능) 학습/평가 샘플 수를 제한하고자 할 때                                                 | `None` (무제한)      | 필요 시 사용 가능                      |
| `template_name`        | `--template_name`          | (추가 가능) 대화 템플릿 이름(예: `"vicuna_v1"`, `"alpaca"`)                                      | `None`               | 필요 시 사용 가능                      |
| ...                    |                             | (필요 시 원하는 커스텀 인자)                                                                     | -                     | -                                     |

---

## 2. `ModelConfig`

**모델 로딩** 및 **양자화(quantization), PEFT** 관련 설정, 그리고 **remote code** 사용 여부 등.

| **필드명**               | **CLI 인자**                   | **설명**                                                                                                            | **기본값**        | **사용 여부**                                                       |
|--------------------------|--------------------------------|---------------------------------------------------------------------------------------------------------------------|-------------------|-----------------------------------------------------------------------|
| `model_name_or_path`     | `--model_name_or_path`         | 로드할 **모델(체크포인트)** 경로 또는 HF Hub 식별자                                                                 | `None` (필수 인자)| O (예: `llava-hf/llava-1.5-7b-hf`)                                   |
| `trust_remote_code`      | `--trust_remote_code`          | Hugging Face Hub에서 제공되는 **커스텀 코드**(ex. `modeling_*.py`)를 신뢰하고 로드할지 여부                        | `False`           | O                                                                     |
| `torch_dtype`            | `--torch_dtype`                | 사용할 PyTorch dtype. `"float16"`, `"bfloat16"`, `"auto"` 등                                                        | `"auto"`          | 예: `--torch_dtype bfloat16`                                          |
| `model_revision`         | `--model_revision`             | 모델 저장소에서 특정 브랜치/커밋/태그를 로드할 때 사용                                                              | `"main"`          | 필요 시 사용 가능                                                     |
| `attn_implementation`    | `--attn_implementation`        | 어텐션 구현 방식 (예: `"flash"`, `"xformers"`, `"default"`)                                                         | `"default"`       | 필요 시 사용 가능                                                     |
| `load_in_4bit`           | `--load_in_4bit`               | 4비트 양자화 로드 여부                                                                                              | `False`           | 양자화 사용 시 True                                                   |
| `load_in_8bit`           | `--load_in_8bit`               | 8비트 양자화 로드 여부                                                                                              | `False`           | 양자화 사용 시 True                                                   |
| `peft_type`              | `--peft_type`                  | (추가 가능) PEFT 기법(LoRA, Prefix Tuning 등) 직접 지정                                                              | `"lora"` (예시)   | `get_peft_config(model_args)` 내 사용                                  |
| `quant_target_bits`      | `--quant_target_bits`          | (추가 가능) 4, 8 등 양자화 비트수 강제 지정                                                                           | `None`            | 필요 시 사용 가능                                                     |
| `device_map`             | (CLI 직접 없음)                | 2+ GPU 환경에서 어떤 레이어를 어느 디바이스에 배치할지 정하는 맵 (Auto로 설정 가능)                                  | `"auto"` or `None`| 내부 `get_kbit_device_map()` 등에서 자동 처리                          |
| ...                      |                                | 기타 사용자 정의 파라미터                                                                                           | -                 | -                                                                     |

---

## 3. `SFTConfig` (내부적으로 **Hugging Face `TrainingArguments`** 확장)

`SFTTrainer`가 사용하는 **학습 설정**을 담고 있으며, 대부분은 **`transformers.TrainingArguments`**와 동일한 이름/기능


| **필드명**                        | **CLI 인자**                             | **설명**                                                                                                             | **기본값**                   | **사용 여부**                                                                                                                            |
|----------------------------------|------------------------------------------|---------------------------------------------------------------------------------------------------------------------|------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| **입출력/기초**                                                                                                                                                                                                                                                                                                                                 |
| `output_dir`                     | `--output_dir`                           | 모델 체크포인트, 로그 등을 저장할 **출력 디렉터리**                                                                 | `None` (필수)               | O (예: `--output_dir ./my-sft-checkpoints`)                                                                                                |
| `overwrite_output_dir`           | `--overwrite_output_dir`                 | 동일 디렉터리에 기존 결과물이 있어도 **덮어쓸지 여부**                                                              | `False`                     | 필요 시 사용 가능                                                                                                                         |
| `do_train`                       | `--do_train`                             | 학습 루프 실행 여부                                                                                                 | `True`                      | 일반적으로 `True`                                                                                                                        |
| `do_eval`                        | `--do_eval`                              | 평가 루프 실행 여부                                                                                                 | `False`                     | `evaluation_strategy != "no"` 일 때 내부적으로 `True`                                                                                     |
| **배치 사이즈/스텝**                                                                                                                                                                                                                                                                                                                              |
| `per_device_train_batch_size`    | `--per_device_train_batch_size`          | 각 디바이스(프로세스)별 **학습 배치 크기**                                                                          | `8` (예시)                  | O                                                                                                                                          |
| `per_device_eval_batch_size`     | `--per_device_eval_batch_size`           | 각 디바이스(프로세스)별 **평가 배치 크기**                                                                          | 동일하게 `8`                | 필요 시 사용 가능                                                                                                                         |
| `gradient_accumulation_steps`    | `--gradient_accumulation_steps`          | 기울기(gradient) 업데이트 전, **스텝 누적 횟수**                                                                   | `1`                         | 예: `8` → 실제 배치 크기 = 배치 \* 8                                                                                                      |
| **학습률/에폭**                                                                                                                                                                                                                                                                                                                                    |
| `learning_rate`                  | `--learning_rate`                        | **기본 학습률**                                                                                                     | `5e-5`                      | 일반적으로 많이 수정                                                                                                                      |
| `num_train_epochs`               | `--num_train_epochs`                     | 학습 에폭 수                                                                                                        | `3`                         |                                                                                                                                            |
| `max_steps`                      | `--max_steps`                            | 총 학습 스텝 수 (설정 시, 에폭보다 스텝 우선)                                                                       | `-1` (미사용)              |                                                                                                                                            |
| `warmup_steps`                   | `--warmup_steps`                         | 워밍업 스텝 수                                                                                                      | `0`                         |                                                                                                                                            |
| `lr_scheduler_type`              | `--lr_scheduler_type`                    | 러닝레이트 스케줄러 (ex: `linear`, `cosine`, `cosine_with_restarts` 등)                                             | `linear`                    |                                                                                                                                            |
| **평가/로그**                                                                                                                                                                                                                                                                                                                                     |
| `evaluation_strategy`            | `--evaluation_strategy`                  | 평가 실행 전략: `no`, `steps`, `epoch`                                                                              | `no`                        | 예: `steps` → 정기적으로 평가                                                                                                              |
| `eval_steps`                     | `--eval_steps`                           | 평가 전략이 `steps`일 때, **몇 스텝마다** 평가를 할지                                                               | `500` (예시)                |                                                                                                                                            |
| `logging_steps`                  | `--logging_steps`                        | 몇 스텝마다 **로그 출력**을 할지                                                                                    | `500`                       |                                                                                                                                            |
| `report_to`                      | `--report_to`                            | 어떤 로거를 사용할지 (ex: `tensorboard`, `wandb`, `comet_ml`)                                                      | `["tensorboard"]` or `None` | 예: `--report_to tensorboard`                                                                                                              |
| `logging_dir`                    | `--logging_dir`                          | 텐서보드 로그 저장 디렉터리                                                                                         | `"runs"`                    |                                                                                                                                            |
| `save_steps`                     | `--save_steps`                           | 몇 스텝마다 **체크포인트 저장**할지                                                                                 | `500`                       |                                                                                                                                            |
| `save_total_limit`               | `--save_total_limit`                     | 저장할 체크포인트 최대 개수 (초과 시 오래된 것 삭제)                                                                 | `None` (무제한)            |                                                                                                                                            |
| **정밀도/가속**                                                                                                                                                                                                                                                                                                                                    |
| `bf16`                           | `--bf16`                                 | **bfloat16 혼합 정밀도** 사용 여부                                                                                  | `False`                     | 필요 시 `True`                                                                                                                             |
| `fp16`                           | `--fp16`                                 | **float16 혼합 정밀도** 사용 여부                                                                                  | `False`                     | 필요 시 `True`                                                                                                                             |
| `gradient_checkpointing`         | `--gradient_checkpointing`               | **그라디언트 체크포인팅** 사용 여부                                                                                 | `False`                     | 코드에서 `True`로 자주 활용                                                                                                                |
| **허브/배포**                                                                                                                                                                                                                                                                                                                                     |
| `push_to_hub`                    | `--push_to_hub`                          | 학습 완료 후 Hugging Face Hub에 모델을 업로드할지 여부                                                               | `False`                     | 필요 시 `True`                                                                                                                             |
| `hub_model_id`                   | `--hub_model_id`                         | 업로드될 모델 이름(리포지토리 식별자)                                                                               | `None`                      | `--push_to_hub`와 함께 사용                                                                                                                 |
| `hub_token`                      | `--hub_token`                            | Hugging Face Hub 토큰(접근 권한)                                                                                    | `None`                      |                                                                                                                                            |
| **기타**                                                                                                                                                                                                                                                                                                                                          |
| `remove_unused_columns`          | `--remove_unused_columns`                | 데이터셋 중 모델에 필요 없는 컬럼 제거 여부                                                                          | `True`                      | 멀티모달에서 종종 `False` 필요                                                                                                             |
| `peft_config`                    | (별도 CLI 없음, 내부처리)               | **PEFT** 설정(LoRA 등). 보통 `get_peft_config(model_args)`로 로딩                                                    | -                          | 스크립트 내부에서 할당                                                                                                                    |
| `dataset_kwargs`                 | (별도 CLI 없음, 내부처리)               | `load_dataset`에 전달할 추가 파라미터                                                                               | `None` or `{}`             | 스크립트 내부에서 예: `{"skip_prepare_dataset": True}`                                                                                     |
| `seed`                           | `--seed`                                 | 랜덤 시드                                                                                                           | `42`                       | 재현성 위해 설정 가능                                                                                                                      |
| `weight_decay`                   | `--weight_decay`                         | 옵티마이저의 weight decay 값                                                                                        | `0.0`                      |                                                                                                                                            |
| `adam_beta1`, `adam_beta2`       | `--adam_beta1`, `--adam_beta2`          | Adam 옵티마이저 베타 값                                                                                             | `0.9`, `0.999`             |                                                                                                                                            |
| `adam_epsilon`                   | `--adam_epsilon`                         | Adam 옵티마이저 epsilon                                                                                             | `1e-8`                     |                                                                                                                                            |
| `max_grad_norm`                  | `--max_grad_norm`                        | 그래디언트 클리핑(clip) 할 때 최대 norm                                                                             | `1.0`                      |                                                                                                                                            |
| `deepspeed`                      | `--deepspeed`                            | DeepSpeed 설정 파일 경로                                                                                            | `None`                      | `accelerate launch`에서 주로 설정                                                                                                          |
| ...                              |                                          | **Hugging Face `TrainingArguments`**에 존재하는 기타 인자                                                           | -                          | 필요 시 사용 가능                                                                                                                          |

> **참고**  
> - 위 표에서 **기본값**은 일반적으로 Hugging Face `TrainingArguments`(v4.x)에서 제공되는 디폴트를 기준으로 표기
> - 사용자가 CLI로 `--some_arg` 형태로 전달하지 않으면, `TrainingArguments`의 **기본값**이 적용
> - `SFTConfig`는 `TrainingArguments`를 **직접 상속**하거나 래핑하기 때문에, HF Trainer의 모든 인자를 대부분 그대로 지원  
>   - 예: `load_best_model_at_end`, `metric_for_best_model`, `greater_is_better`, `optim`, `report_to`, `group_by_length`, `label_smoothing_factor`, 등도 사용 가능.

---

## 4. 요약

- **`ScriptArguments`**  
  - 데이터셋(이름, 스플릿) 및 스크립트 전용 인자.  
  - “최초 제공 코드”에서는 `--dataset_name`, `--dataset_train_split`, `--dataset_test_split` 등을 사용.

- **`ModelConfig`**  
  - 모델 체크포인트 경로, **양자화(load_in_8bit/4bit)**, dtype, remote code 등 **모델 로딩** 관련 설정.  
  - 필요 시 PEFT 타입(`--peft_type`), attention 구현 옵션, revision 등을 세부 조정 가능.

- **`SFTConfig`** (내부적으로 `TrainingArguments`)  
  - **학습(Training) 파라미터** 전반(**배치 사이즈, 학습률, 에폭, 혼합정밀도** 등).  
  - `--evaluation_strategy steps`, `--eval_steps`, `--gradient_checkpointing`, `--report_to tensorboard` 등의 인자를 통해 **평가/로그/체크포인트**를 조정.  
  - Hub 업로드(`--push_to_hub`)나 PEFT 설정(`peft_config`) 등도 포함.

