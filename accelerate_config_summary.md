# Accelerate 설정 (deepspeed_zero3.yaml) 요약

아래 표들은 Accelerate의 설정 파일(특히 DeepSpeed Stage3 예시)을 중심으로, **최상위 옵션**(Top-level)과 **`deepspeed_config` 내부 옵션**을 간략히 나열합니다.

## 최상위 옵션 (Top-level)

| **필드명**              | **설명**                                                                                                         | **예시 / 기본값**        |
|-------------------------|------------------------------------------------------------------------------------------------------------------|--------------------------|
| `compute_environment`   | 학습 환경을 지정. <br>- `LOCAL_MACHINE`: 단일 머신 <br>- `MACHINE` 또는 `AMAZON_SAGEMAKER` 등 다른 옵션 가능       | `LOCAL_MACHINE`         |
| `debug`                | 디버그 모드로 추가 로그/체크 활성화 여부                                                                          | `false`                 |
| `distributed_type`      | 분산 학습 방식. <br>- `DEEPSPEED`, `MULTI_GPU`, `TPU` 등                                                         | `DEEPSPEED`             |
| `downcast_bf16`        | bfloat16 다운캐스팅 사용 여부. <br>보통 `'no'`로 두어 GPU가 자동 지원하도록 함                                    | `'no'`                  |
| `machine_rank`          | 멀티 노드 환경에서 **현재 노드 랭크**. <br>0부터 시작                                                            | `0`                     |
| `main_training_function`| 실제 훈련을 진행하는 메인 함수 이름                                                                              | `"main"`                |
| `mixed_precision`       | 혼합 정밀도 설정: <br>- `"no"`, `"fp16"`, `"bf16"`                                                               | `bf16`                  |
| `num_machines`          | 전체 머신(노드) 수                                                                                                | `1`                     |
| `num_processes`         | 전체 프로세스(=GPU) 수. <br>단일 머신에서 8 GPU면 `8`                                                            | `8`                     |
| `rdzv_backend`          | 프로세스 동기화(Rendezvous) 백엔드. <br>`"static"`, `"c10d"`, `"etcd"` 등                                        | `"static"`              |
| `same_network`          | 여러 노드가 같은 네트워크인지 여부                                                                                | `true`                  |
| `tpu_env`, `tpu_use_*`  | TPU 관련 설정. 대부분 TPU를 안 쓰면 `[]`, `false`                                                                | `[]`, `false`           |
| `use_cpu`               | CPU만 사용할지 여부. <br>`false`면 GPU/TPU 활용                                                                   | `false`                 |

---

## `deepspeed_config` 내부 옵션

| **필드명**                   | **설명**                                                                                         | **예시 / 기본값** |
|-----------------------------|--------------------------------------------------------------------------------------------------|-------------------|
| `deepspeed_multinode_launcher` | 멀티 노드 시 DeepSpeed 런처 방식. <br>`standard`는 일반적인 실행                                | `standard`        |
| `offload_optimizer_device`  | 옵티마이저 상태를 CPU/NVMe로 오프로딩할지 여부. <br>`none`이면 오프로딩 안 함                     | `none`            |
| `offload_param_device`      | 파라미터(가중치) 오프로딩 대상. <br>`none`이면 오프로딩 안 함                                      | `none`            |
| `zero3_init_flag`           | ZeRO Stage 3 관련 초기화 플래그. <br>`true`면 모델/옵티마이저를 레이어 단위로 분산 로드           | `true`            |
| `zero3_save_16bit_model`    | 체크포인트 저장 시 16비트(fp16) 형태로 병합/저장할지                                             | `true`            |
| `zero_stage`                | ZeRO 최적화 단계 (0~3). <br>`3`은 가중치·옵티마이저·그래디언트 모두 분산                           | `3`               |

---

### 사용 예시

```bash
accelerate launch \
  --config_file deepspeed_zero3.yaml \
  my_script.py \
  --model_name_or_path my_model \
  --per_device_train_batch_size 4 \
  ...
