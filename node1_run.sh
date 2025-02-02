# Node B(랭크=1) 실행
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=12345

accelerate launch \
  --config_file ds_config_node1.yaml \
  examples/scripts/sft_vlm.py \
  --model_name_or_path ... \
  --dataset_name ... \
  --num_train_epochs 3 \
  ...
