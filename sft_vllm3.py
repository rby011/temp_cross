"""
sft_vllm.py

- Full Fine-Tuning of Llama Vision-Instruct (~11B) model with TRL + Accelerate
- No internal default overrides. We rely on CLI arguments (e.g., from guild.yml) entirely.
- Fields are directly mapped from:
    1) ScriptArguments (dataset & script usage)
    2) ModelConfig     (model loading / quantization / remote code)
    3) SFTConfig       (TrainingArguments: LR, scheduler, logging, checkpoint, etc.)

Example run (Accelerate CLI):
  accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero3.yaml \
    sft_vllm.py \
    --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct \
    --dataset_name MyVisionInstructionDataset \
    --lr_scheduler_type cosine \
    ...
Or from Guild:
  guild run sft_vlm:train [flags...]
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

def main():
    # 1. Parse arguments from CLI (no defaults in code!)
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    # We do NOT override or fill in any defaults here.
    # All logic will rely on what user passes from CLI/guild.yml.

    # 2. Prepare torch_dtype from the string argument
    #    (e.g. "bfloat16" => torch.bfloat16, "float16" => torch.float16, etc.)
    if model_args.torch_dtype in ["auto", None]:
        torch_dtype = "auto"
    else:
        torch_dtype = getattr(torch, model_args.torch_dtype)

    # 3. Quantization config (for 4bit/8bit, if user sets load_in_4bit= or load_in_8bit= True)
    quantization_config = get_quantization_config(model_args)

    # If user sets load_in_4bit/8bit => device_map from get_kbit_device_map()
    device_map = None
    if quantization_config is not None:
        device_map = get_kbit_device_map()

    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=device_map,
        quantization_config=quantization_config,
        trust_remote_code=model_args.trust_remote_code,
    )

    # 4. Load Processor & Model
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )

    # 5. Data Collator
    def collate_fn(examples):
        # If user gave template_name, user can implement logic here. For now, just apply default.
        texts = [processor.apply_chat_template(e["messages"], tokenize=False) for e in examples]
        images = [e["images"] for e in examples]
        if isinstance(model, LlavaForConditionalGeneration):
            # Some LLaVA models do not handle multiple images
            images = [img[0] for img in images]

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()

        # Mask out pad tokens
        labels[labels == processor.tokenizer.pad_token_id] = -100
        # Mask out image tokens
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        labels[labels == image_token_id] = -100
        batch["labels"] = labels
        return batch

    # 6. Load Dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # If user wants skip_prepare_dataset, set training_args.dataset_kwargs
    training_args.dataset_kwargs = {"skip_prepare_dataset": script_args.skip_prepare_dataset}

    # If user sets max_samples => limit dataset size
    if script_args.max_samples is not None:
        max_s = script_args.max_samples
        if script_args.dataset_train_split in dataset:
            ds_train = dataset[script_args.dataset_train_split]
            if len(ds_train) > max_s:
                dataset[script_args.dataset_train_split] = ds_train.select(range(max_s))
        if script_args.dataset_test_split in dataset and training_args.evaluation_strategy != "no":
            ds_eval = dataset[script_args.dataset_test_split]
            if len(ds_eval) > max_s:
                dataset[script_args.dataset_test_split] = ds_eval.select(range(max_s))

    # 7. Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.evaluation_strategy != "no"
            else None
        ),
        processing_class=processor.tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # 8. Train
    trainer.train()

    # 9. Save final model
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        # If user sets hub_model_id, push the processor as well
        if trainer.accelerator.is_main_process and hasattr(training_args, "hub_model_id"):
            processor.push_to_hub(training_args.hub_model_id)


if __name__ == "__main__":
    main()
