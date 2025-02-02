"""
sft_vlm.py

Example TRL script for Full Fine-Tuning Llama Vision-Instruct (~11B) model on an 8×80GB GPU environment,
using the recommended settings from the table. This script is intended to be launched via:

    accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero3.yaml sft_vlm.py ...

or via Guild AI:

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


if __name__ == "__main__":
    # ----------------
    # Parse arguments
    # ----------------
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # We apply some recommended default overrides for Full FT with large Vision-Llama model
    # (Only if the user hasn't explicitly set them)
    if training_args.per_device_train_batch_size is None:
        training_args.per_device_train_batch_size = 4
    if training_args.gradient_accumulation_steps is None:
        training_args.gradient_accumulation_steps = 4
    if training_args.learning_rate is None:
        training_args.learning_rate = 5e-5
    if training_args.weight_decay is None:
        training_args.weight_decay = 0.01
    if training_args.lr_scheduler_type is None:
        training_args.lr_scheduler_type = "cosine"
    if training_args.warmup_steps is None:
        training_args.warmup_steps = 1000
    if training_args.remove_unused_columns is None:
        training_args.remove_unused_columns = False
    if training_args.evaluation_strategy is None:
        training_args.evaluation_strategy = "steps"
    if training_args.eval_steps is None and training_args.evaluation_strategy == "steps":
        training_args.eval_steps = 500
    if training_args.logging_steps is None:
        training_args.logging_steps = 100
    if training_args.save_steps is None:
        training_args.save_steps = 2000
    if training_args.save_total_limit is None:
        training_args.save_total_limit = 3
    if training_args.output_dir is None:
        training_args.output_dir = "sft-llama3.2-11b-full"

    # For big models, some recommended settings
    # gradient_checkpointing: True (if not set)
    if not hasattr(training_args, "gradient_checkpointing") or training_args.gradient_checkpointing is None:
        training_args.gradient_checkpointing = True

    # Optionally set the optimizer to fused AdamW if desired (to reflect "AdamW(Fused)" usage).
    # training_args.optim = "adamw_torch_fused"  # or "adamw_torch" if environment supports
    # (If you want to ensure a fused version is used, HF Transformers >=4.30+ might pick automatically.)

    # Additional advanced usage
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    # skip dataset preparation
    training_args.dataset_kwargs = {"skip_prepare_dataset": script_args.skip_prepare_dataset}

    # ---------------
    # Prepare model
    # ---------------
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    # Because Full FT -> load_in_4bit / load_in_8bit is typically False
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs
    )

    # ---------------
    # Collate fn
    # ---------------
    def collate_fn(examples):
        # If you use script_args.template_name, you could add logic here to switch templates
        # For now, we just do a default template
        texts = [processor.apply_chat_template(e["messages"], tokenize=False) for e in examples]
        images = [e["images"] for e in examples]
        if isinstance(model, LlavaForConditionalGeneration):
            # LLaVA-based models often do not support multiple images
            images = [img[0] for img in images]

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        labels[labels == image_token_id] = -100
        batch["labels"] = labels
        return batch

    # ---------------
    # Load dataset
    # ---------------
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    if script_args.max_samples is not None:
        max_s = script_args.max_samples
        # limit train split
        if script_args.dataset_train_split in dataset:
            train_ds = dataset[script_args.dataset_train_split]
            if len(train_ds) > max_s:
                dataset[script_args.dataset_train_split] = train_ds.select(range(max_s))
        # limit eval split
        if (script_args.dataset_test_split in dataset) and (training_args.evaluation_strategy != "no"):
            eval_ds = dataset[script_args.dataset_test_split]
            if len(eval_ds) > max_s:
                dataset[script_args.dataset_test_split] = eval_ds.select(range(max_s))

    # ---------------
    # Trainer
    # ---------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split] if training_args.evaluation_strategy != "no" else None
        ),
        processing_class=processor.tokenizer,
        peft_config=get_peft_config(model_args),  # Full FT => will do minimal/ no PEFT if not set
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        if trainer.accelerator.is_main_process and hasattr(training_args, "hub_model_id"):
            processor.push_to_hub(training_args.hub_model_id)
