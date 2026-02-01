from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor
from utils import MultiModalDataset, MultiModalCollator
import torch
from torch.utils.data import Dataset
from PIL import Image
import av
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
import os

from trainer import print_lime_summary, LiMEArguments, LiMETrainer
from transformers import TrainingArguments
from evaluation import run_full_evaluation
from datasets import load_from_disk
from transformers import TrainingArguments, Trainer
from torch import nn

# ⭐ Import with trainer patterns
from LiMEPromtTuning import apply_peft, get_trainer_patterns, print_prompt_tuning_summary


cache_dir = "/hf_cache"
model_name = "llava-onevision-qwen2-7b-ov-hf"
save_csv="evaluation_results_LiMEPromtTuning.csv"
data_root="/dataset"
dataset_name="MMT_47"






# Load processor
processor = LlavaOnevisionProcessor.from_pretrained(model_name, cache_dir=cache_dir)

# ⭐ CRITICAL: Set fixed resolution BEFORE using processor
processor.image_processor.image_grid_pinpoints = [[384, 384]]
processor.image_processor.size = {"height": 384, "width": 384}
processor.tokenizer.padding_side = "right" 

if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

# Load model
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=cache_dir,
)

# ⭐ CRITICAL: Update model config to match
model.config.image_grid_pinpoints = [[384, 384]]

# ⭐ VERIFY the settings
print("="*50)
print("VERIFICATION:")
print("="*50)
print(f"Processor image_grid_pinpoints: {processor.image_processor.image_grid_pinpoints}")
print(f"Model config image_grid_pinpoints: {model.config.image_grid_pinpoints}")
print(f"Vision config image_size: {model.config.vision_config.image_size}")
print("="*50)

# ⭐ Get the correct hidden size for prompt tuning
hidden_size = model.config.text_config.hidden_size
print(f"Language model hidden_size: {hidden_size}")

model.gradient_checkpointing_enable()

# Load dataset
dataset = load_from_disk(dataset_name)

train_dataset = MultiModalDataset(
    dataset=dataset['train'],
    processor=processor,
    data_root=data_root,
    num_video_frames=8,
    max_length=2048,
)
collator = MultiModalCollator(
    processor=processor,
    max_length=2048,
    padding_side='right',
)

# ============ TEST FIRST ============
print("\n" + "="*50)
print("TESTING SINGLE SAMPLES:")
print("="*50)

# Test one of each modality
for i in range(min(50, len(train_dataset))):
    sample = train_dataset[i]
    src_type = sample.get('source_type', 'unknown')
    print(f"\nSample {i} ({src_type}):")
    print(f"  input_ids: {sample['input_ids'].shape}")
    if 'pixel_values' in sample:
        print(f"  pixel_values: {sample['pixel_values'].shape}")
    if 'image_sizes' in sample:
        print(f"  image_sizes: {sample['image_sizes']}")
    if 'pixel_values_videos' in sample:
        print(f"  pixel_values_videos: {sample['pixel_values_videos'].shape}")
    
    # Stop after finding one of each
    if i > 10:
        break

# Test collator with small batch
print("\n" + "="*50)
print("TESTING COLLATOR:")
print("="*50)

test_samples = [train_dataset[i] for i in range(4)]
test_batch = collator(test_samples)
print("Batch keys:", list(test_batch.keys()))
for k, v in test_batch.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape}")

# ⭐ Apply LiME Prompt Tuning with CORRECT hidden_size
prompt_module = apply_peft(
    model,
    num_virtual_tokens=10,
    num_experts=4,
    hidden_size=hidden_size,  # ⭐ AUTO-DETECTED from model config
    auto_topk = True, 
    use_shared_LiME=True,
    n_gram=1,
    jitter_noise=0.1,
    temperature=1.0,
    gamma_routing=0.5,
    sparse_mode=False,
    input_conditioned=True,
    freeze_model=True,
    conditioning_projection=False,
)

# ⭐ Print prompt tuning summary
print_prompt_tuning_summary(prompt_module)

# 1) Count params that require grad
trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
total = sum(p.numel() for _, p in model.named_parameters())
trainable_num = sum(p.numel() for _, p in trainable)
print(f"trainable params: {trainable_num:,} / {total:,}")

# ⭐ Debug: Show what params are trainable
print("\n" + "="*50)
print("TRAINABLE PARAMETERS:")
print("="*50)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  {name}: {param.shape}")


# ⭐ Get trainer patterns for prompt tuning
peft_patterns, moe_patterns = get_trainer_patterns()

training_args = LiMEArguments(
    output_dir="./llava-lime-prompt-tuning",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,  
    save_total_limit=2,
    save_steps=500000,
    num_train_epochs=4,
    bf16=True,  
    logging_dir="./logs",
    logging_steps=100,
    remove_unused_columns=False, 
    eval_steps=100,
    save_strategy="steps",
    optim="adamw_bnb_8bit",
    learning_rate=2e-4,
    warmup_ratio=0.03,
    weight_decay=0.01,
    report_to=[],
    disable_tqdm=False,
    
    # ⭐ LiME specific settings
    moe_lr=1e-3,
    peft_lr=2e-4,
    importance_coef=0.1,
    kl_coef=0.01,
    balance_every_n_steps=50,
    
    # ⭐ CRITICAL: Add patterns for prompt tuning detection
    custom_peft_patterns=peft_patterns,  # ['prompt_delta', 'input_proj']
    custom_moe_patterns=moe_patterns,    # ['LiMEs', 'LiME_shared', '.gamma']
)

# Example instantiation:
trainer = LiMETrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=processor.tokenizer,
    data_collator=collator,
)

# ⭐ Train!
trainer.train()


model.eval()

from evaluation import run_full_evaluation
results = run_full_evaluation(
    model=model,
    processor=processor,
    dataset=dataset,
    data_root=data_root,
    num_samples_per_split=None,
    batch_size=6,
    max_new_tokens=50, 
    save_csv=save_csv,
)
