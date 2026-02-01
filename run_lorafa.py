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
from LiMELoRAFA import apply_peft


cache_dir = "/hf_cache"
model_name = "llava-onevision-qwen2-7b-ov-hf"
save_csv="evaluation_results_LiMELoRAFA.csv"
data_root="dataset"
dataset_name="MMT_47"




# Load processor
processor = LlavaOnevisionProcessor.from_pretrained(model_name, cache_dir=cache_dir)

# ⭐ CRITICAL: Set fixed resolution BEFORE using processor
processor.image_processor.image_grid_pinpoints = [[384, 384]]
processor.image_processor.size = {"height": 384, "width": 384}
processor.tokenizer.padding_side = "left" 

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



import torch

# Collect exactly 20 samples from the PROCESSED dataset
samples = []
for i in range(20):
    samples.append(train_dataset[i])  # ✅ Use train_dataset, not dataset['train']

# Track modalities before collating
modalities = [s['source_type'] for s in samples]

# Run through collator
batch = collator(samples)

# Inspect tensor shapes
print("="*70)
print("BATCH TENSOR SHAPES:")
print("="*70)
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"{k}: {v.shape}, dtype={v.dtype}")
    elif isinstance(v, list):
        print(f"{k}: list of {len(v)} items")

# Check modalities
print("\n" + "="*70)
print("SAMPLE MODALITIES:")
print("="*70)
for i, mod in enumerate(modalities):
    print(f"Sample {i}: {mod}")

# Count by modality
from collections import Counter
mod_counts = Counter(modalities)
print(f"\nModality distribution: {dict(mod_counts)}")

# Check visual data
print("\n" + "="*70)
print("VISUAL DATA INFO:")
print("="*70)
if 'pixel_values' in batch:
    print(f"pixel_values (images): {batch['pixel_values'].shape}")
    print(f"  - Number of image samples in batch: {modalities.count('image')}")
else:
    print("No images in this batch")

if 'pixel_values_videos' in batch:
    print(f"pixel_values_videos: {batch['pixel_values_videos'].shape}")
    print(f"  - Number of video samples in batch: {modalities.count('video')}")
else:
    print("No videos in this batch")

if 'image_sizes' in batch:
    print(f"image_sizes: {batch['image_sizes'].shape}")

# Decode and inspect
print("\n" + "="*70)
print("DECODED TEXT (last 150 tokens of each sample):")
print("="*70)

for i in range(min(10, len(samples))):  # First 10 for readability
    modality = modalities[i]
    
    # Get last 150 tokens
    last_tokens = batch["input_ids"][i][-150:]
    decoded = processor.tokenizer.decode(last_tokens, skip_special_tokens=False)
    
    # Get labels
    labels = batch["labels"][i]
    label_tokens = labels[labels != -100]
    label_text = processor.tokenizer.decode(label_tokens, skip_special_tokens=False) if len(label_tokens) > 0 else "[NO LABELS]"
    
    print(f"\n{'='*70}")
    print(f"SAMPLE {i} | Modality: {modality}")
    print(f"{'='*70}")
    print(f"Last 150 tokens decoded:\n{decoded}")
    print(f"\n📝 LABELS (what model learns to predict):\n{label_text}")
    print(f"Label token count: {len(label_tokens)}")

# Verify token-feature alignment for images
print("\n" + "="*70)
print("TOKEN-FEATURE ALIGNMENT CHECK:")
print("="*70)
image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
video_token_id = processor.tokenizer.convert_tokens_to_ids("<video>")

total_image_tokens = (batch['input_ids'] == image_token_id).sum().item()
total_video_tokens = (batch['input_ids'] == video_token_id).sum().item()

print(f"Total <image> tokens in batch: {total_image_tokens}")
print(f"Total <video> tokens in batch: {total_video_tokens}")

if 'pixel_values' in batch:
    # For fixed 384x384, each image = 1 patch = 729 features (27x27) after pooling
    # But depends on model config
    print(f"pixel_values shape: {batch['pixel_values'].shape}")
    
if 'pixel_values_videos' in batch:
    print(f"pixel_values_videos shape: {batch['pixel_values_videos'].shape}")
    
    


targets = ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj"]


def lm_only_targets(path, module):
    # Only wrap modules under model.language_model
    if not path.startswith("model.language_model."):
        return False

    # Expect paths like: model.language_model.layers.<idx>.*
    if ".layers." not in path:
        return False

    try:
        layer_id = int(path.split(".layers.")[1].split(".")[0])
    except (IndexError, ValueError):
        return False


    return any(path.endswith(name) for name in targets)

model = apply_peft(
    model,
    targets=targets,
    num_experts=4,
    rank=2,
    use_shared_LiME=True,
    n_gram=1,
    top_k=1,
    rep_mode="token",
    jitter_noise=0.1,
    tokenizer=processor.tokenizer,
    temperature=0.5,
    gamma_routing = 0.7, 
    auto_topk=True, 
    auto_topk_threshold=0.5, 
    peft_dtype=torch.float32,   # A, B in float32
    moe_dtype=torch.float32,    # moe3s, gamma in float32

)

print_lime_summary(model)

# 1) Count params that require grad
trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
total = sum(p.numel() for _, p in model.named_parameters())
trainable_num = sum(p.numel() for _, p in trainable)
print(f"trainable params: {trainable_num:,} / {total:,}")



#model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
training_args = LiMEArguments(
    output_dir="./llava-lorafa-finetuned",
    per_device_train_batch_size=5,
    gradient_accumulation_steps=4,  
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
    disable_tqdm=False,          # makes it print log lines instead of tqdm bar behavior
    
    #log_level="info",
    #logging_first_step=True,
    moe_lr=1e-3,          # For propulsions, gamma (float32)
    peft_lr=4e-4,         # For LoRA A/B (float32)
    importance_coef=0.1,
    kl_coef=0.01,
    peft_type="auto",
    balance_every_n_steps=50,

  
)

# Also make sure model doesn't have it enabled
model.gradient_checkpointing_disable()  # ⭐ Call this explicitly



# Example instantiation:
trainer = LiMETrainer(
    model=model,
    args=training_args,                  # your HF TrainingArguments
    train_dataset=train_dataset,

    tokenizer=processor.tokenizer,
    data_collator=collator,  # ✅ Custom collator dynamically pads batch sequences


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
