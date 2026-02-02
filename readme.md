<div align="center">

<img src="assets/lime.png" alt="LiME" width="160">

# LiME: Lightweight Mixture of Experts for Efficient Multimodal Multi-task Learning

[![Paper](https://img.shields.io/badge/arXiv-2510.08513-b31b1b.svg)](https://arxiv.org/abs/2510.08513)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-MMT--47-yellow)](https://huggingface.co/datasets/Kowsher/MMT-47)
[![Conference](https://img.shields.io/badge/ICML-2026-4b44ce.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Expert specialization through lightweight modulation vectors and zero-parameter routing*

[Paper](https://arxiv.org/abs/2510.08513) · [Project Page](https://kowsher.github.io/LiME/) · [Dataset](https://huggingface.co/datasets/Kowsher/MMT-47)

</div>

---

## Overview

LiME is a parameter-efficient Mixture-of-Experts (MoE) framework that achieves expert specialization **without** replicating full adapters per expert or learning separate router weights. It works by:

1. **Sharing a single PEFT adapter** (e.g., LoRA) across all experts
2. **Modulating** the shared adapter with lightweight per-expert scaling vectors
3. **Routing tokens to experts** using a zero-parameter mechanism derived from frozen and adapted hidden representations

This design reduces trainable MoE parameters by up to **4×** compared to existing methods while maintaining competitive or superior performance across 47 multimodal tasks.

<div align="center">
<img src="assets/method.png" alt="LiME Architecture" width="100%">

**Figure 1.** LiME architecture — (a) Standard MoE-LoRA, (b) LiME shared adapter + modulation, (c) AutoTop-K routing, (d) Load balancing losses.
</div>

## Key Features

- **🧬 Shared Adapter + Modulation** — A single LoRA adapter modulated by lightweight per-expert vectors (`p_i ∈ ℝ^{d_o}`), reducing adapter parameters from `E × |ϕ|` to `|ϕ| + E × d_o`
- **⚡ Zero-Parameter Routing** — Expert routing via n-gram hidden-state similarity between frozen and adapted representations — no learned gating weights
- **🎯 AutoTop-K** — Dynamically selects active experts per token based on confidence thresholds (`w_i ≥ θ · max_j w_j`), enabling adaptive computation
- **⚖️ Load Balancing** — Importance loss + KL-uniform divergence prevents expert collapse and encourages even utilization
- **🔌 PEFT-Agnostic** — Compatible with LoRA, DoRA, LoRA-FA, SliceFine, Prompt Tuning, and more

## Results

LiME variants achieve competitive or superior performance to MoE-PEFT baselines while requiring significantly fewer trainable parameters:

<div align="center">
<img src="assets/result.png" alt="Results" width="100%">
</div>

| Method | #Params | Vision | Image Clf. | Commonsense | GLUE | HLR | Motion & Spatial | Action |
|:-------|--------:|:------:|:----------:|:-----------:|:----:|:---:|:----------------:|:------:|
| LoRA | 1.74M | 77.02 | 93.92 | 83.80 | 90.64 | 43.23 | 62.85 | 50.99 |
| MoELoRA | 10.79M | 77.27 | 93.97 | 84.08 | **91.21** | 43.84 | 63.07 | 51.13 |
| **LiMELoRA** | **3.49M** | 78.01 | 94.47 | **84.98** | 91.02 | 45.00 | **65.05** | 53.19 |
| **LiMEDoRA** | **3.84M** | **78.12** | 94.50 | 84.76 | 91.11 | **45.65** | 65.41 | **53.39** |

<div align="center">
<img src="assets/efi.png" alt="Efficiency" width="100%">

**Figure 2.** LiME achieves higher throughput (4.52 samples/s), lower memory, and up to 4× fewer trainable parameters.
</div>

## Installation

```bash
git clone https://github.com/Kowsher/LiME.git
cd LiME
pip install -r requirements.txt
```

### Dependencies

- Python ≥ 3.9
- PyTorch ≥ 2.0
- Transformers ≥ 4.40
- datasets
- bitsandbytes (for 8-bit optimizer)
- av (for video processing)
- Pillow

## Quick Start

### 1. Apply LiME to Any Model

```python
from LiMELoRA import apply_peft
from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor
import torch

# Load model and processor
model_name = "llava-onevision-qwen2-7b-ov-hf"
processor = LlavaOnevisionProcessor.from_pretrained(model_name)
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Apply LiME
model = apply_peft(
    model,
    targets=["q_proj", "k_proj", "v_proj", "o_proj", "out_proj"],
    num_experts=4,            # Number of experts
    rank=2,                   # LoRA rank
    use_shared_LiME=True,    # Share adapter across experts
    n_gram=1,                 # N-gram window for routing
    top_k=1,                  # Top-K expert selection
    rep_mode="token",         # Routing representation mode
    jitter_noise=0.1,         # Routing noise for exploration
    tokenizer=processor.tokenizer,
    temperature=0.5,          # Routing softmax temperature
    gamma_routing=0.7,        # Frozen vs adapted blend ratio
    auto_topk=True,           # Enable AutoTop-K
    auto_topk_threshold=0.5,  # AutoTop-K confidence threshold
    peft_dtype=torch.float32, # Precision for LoRA A, B
    moe_dtype=torch.float32,  # Precision for modulation vectors
)
```

### 2. Train with LiMETrainer

```python
from trainer import LiMEArguments, LiMETrainer, print_lime_summary
from utils import MultiModalDataset, MultiModalCollator
from datasets import load_from_disk

# Print model summary
print_lime_summary(model)

# Prepare data
dataset = load_from_disk("MMT_47")
train_dataset = MultiModalDataset(
    dataset=dataset['train'],
    processor=processor,
    data_root="dataset",
    num_video_frames=8,
    max_length=2048,
)
collator = MultiModalCollator(processor=processor, max_length=2048)

# Configure training with separate LRs for MoE and PEFT components
training_args = LiMEArguments(
    output_dir="./llava-lime-finetuned",
    per_device_train_batch_size=5,
    gradient_accumulation_steps=4,
    num_train_epochs=4,
    bf16=True,
    learning_rate=2e-4,
    optim="adamw_bnb_8bit",
    warmup_ratio=0.03,
    weight_decay=0.01,
    # LiME-specific arguments
    moe_lr=1e-3,                # LR for modulation vectors & gamma
    peft_lr=4e-4,               # LR for LoRA A/B matrices
    importance_coef=0.1,        # Importance loss coefficient
    kl_coef=0.01,               # KL-uniform loss coefficient
    balance_every_n_steps=50,   # Load balancing frequency
)

trainer = LiMETrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=processor.tokenizer,
    data_collator=collator,
)

trainer.train()
```

### 3. Evaluate

```python
from evaluation import run_full_evaluation

model.eval()
results = run_full_evaluation(
    model=model,
    processor=processor,
    dataset=dataset,
    data_root="dataset",
    num_samples_per_split=None,
    batch_size=6,
    max_new_tokens=50,
    save_csv="evaluation_results.csv",
)
```

## Dataset: MMT-47

MMT-47 is a comprehensive multimodal multi-task benchmark with **47 tasks** spanning 7 categories:

| Category | Tasks | Examples |
|:---------|------:|:---------|
| Vision Benchmark | 6 | VQAv2, GQA, TextVQA, POPE, MMBench, ScienceQA |
| Image Classification | 8 | CIFAR-10, Food101, Oxford Pets, DTD, EuroSAT, ... |
| Commonsense Reasoning | 6 | PIQA, ARC, HellaSwag, WinoGrande, BoolQ, CSQA |
| GLUE | 8 | SST-2, MNLI, QNLI, QQP, RTE, CoLA, MRPC, STS-B |
| High-Level Reasoning | 5 | GSM8K, MATH, StrategyQA, AQuA, LogiQA |
| Object Motion & Spatial | 7 | CLEVR, spatial reasoning, motion detection, ... |
| Action Understanding | 7 | MVTamperBench video tasks |

### Download

```bash
# Download dataset metadata
pip install datasets
python -c "from datasets import load_dataset; load_dataset('Kowsher/MMT-47')"

# Download images
huggingface-cli download \
  Kowsher/MMT-47 \
  --repo-type dataset \
  --include "images/*" \
  --local-dir images/

# Extract images
cd images && unzip images.zip && cd ..
```

### Download Video Data

Video samples are sourced from [MVTamperBench](https://huggingface.co/datasets/Srikant86/MVTamperBench):

```bash
huggingface-cli download \
  Srikant86/MVTamperBench \
  --repo-type dataset \
  --include "video/*" \
  --local-dir videos/

# Extract all video zip files
cd videos/
for f in *.zip; do
  d="${f%.zip}"
  if [ -d "$d" ]; then
    echo "Skipping $f (already extracted)"
  else
    echo "Extracting $f"
    unzip "$f" -d "$d"
  fi
done
cd ..
```

> **⚠️ License Notice:** MMT-47 aggregates data from multiple existing datasets, each governed by its own license. By using MMT-47, you agree to respect and comply with the individual license terms of every constituent dataset. Please review the original dataset licenses before use. See the [dataset card](https://huggingface.co/datasets/Kowsher/MMT-47) for the full list of sources and their respective licenses.

## Project Structure

```
LiME/
├── LiMELoRA.py          # Core LiME implementation (apply_peft)
├── trainer.py            # LiMETrainer & LiMEArguments
├── evaluation.py         # Evaluation pipeline
├── utils.py              # MultiModalDataset & MultiModalCollator
├── run_lora.py           # Full training script (LiMELoRA)
├── requirements.txt
└── assets/
    ├── lime.png          # Project logo
    ├── method.png        # Architecture figure
    ├── result.png        # Results table
    └── efi.png           # Efficiency comparison
```

## LiME Variants

LiME is PEFT-agnostic. We provide implementations for:

| Variant | PEFT Base | Description |
|:--------|:----------|:------------|
| **LiMELoRA** | LoRA | Shared LoRA + expert modulation |
| **LiMEDoRA** | DoRA | Shared DoRA + expert modulation |
| **LiMELoRA-FA** | LoRA-FA | Frozen-A LoRA + expert modulation |
| **LiMESliceFine** | SliceFine | Shared SliceFine + expert modulation |
| **LiMEPromptTuning** | Prompt Tuning | Shared soft prompts + expert modulation |

## Hyperparameters

Key hyperparameters and their recommended values:

| Parameter | Default | Description |
|:----------|--------:|:------------|
| `num_experts` | 4 | Number of experts |
| `rank` | 2 | LoRA rank |
| `n_gram` | 1 | N-gram routing window size |
| `top_k` | 1 | Number of active experts per token |
| `temperature` | 0.5 | Routing softmax temperature |
| `gamma_routing` | 0.7 | Blend ratio (frozen vs adapted) |
| `auto_topk_threshold` | 0.5 | AutoTop-K confidence threshold |
| `jitter_noise` | 0.1 | Routing exploration noise |
| `moe_lr` | 1e-3 | Learning rate for MoE components |
| `peft_lr` | 4e-4 | Learning rate for PEFT parameters |
| `importance_coef` | 0.1 | Importance loss weight |
| `kl_coef` | 0.01 | KL-uniform loss weight |
| `balance_every_n_steps` | 50 | Load balancing frequency |

## Citation

If you find LiME useful in your research, please cite our paper:

```bibtex
@inproceedings{lime2026,
  title     = {LiME: Lightweight Mixture of Experts for
               Efficient Multimodal Multi-task Learning},
  author    = {[Authors]},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## Acknowledgements

We thank the authors of the constituent datasets in MMT-47 for making their data publicly available. This work builds on [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT), [LoRA](https://github.com/microsoft/LoRA), and the Hugging Face [Transformers](https://github.com/huggingface/transformers) library.

