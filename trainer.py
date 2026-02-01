# lime_trainer.py
"""
LiME Trainer - Optimized with SliceFine Support

Combines:
- Document 1: Auto-detection of PEFT types (LoRA, DoRA, SliceFine, IA3)
- Document 2: Performance optimizations (balance_every_n_steps, deferred .item(), cached modules)

Features:
- Auto-detects PEFT type from model parameters
- balance_every_n_steps: Skip aux computation on non-balance steps
- set_aux_tracking(False): Tells LiME modules to skip aux computation
- Cached module list: No model.modules() iteration every step
- Deferred .item(): Only CUDA sync on logging steps
- Multi-GPU support

Usage:
    from lime_trainer import LiMETrainer, LiMEArguments
    
    args = LiMEArguments(
        output_dir="./out",
        max_steps=1000,
        balance_every_n_steps=4,  # Compute balance loss every 4 steps
        peft_type="auto",         # Auto-detect SliceFine/LoRA/DoRA
        ...
    )
    trainer = LiMETrainer(model=model, args=args, ...)
"""

from transformers import Trainer, TrainingArguments
from typing import Optional, Dict, List, Set, Union
from dataclasses import dataclass, field
import math
import torch
import torch.nn as nn


# ============================================================
# PEFT Pattern Detection
# ============================================================

PEFT_PATTERNS = {
    'lora': ['.A', '.B'],
    'dora': ['.A', '.B', '.m_vector', '.magnitude', '.m'],
    'slicefine': ['part_T'],
    'ia3': ['.ia3_'],
    'adalora': ['.lora_A', '.lora_B', '.lora_E'],
}

MOE_PATTERNS = ['LiMEs', 'LiME_shared', '.gamma', 'routing_head', 'router']


def detect_peft_type(model: nn.Module) -> str:
    """Auto-detect PEFT type from model parameters."""
    param_names = [name for name, _ in model.named_parameters()]
    
    # Check for SliceFine first (most specific)
    if any('part_T' in name and 'frozen' not in name for name in param_names):
        return 'slicefine'
    
    # Check for DoRA (has magnitude vector)
    if any('.m_vector' in name or '.magnitude' in name or name.endswith('.m') for name in param_names):
        return 'dora'
    
    # Check for LoRA
    if any('.A' in name or '.B' in name for name in param_names):
        return 'lora'
    
    # Check for IA3
    if any('.ia3_' in name for name in param_names):
        return 'ia3'
    
    return 'unknown'


def get_peft_patterns(peft_type: str = 'auto', model: nn.Module = None) -> List[str]:
    """Get PEFT parameter patterns for given type."""
    if peft_type == 'auto' and model is not None:
        peft_type = detect_peft_type(model)
    
    if peft_type in PEFT_PATTERNS:
        return PEFT_PATTERNS[peft_type]
    
    # Fallback: combine all known patterns
    all_patterns = []
    for patterns in PEFT_PATTERNS.values():
        all_patterns.extend(patterns)
    return list(set(all_patterns))


# ============================================================
# Module Utilities
# ============================================================

def find_lime_modules(model: nn.Module) -> List[nn.Module]:
    """
    Find all LiME modules in the model.
    
    LiME modules are identified by having both:
    - _last_aux: Storage for auxiliary losses (importance, kl, load)
    - track_aux: Flag to enable/disable aux computation
    
    This works with any LiME variant (LoRA-LiME, DoRA-LiME, SliceFine-LiME).
    """
    modules = []
    for m in model.modules():
        # Check for _StoresAux mixin (common to all LiME variants)
        if hasattr(m, '_last_aux') and hasattr(m, 'track_aux'):
            modules.append(m)
    return modules


def set_aux_tracking(lime_modules: List[nn.Module], enabled: bool):
    """
    Enable/disable aux computation in LiME modules.
    
    When disabled (enabled=False):
    - Modules skip _moe_balance_losses computation entirely
    - Forward pass is faster
    - No aux data is stored
    
    This is the KEY optimization for balance_every_n_steps.
    """
    for m in lime_modules:
        m.track_aux = enabled


def clear_aux(lime_modules: List[nn.Module]):
    """Clear aux data from all modules without collecting."""
    for m in lime_modules:
        m._last_aux = None
        #if hasattr(m, '_last_usage'):
            #m._last_usage = None


def collect_aux(lime_modules: List[nn.Module], device: torch.device) -> tuple:
    """
    Collect aux tensors from all LiME modules.
    
    This function:
    1. Iterates through all LiME modules
    2. Sums up importance, kl_uniform, and load losses
    3. Preserves usage stats for logging (stores in _last_usage)
    4. Clears _last_aux to prevent accumulation
    5. Returns tensors (no .item() calls) for deferred CUDA sync
    
    Args:
        lime_modules: List of LiME modules (from find_lime_modules)
        device: Target device for tensor operations (usually task_loss.device)
    
    Returns:
        (importance, kl, load, count) - all as tensors except count
    """
    imp = torch.tensor(0.0, device=device)
    kl = torch.tensor(0.0, device=device)
    load = torch.tensor(0.0, device=device)
    count = 0
    
    for m in lime_modules:
        aux = m._last_aux
        if aux is not None:
            # Importance loss
            v = aux.get("importance")
            if v is not None:
                if isinstance(v, torch.Tensor):
                    imp = imp + (v.to(device) if v.device != device else v)
                else:
                    imp = imp + v
            
            # KL uniform loss
            v = aux.get("kl_uniform")
            if v is not None:
                if isinstance(v, torch.Tensor):
                    kl = kl + (v.to(device) if v.device != device else v)
                else:
                    kl = kl + v
            
            # Load loss
            v = aux.get("load")
            if v is not None:
                if isinstance(v, torch.Tensor):
                    load = load + (v.to(device) if v.device != device else v)
                else:
                    load = load + v
            
            # IMPORTANT: Save usage before clearing aux (for logging)
            # This preserves expert usage stats for get_usage_stats()
            if "usage" in aux and aux["usage"] is not None:
                m._last_usage = aux["usage"].detach()
            
            # Clear aux to prevent accumulation across steps
            m._last_aux = None
            count += 1
    
    return imp, kl, load, count


def get_usage_stats(lime_modules: List[nn.Module]) -> Dict:
    """
    Get expert usage statistics for logging.
    
    Reads from _last_usage which is preserved by collect_aux().
    
    Returns dict with:
    - moe/entropy: How evenly distributed the expert usage is
    - moe/imbalance: Ratio of max to min usage
    - moe/max_usage: Highest expert usage
    - moe/min_usage: Lowest expert usage
    """
    usages = []
    for m in lime_modules:
        if hasattr(m, '_last_usage') and m._last_usage is not None:
            usages.append(m._last_usage.cpu())
    
    if not usages:
        return {}
    
    avg = torch.stack(usages).mean(dim=0)
    return {
        "moe/entropy": -(avg * torch.log(avg + 1e-9)).sum().item(),
        "moe/imbalance": avg.max().item() / (avg.min().item() + 1e-9),
        "moe/max_usage": avg.max().item(),
        "moe/min_usage": avg.min().item(),
    }


# ============================================================
# Parameter Groups
# ============================================================

def make_param_groups(
    model: nn.Module,
    moe_lr: float,
    peft_lr: float,
    weight_decay: float,
    peft_type: str = "auto",
    custom_peft_patterns: Optional[List[str]] = None,
    custom_moe_patterns: Optional[List[str]] = None,
    verbose: bool = False,
) -> tuple:
    """
    Create optimizer param groups with separate learning rates.
    
    Groups:
    - moe: LiMEs, gamma, routing params (higher LR, no weight decay)
    - peft: A, B, part_T, etc. (standard LR, with weight decay)
    - other: Any remaining trainable params
    
    Args:
        model: The model with LiME layers
        moe_lr: Learning rate for MoE parameters
        peft_lr: Learning rate for PEFT parameters
        weight_decay: Weight decay for PEFT/other params
        peft_type: "auto", "lora", "dora", "slicefine", "ia3"
        custom_peft_patterns: Override PEFT detection patterns
        custom_moe_patterns: Override MoE detection patterns
        verbose: Print detection info
    
    Returns:
        (groups, names_dict) where groups is list of param dicts
        and names_dict maps group name to parameter names.
    """
    # Determine patterns
    if custom_peft_patterns:
        peft_patterns = custom_peft_patterns
    else:
        peft_patterns = get_peft_patterns(peft_type, model)
    
    moe_patterns = custom_moe_patterns if custom_moe_patterns else MOE_PATTERNS
    
    moe_params, peft_params, other_params = [], [], []
    moe_names, peft_names, other_names = [], [], []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check MoE patterns first (highest priority)
        if any(x in name for x in moe_patterns):
            moe_params.append(param)
            moe_names.append(name)
        # Check PEFT patterns (exclude frozen copies)
        elif any(x in name for x in peft_patterns) and 'frozen' not in name:
            peft_params.append(param)
            peft_names.append(name)
        else:
            other_params.append(param)
            other_names.append(name)
    
    groups = []
    if moe_params:
        groups.append({
            'params': moe_params,
            'lr': moe_lr,
            'weight_decay': 0.0,  # No weight decay for MoE params
            'name': 'moe'
        })
    if peft_params:
        groups.append({
            'params': peft_params,
            'lr': peft_lr,
            'weight_decay': weight_decay,
            'name': 'peft'
        })
    if other_params:
        groups.append({
            'params': other_params,
            'lr': peft_lr,
            'weight_decay': weight_decay,
            'name': 'other'
        })
    
    names_dict = {'moe': moe_names, 'peft': peft_names, 'other': other_names}
    
    if verbose:
        detected = detect_peft_type(model)
        print(f"Detected PEFT type: {detected}")
        print(f"Using patterns - MoE: {moe_patterns}, PEFT: {peft_patterns}")
    
    return groups, names_dict


def print_param_summary(groups: List[Dict], names: Dict[str, List[str]]):
    """Print parameter group summary."""
    print("=" * 60)
    print("Parameter Groups")
    print("=" * 60)
    for g in groups:
        n = g.get('name', '?')
        num = sum(p.numel() for p in g['params'])
        print(f"[{n.upper()}] {len(g['params'])} tensors, {num:,} params, lr={g['lr']:.2e}")
        if n in names and names[n]:
            for ex in names[n][:3]:
                print(f"  - {ex}")
            if len(names[n]) > 3:
                print(f"  ... and {len(names[n]) - 3} more")
    print("=" * 60)


# ============================================================
# Training Arguments
# ============================================================

@dataclass
class LiMEArguments(TrainingArguments):
    """
    Training arguments for LiME with all optimizations.
    
    Inherits from TrainingArguments and adds:
    - Dual learning rate support (moe_lr, peft_lr)
    - Balance loss configuration
    - Performance optimizations (balance_every_n_steps)
    - PEFT type detection
    """
    
    # Learning rates
    moe_lr: float = field(
        default=1e-3, 
        metadata={"help": "Learning rate for MoE params (LiMEs, gamma, routing)"}
    )
    peft_lr: float = field(
        default=2e-4, 
        metadata={"help": "Learning rate for PEFT params (A, B, part_T, etc.)"}
    )
    
    # Balance loss coefficients
    balance_coef: float = field(
        default=0.01, 
        metadata={"help": "Overall balance loss coefficient"}
    )
    importance_coef: float = field(
        default=1.0, 
        metadata={"help": "Weight for importance loss component"}
    )
    kl_coef: float = field(
        default=0.1, 
        metadata={"help": "Weight for KL uniform loss component"}
    )
    
    # Warmup configuration
    balance_warmup_steps: int = field(
        default=0, 
        metadata={"help": "Warmup steps for balance coefficient (0 = no warmup)"}
    )
    balance_schedule: str = field(
        default="linear", 
        metadata={"help": "Warmup schedule: 'constant', 'linear', or 'cosine'"}
    )
    
    # SPEED OPTIMIZATION: compute balance every N steps
    balance_every_n_steps: int = field(
        default=1, 
        metadata={"help": "Compute balance loss every N steps (1=every step, 4=every 4th step)"}
    )
    
    # PEFT type (auto-detect or specify)
    peft_type: str = field(
        default="auto", 
        metadata={"help": "PEFT type: 'auto', 'lora', 'dora', 'slicefine', 'ia3'"}
    )
    
    # Custom patterns (optional override)
    custom_peft_patterns: Optional[List[str]] = field(
        default=None, 
        metadata={"help": "Custom PEFT param patterns (overrides auto-detection)"}
    )
    custom_moe_patterns: Optional[List[str]] = field(
        default=None, 
        metadata={"help": "Custom MoE param patterns (overrides defaults)"}
    )
    
    # Logging configuration
    log_expert_usage: bool = field(
        default=True, 
        metadata={"help": "Log expert usage statistics"}
    )
    log_expert_freq: int = field(
        default=100, 
        metadata={"help": "Frequency for logging expert stats (in steps)"}
    )
    moe_verbose: bool = field(
        default=False, 
        metadata={"help": "Verbose MoE logging during initialization"}
    )
    
    def __post_init__(self):
        # Set base learning_rate to peft_lr for compatibility
        self.learning_rate = self.peft_lr
        super().__post_init__()


# ============================================================
# LiME Trainer
# ============================================================

class LiMETrainer(Trainer):
    """
    Optimized LiME Trainer with SliceFine support.
    
    Main optimizations:
    1. balance_every_n_steps: Skip aux computation on non-balance steps
       - When N > 1, aux tracking is disabled on N-1 out of N steps
       - This skips _moe_balance_losses in the LiME modules entirely
    
    2. set_aux_tracking(False): Tells LiME modules to skip aux computation
       - Modules check track_aux flag before computing balance losses
       - Reduces forward pass overhead
    
    3. Cached module list: No model.modules() iteration every step
       - LiME modules are found once and cached
    
    4. Deferred .item(): Only CUDA sync on logging steps
       - Losses stored as tensors until logging
       - Avoids CPU-GPU sync on every step
    
    Loss formula:
        total_loss = task_loss + balance_coef * (importance_coef * importance + kl_coef * kl_uniform)
    
    Example:
        args = LiMEArguments(
            output_dir="./out",
            max_steps=1000,
            balance_every_n_steps=4,  # 4x speedup on balance computation
            peft_type="auto",
        )
        trainer = LiMETrainer(model=model, args=args, train_dataset=dataset)
        trainer.train()
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Extract config from args
        if isinstance(self.args, LiMEArguments):
            self._moe_lr = self.args.moe_lr
            self._peft_lr = self.args.peft_lr
            self._balance_coef = self.args.balance_coef
            self._importance_coef = self.args.importance_coef
            self._kl_coef = self.args.kl_coef
            self._balance_warmup = self.args.balance_warmup_steps
            self._balance_schedule = self.args.balance_schedule
            self._balance_every_n = self.args.balance_every_n_steps
            self._peft_type = self.args.peft_type
            self._custom_peft_patterns = self.args.custom_peft_patterns
            self._custom_moe_patterns = self.args.custom_moe_patterns
            self._log_expert = self.args.log_expert_usage
            self._log_freq = self.args.log_expert_freq
            self._verbose = self.args.moe_verbose
        else:
            # Fallback defaults for standard TrainingArguments
            self._moe_lr = 1e-3
            self._peft_lr = self.args.learning_rate
            self._balance_coef = 0.01
            self._importance_coef = 1.0
            self._kl_coef = 0.1
            self._balance_warmup = 0
            self._balance_schedule = "linear"
            self._balance_every_n = 1
            self._peft_type = "auto"
            self._custom_peft_patterns = None
            self._custom_moe_patterns = None
            self._log_expert = True
            self._log_freq = 100
            self._verbose = False
        
        # State (lazy initialized)
        self._lime_modules: Optional[List[nn.Module]] = None
        self._cycle_step = 0  # Tracks position in balance_every_n cycle
        
        # Loss tracking (tensors for deferred .item())
        self._last_task_loss: Optional[torch.Tensor] = None
        self._last_balance: Optional[torch.Tensor] = None
        self._last_importance: Optional[torch.Tensor] = None
        self._last_kl: Optional[torch.Tensor] = None
        self._last_load: Optional[torch.Tensor] = None
        self._cached_balance: Optional[torch.Tensor] = None  # For non-balance steps

    def _get_lime_modules(self) -> List[nn.Module]:
        """Lazily find and cache LiME modules."""
        if self._lime_modules is None:
            model = self.model
            if hasattr(self, 'accelerator') and self.accelerator:
                model = self.accelerator.unwrap_model(model)
            self._lime_modules = find_lime_modules(model)
            
            if self._verbose:
                print(f"Found {len(self._lime_modules)} LiME modules")
        
        return self._lime_modules

    def _get_balance_coef(self) -> float:
        """
        Get current balance coefficient with optional warmup.
        
        Schedules:
        - constant: Full coefficient immediately (no warmup effect)
        - linear: Linear ramp from 0 to balance_coef
        - cosine: Cosine ramp from 0 to balance_coef
        """
        if self._balance_warmup <= 0:
            return self._balance_coef
        
        step = self.state.global_step
        if step >= self._balance_warmup:
            return self._balance_coef
        
        progress = step / self._balance_warmup
        
        if self._balance_schedule == "constant":
            # No warmup - return full coefficient immediately
            return self._balance_coef
        elif self._balance_schedule == "linear":
            return self._balance_coef * progress
        elif self._balance_schedule == "cosine":
            # Smooth cosine ramp
            return self._balance_coef * (1 - math.cos(math.pi * progress)) / 2
        else:
            # Default to linear
            return self._balance_coef * progress

    def create_optimizer(self):
        """Create optimizer with separate LRs for MoE and PEFT params."""
        if self.optimizer is not None:
            return self.optimizer
        
        groups, names = make_param_groups(
            self.model,
            moe_lr=self._moe_lr,
            peft_lr=self._peft_lr,
            weight_decay=self.args.weight_decay,
            peft_type=self._peft_type,
            custom_peft_patterns=self._custom_peft_patterns,
            custom_moe_patterns=self._custom_moe_patterns,
            verbose=self._verbose,
        )
        
        # Print summary on main process only
        is_main = getattr(self.args, 'local_rank', 0) <= 0
        if is_main:
            detected = detect_peft_type(self.model)
            print(f"\n{'=' * 60}")
            print(f"LiME Trainer Initialization")
            print(f"{'=' * 60}")
            print(f"Detected PEFT type: {detected}")
            print(f"balance_every_n_steps: {self._balance_every_n}")
            print(f"balance_coef: {self._balance_coef}")
            print(f"balance_warmup_steps: {self._balance_warmup}")
            print(f"balance_schedule: {self._balance_schedule}")
            print_param_summary(groups, names)
        
        opt_cls, opt_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model)
        opt_kwargs.pop('lr', None)  # We set LR per group
        self.optimizer = opt_cls(groups, **opt_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute task loss + balance loss with optimizations.
        
        Flow:
        1. Check if this is a "balance step" (based on cycle)
        2. Enable/disable aux tracking accordingly
        3. Forward pass (computes task loss, optionally aux)
        4. If balance step: collect aux, compute balance loss
        5. Return total loss
        """
        lime_modules = self._get_lime_modules()
        
        # Check if we compute balance this step
        do_balance = (self._cycle_step == 0) and (self._get_balance_coef() > 0)
        
        # KEY OPTIMIZATION: disable aux tracking when not needed
        # This makes the forward pass faster on non-balance steps
        set_aux_tracking(lime_modules, do_balance)
        
        # Forward pass (task loss)
        if return_outputs:
            task_loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
        else:
            task_loss = super().compute_loss(
                model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch
            )
            outputs = None
        
        # Store task loss (no .item() yet - deferred for performance)
        self._last_task_loss = task_loss.detach()
        
        # Balance loss computation
        if do_balance:
            imp, kl, load, count = collect_aux(lime_modules, task_loss.device)
            
            if count > 0:
                coef = self._get_balance_coef()
                
                # Compute average losses
                imp_avg = imp / count
                kl_avg = kl / count
                load_avg = load / count
                
                # Combined balance loss
                balance_loss = self._importance_coef * imp_avg + self._kl_coef * kl_avg
                weighted_balance = coef * balance_loss
                
                # Store for logging (tensors - deferred .item())
                self._last_importance = imp_avg.detach()
                self._last_kl = kl_avg.detach()
                self._last_load = load_avg.detach()
                self._cached_balance = weighted_balance.detach()
                self._last_balance = self._cached_balance
                
                # Add to total loss
                loss = task_loss + weighted_balance
            else:
                # No LiME modules found or no aux collected
                self._cached_balance = None
                self._last_balance = None
                loss = task_loss
        else:
            # Non-balance step: reuse cached balance for logging consistency
            self._last_balance = self._cached_balance
            # Clear any stale aux (shouldn't be any, but safety)
            clear_aux(lime_modules)
            loss = task_loss
        
        # Update cycle counter
        self._cycle_step = (self._cycle_step + 1) % self._balance_every_n
        
        if return_outputs:
            return loss, outputs
        return loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Training step with optimized logging."""
        if num_items_in_batch is not None:
            loss = super().training_step(model, inputs, num_items_in_batch)
        else:
            loss = super().training_step(model, inputs)
        
        # Logging (CUDA sync point - only here do we call .item())
        step = self.state.global_step
        if step > 0 and step % self.args.logging_steps == 0:
            logs = {
                "task_loss": self._last_task_loss.item() if self._last_task_loss is not None else 0.0,
                "balance_loss": self._last_balance.item() if self._last_balance is not None else 0.0,
                "balance_coef": self._get_balance_coef(),
            }
            
            # Add component losses if available
            if self._last_importance is not None:
                logs["moe/importance"] = self._last_importance.item()
            if self._last_kl is not None:
                logs["moe/kl"] = self._last_kl.item()
            if self._last_load is not None:
                logs["moe/load"] = self._last_load.item()
            
            # Expert usage stats (less frequent to reduce overhead)
            if self._log_expert and step % self._log_freq == 0:
                usage_stats = get_usage_stats(self._get_lime_modules())
                logs.update(usage_stats)
            
            self.log(logs)
        
        return loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Log with task/balance loss included."""
        # Ensure task/balance loss are in logs
        if hasattr(self, '_last_task_loss') and self._last_task_loss is not None and 'task_loss' not in logs:
            logs['task_loss'] = self._last_task_loss.item()
        if hasattr(self, '_last_balance') and self._last_balance is not None and 'balance_loss' not in logs:
            logs['balance_loss'] = self._last_balance.item()
        super().log(logs, start_time)


# ============================================================
# Utilities for Manual Training Loops
# ============================================================

def collect_and_zero_moe_aux(
    model: nn.Module,
    target_device: Optional[torch.device] = None
) -> tuple:
    """
    Collect aux losses from model (compatible API with original trainer).
    
    Args:
        model: Model containing LiME modules
        target_device: Device to place tensors on (defaults to model device)
    
    Returns:
        (totals_dict, count) where totals_dict has 'importance', 'kl_uniform', 'load'
    """
    modules = find_lime_modules(model)
    device = target_device or next(model.parameters()).device
    imp, kl, load, count = collect_aux(modules, device)
    
    return {
        "importance": imp,
        "kl_uniform": kl,
        "load": load,
    }, count


def compute_balance_loss(
    model: nn.Module,
    importance_coef: float = 1.0,
    kl_coef: float = 0.1,
    target_device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute balance loss from all LiME layers.
    
    Use this in manual training loops:
        task_loss = model(**inputs).loss
        balance = compute_balance_loss(model)
        loss = task_loss + 0.01 * balance
        loss.backward()
    """
    totals, count = collect_and_zero_moe_aux(model, target_device)
    if count == 0:
        device = target_device or next(model.parameters()).device
        return torch.tensor(0.0, device=device, requires_grad=True)
    return (importance_coef * totals["importance"] + kl_coef * totals["kl_uniform"]) / count


def add_balance_loss(
    task_loss: torch.Tensor,
    model: nn.Module,
    coef: float = 0.01,
    importance_coef: float = 1.0,
    kl_coef: float = 0.1,
    return_components: bool = False,
) -> Union[torch.Tensor, tuple]:
    """
    Add balance loss to task loss for manual training loops.
    
    Example:
        task_loss = model(**inputs).loss
        total_loss = add_balance_loss(task_loss, model, coef=0.01)
        total_loss.backward()
    
    With components:
        total, task, balance, stats = add_balance_loss(
            task_loss, model, return_components=True
        )
    
    Args:
        task_loss: The task/language modeling loss
        model: Model containing LiME modules
        coef: Overall balance loss coefficient
        importance_coef: Weight for importance loss
        kl_coef: Weight for KL uniform loss
        return_components: If True, return (total, task, balance, stats)
    
    Returns:
        total_loss or (total_loss, task_loss, balance_loss, stats_dict)
    """
    target_device = task_loss.device
    
    if coef <= 0:
        if return_components:
            return task_loss, task_loss, torch.tensor(0.0, device=target_device), {}
        return task_loss
    
    totals, count = collect_and_zero_moe_aux(model, target_device)
    if count == 0:
        if return_components:
            return task_loss, task_loss, torch.tensor(0.0, device=target_device), {}
        return task_loss
    
    importance = totals["importance"] / count
    kl = totals["kl_uniform"] / count
    balance = importance_coef * importance + kl_coef * kl
    weighted_balance = coef * balance
    
    total_loss = task_loss + weighted_balance
    
    if return_components:
        stats = {
            "importance": importance.detach().item() if isinstance(importance, torch.Tensor) else importance,
            "kl": kl.detach().item() if isinstance(kl, torch.Tensor) else kl,
            "load": totals["load"].detach().item() / count if isinstance(totals["load"], torch.Tensor) else 0.0,
        }
        return total_loss, task_loss, weighted_balance, stats
    
    return total_loss


def create_optimizer(
    model: nn.Module,
    moe_lr: float = 1e-3,
    peft_lr: float = 2e-4,
    weight_decay: float = 0.01,
    peft_type: str = "auto",
) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer with dual LRs for any LiME variant.
    
    Use this in manual training loops:
        optimizer = create_optimizer(model, moe_lr=1e-3, peft_lr=2e-4)
        
        for batch in dataloader:
            loss = compute_loss(model, batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    """
    groups, _ = make_param_groups(
        model,
        moe_lr=moe_lr,
        peft_lr=peft_lr,
        weight_decay=weight_decay,
        peft_type=peft_type,
    )
    return torch.optim.AdamW(groups)


# ============================================================
# Summary / Debug Utilities
# ============================================================

def print_lime_summary(model: nn.Module):
    """Print comprehensive LiME model summary."""
    modules = find_lime_modules(model)
    peft_type = detect_peft_type(model)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print("=" * 60)
    print("LiME Model Summary")
    print("=" * 60)
    print(f"PEFT Type:          {peft_type}")
    print(f"LiME Layers:        {len(modules)}")
    print(f"Total Parameters:   {total:,}")
    print(f"Trainable:          {trainable:,} ({100*trainable/total:.2f}%)")
    
    if modules:
        m = modules[0]
        print("-" * 40)
        print("First LiME module config:")
        for attr in ['num_experts', 'top_k', 'rank', 'sparse_mode', 'soft_topk', 
                     'auto_topk', 'temperature', 'jitter_noise', 'n_gram', 'rep_mode']:
            if hasattr(m, attr):
                val = getattr(m, attr)
                # Handle property vs attribute
                if callable(val) and not isinstance(val, bool):
                    continue
                print(f"  {attr}: {val}")
    
    print("=" * 60)


def get_usage_stats(lime_modules: List[nn.Module]) -> Dict:
    """Get expert usage statistics for logging."""
    usages = []
    for m in lime_modules:
        if hasattr(m, '_last_usage') and m._last_usage is not None:
            usages.append(m._last_usage.cpu())
    
    if not usages:
        return {}
    
    avg = torch.stack(usages).mean(dim=0)
    
    stats = {
        "moe/entropy": -(avg * torch.log(avg + 1e-9)).sum().item(),
        "moe/imbalance": avg.max().item() / (avg.min().item() + 1e-9),
        "moe/max_usage": avg.max().item(),
        "moe/min_usage": avg.min().item(),
    }
    
    # Add per-expert usage
    for i, usage in enumerate(avg.tolist()):
        stats[f"moe/expert_{i}"] = usage
    
    return stats


def clear_model_aux(model: nn.Module):
    """Clear all auxiliary data from model."""
    modules = find_lime_modules(model)
    clear_aux(modules)


def set_model_aux_tracking(model: nn.Module, enabled: bool):
    """Enable/disable aux tracking for all LiME modules in model."""
    modules = find_lime_modules(model)
    set_aux_tracking(modules, enabled)


# Aliases for backwards compatibility
print_LiME_summary = print_lime_summary
find_LiME_modules = find_lime_modules