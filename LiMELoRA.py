# LimE for LoRA
# FULLY OPTIMIZED version with AUTO TOP-K support and MULTI-GPU FIX
# + SPARSE MODE SUPPORT
#
# Optimizations:
# 1. Replace einsum with matmul (faster CUDA kernels)
# 2. Fused JIT routing block (single kernel)
# 3. torch.compile support
# 4. Sparse hard top-k (memory efficient)
# 5. Reduced routing prob storage (only stats, not full tensor)
# 6. In-place operations where safe
# 7. Proper gradient flow with dtype casting
# 8. MULTI-GPU: collect_and_zero_moe_aux moves tensors to common device
# 9. SPARSE MODE: MoE-LoRA style sparse gather when enabled

import math
from typing import Union, Callable, Iterable, Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Aux Storage Mixin
# -----------------------------
class _StoresAux:
    _last_aux: dict | None
    _last_usage: torch.Tensor | None
    track_aux: bool
    
    def _store_aux(self, aux): 
        self._last_aux = aux
    
    def _store_usage(self, usage):
        self._last_usage = usage


# -----------------------------
# OPTIMIZED Balance losses
# -----------------------------
# Replace the existing _moe_balance_losses_optimized with this:

def _moe_balance_losses_optimized(probs: torch.Tensor, 
                                   topk_idx: Optional[torch.Tensor] = None,
                                   selection_mask: Optional[torch.Tensor] = None,
                                   mask: Optional[torch.Tensor] = None) -> dict:
    """Full balance loss (not the fast-but-incomplete version)."""
    if probs.dim() == 2:
        probs = probs.unsqueeze(1)
        if topk_idx is not None:
            topk_idx = topk_idx.unsqueeze(1)
        if selection_mask is not None:
            selection_mask = selection_mask.unsqueeze(1)
    
    B, T, E = probs.shape
    device = probs.device
    E_float = float(E)
    
    if mask is not None:
        m = mask.float()
        if m.dim() == 1:
            m = m.unsqueeze(0)
        denom = m.sum().clamp_min(1.0)
        p_bar = (probs * m.unsqueeze(-1)).sum(dim=(0, 1)) / denom
    else:
        p_bar = probs.mean(dim=(0, 1))
    
    # Importance (SCV)
    scv_importance = E_float * (p_bar * p_bar).sum() - 1.0
    
    # Load
    if selection_mask is not None:
        load = selection_mask.float().sum(dim=(0, 1))
    elif topk_idx is not None:
        load = torch.zeros(E, device=device, dtype=probs.dtype)
        load.scatter_add_(0, topk_idx.reshape(-1), 
                          torch.ones(topk_idx.numel(), device=device, dtype=probs.dtype))
    else:
        load = probs.detach().sum(dim=(0, 1))
    
    load_sum = load.sum().clamp_min(1.0)
    load_norm = load / load_sum
    scv_load = (E_float * (load_norm * load_norm).sum() - 1.0).detach()
    
    # KL divergence from uniform
    kl_uniform = (p_bar * (p_bar.clamp_min(1e-8).log() + math.log(E_float))).sum()
    
    return {
        "importance": scv_importance,
        "load": scv_load,
        "kl_uniform": kl_uniform,
        "usage": p_bar.detach().clone(),
    }


# -----------------------------
# FUSED JIT Helpers
# -----------------------------
@torch.jit.script
def _fused_routing_softmax(H_slice: torch.Tensor, delta_slice: torch.Tensor,
                            gamma: float, temperature: float, eps: float = 1e-6) -> torch.Tensor:
    H_scale = H_slice.abs().max().clamp_min(eps)
    delta_scale = delta_slice.abs().max().clamp_min(eps)
    logits = (1.0 - gamma) * (H_slice / H_scale) + gamma * (delta_slice / delta_scale)
    logits = logits * (1.0 / max(temperature, eps))
    return F.softmax(logits, dim=-1)


@torch.jit.script
def _fused_routing_with_jitter(H_slice: torch.Tensor, delta_slice: torch.Tensor,
                                gamma: float, temperature: float, 
                                jitter_low: float, jitter_high: float,
                                eps: float = 1e-6) -> torch.Tensor:
    H_scale = H_slice.abs().max().clamp_min(eps)
    delta_scale = delta_slice.abs().max().clamp_min(eps)
    logits = (1.0 - gamma) * (H_slice / H_scale) + gamma * (delta_slice / delta_scale)
    logits = logits * (1.0 / max(temperature, eps))
    jitter = torch.empty_like(logits).uniform_(jitter_low, jitter_high)
    logits = logits * jitter
    return F.softmax(logits, dim=-1)


# -----------------------------
# DENSE JIT Helpers (soft masking)
# -----------------------------
@torch.jit.script
def _fast_soft_topk_jit(probs: torch.Tensor, k: int, temperature: float = 0.5):
    topk_vals, topk_idx = torch.topk(probs, k=k, dim=-1)
    threshold = topk_vals[..., -1:]
    mask = torch.sigmoid((probs - threshold) * (1.0 / temperature))
    weights = probs * mask
    weight_sum = weights.sum(dim=-1, keepdim=True) + 1e-9
    weights = weights / weight_sum
    return weights, topk_idx


@torch.jit.script
def _fast_auto_topk_jit(probs: torch.Tensor, threshold: float, temperature: float = 0.5):
    max_prob = probs.max(dim=-1, keepdim=True).values
    thresh_val = threshold * max_prob
    mask = torch.sigmoid((probs - thresh_val) * (1.0 / temperature))
    weights = probs * mask
    weight_sum = weights.sum(dim=-1, keepdim=True) + 1e-9
    weights = weights / weight_sum
    selection_mask = probs >= thresh_val
    return weights, selection_mask


# -----------------------------
# SPARSE JIT Helpers (MoE-LoRA style)
# -----------------------------
@torch.jit.script
def _sparse_topk_k1(probs: torch.Tensor, LiMEs: torch.Tensor):
    """Ultra-fast K=1: just argmax + direct gather."""
    idx = torch.argmax(probs, dim=-1)  # (B, T)
    p_mix = LiMEs[idx]  # (B, T, H)
    return p_mix, idx.unsqueeze(-1)


@torch.jit.script
def _sparse_topk(probs: torch.Tensor, LiMEs: torch.Tensor, k: int):
    """Sparse top-k: gather K experts + weighted sum."""
    topk_vals, topk_idx = torch.topk(probs, k=k, dim=-1)  # (B, T, K)
    topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    selected = LiMEs[topk_idx]  # (B, T, K, H)
    p_mix = (topk_vals.unsqueeze(-1) * selected).sum(dim=-2)  # (B, T, H)
    return p_mix, topk_idx


@torch.jit.script
def _sparse_auto_topk(probs: torch.Tensor, LiMEs: torch.Tensor, 
                      threshold: float, max_k: int):
    """
    Sparse auto top-k: select experts above threshold, up to max_k.
    """
    B, T, E = probs.shape
    
    # Get top max_k candidates
    topk_vals, topk_idx = torch.topk(probs, k=max_k, dim=-1)  # (B, T, K)
    
    # Threshold: keep only above threshold * max_prob
    max_prob = probs.max(dim=-1, keepdim=True).values  # (B, T, 1)
    thresh_val = threshold * max_prob
    
    # Mask below-threshold values
    topk_mask = topk_vals >= thresh_val  # (B, T, K)
    topk_vals = topk_vals * topk_mask.float()
    
    # Handle all-zero case: fallback to top-1
    row_sums = topk_vals.sum(dim=-1, keepdim=True)  # (B, T, 1)
    zero_mask = row_sums < 1e-9
    
    # For zero rows, set first value to top prob
    first_val = probs.max(dim=-1, keepdim=True).values  # (B, T, 1)
    topk_vals = torch.where(
        zero_mask.expand_as(topk_vals),
        torch.cat([first_val, torch.zeros(B, T, max_k - 1, device=probs.device, dtype=probs.dtype)], dim=-1),
        topk_vals
    )
    
    # Renormalize
    topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    
    # Gather and mix
    selected = LiMEs[topk_idx]  # (B, T, K, H)
    p_mix = (topk_vals.unsqueeze(-1) * selected).sum(dim=-2)  # (B, T, H)
    
    return p_mix, topk_idx


@torch.jit.script
def _sparse_expert_mix(topk_vals: torch.Tensor, topk_idx: torch.Tensor, 
                       LiMEs: torch.Tensor) -> torch.Tensor:
    """Original sparse mix for hard top-k."""
    expert_vecs = LiMEs[topk_idx]
    p_mix = (topk_vals.unsqueeze(-1) * expert_vecs).sum(dim=-2)
    return p_mix


# =========================
# OPTIMIZED LoRALiMELinear
# =========================
class LoRALiMELinear(nn.Module, _StoresAux):
    """
    Fully optimized LoRA + MoE with ZERO routing parameters.
    
    sparse_mode: If True, uses MoE-LoRA style sparse gather (faster).
                 If False, uses dense matmul (original behavior).
    """
    
    def __init__(
        self,
        linear: nn.Linear,
        num_experts: int = 8,
        rank: int = 8,
        alpha: float = 16.0,
        top_k: int = 2,
        dropout: float = 0.0,
        temperature: float = 1.0,
        use_shared_LiME: bool = True,
        track_aux: bool = True,
        rep_mode: str = "token",
        jitter_noise: float = 0.0,
        n_gram: int = 1,
        LiME_init_std: float = 0.1,
        gamma_routing: float = 0.5,
        soft_topk: bool = True,
        soft_topk_temperature: float = 0.5,
        auto_topk: bool = False,
        auto_topk_threshold: float = 0.5,
        peft_dtype: Optional[torch.dtype] = None,
        moe_dtype: Optional[torch.dtype] = None,
        # NEW: sparse mode
        sparse_mode: bool = False,
    ):
        super().__init__()
        assert isinstance(linear, nn.Linear)
        assert 1 <= top_k <= num_experts
        assert rep_mode in ("token", "last", "mean")
        assert num_experts <= linear.out_features
        assert 0.0 < auto_topk_threshold <= 1.0

        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False

        self.num_experts = int(num_experts)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.top_k = int(top_k)
        self.use_shared_LiME = bool(use_shared_LiME)
        self.track_aux = bool(track_aux)
        self.rep_mode = rep_mode
        self.jitter_noise = float(jitter_noise)
        self.n_gram = int(n_gram)
        self._last_aux = None
        self._last_usage = None
        
        self.soft_topk = bool(soft_topk)
        self.soft_topk_temperature = float(soft_topk_temperature)
        self.auto_topk = bool(auto_topk)
        self.auto_topk_threshold = float(auto_topk_threshold)
        self.sparse_mode = bool(sparse_mode)

        Din, H = linear.in_features, linear.out_features
        r, E = self.rank, self.num_experts
        base_device = linear.weight.device
        
        self.peft_dtype = peft_dtype if peft_dtype is not None else torch.float32
        self.moe_dtype = moe_dtype if moe_dtype is not None else torch.float32

        self._scaling_val = self.alpha / float(r)
        self._temperature_val = float(temperature)
        self._gamma_val = float(gamma_routing)

        with torch.no_grad():
            self.A = nn.Parameter(torch.empty(r, Din, device=base_device, dtype=self.peft_dtype))
            self.B = nn.Parameter(torch.empty(H, r, device=base_device, dtype=self.peft_dtype))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else None

        self.LiMEs = nn.Parameter(
            torch.empty(E, H, device=base_device, dtype=self.moe_dtype).uniform_(
                1.0 - LiME_init_std, 1.0 + LiME_init_std
            )
        )
        
        if use_shared_LiME:
            self.LiME_shared = nn.Parameter(torch.randn(H, device=base_device, dtype=self.moe_dtype) * 0.1)
            self.gamma = nn.Parameter(torch.zeros(1, device=base_device, dtype=self.moe_dtype))
        else:
            self.register_parameter("LiME_shared", None)
            self.gamma = None

    @property
    def gamma_routing(self):
        return self._gamma_val

    def _get_cached_params(self, compute_dtype: torch.dtype):
        A_cast = self.A.to(compute_dtype)
        B_cast = self.B.to(compute_dtype)
        LiMEs_cast = self.LiMEs.to(compute_dtype)
        
        shared_cast = None
        gamma_cast = None
        if self.LiME_shared is not None:
            shared_cast = self.LiME_shared.to(compute_dtype)
            gamma_cast = self.gamma.to(compute_dtype)
        
        return A_cast, B_cast, LiMEs_cast, shared_cast, gamma_cast

    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = getattr(self, "_broadcast_attention_mask", None)

        is_cls = hidden_states.dim() == 2
        if is_cls:
            hidden_states = hidden_states.unsqueeze(1)

        H_out = self.linear(hidden_states)
        B, T, H = H_out.shape
        E = self.num_experts
        compute_dtype = H_out.dtype

        A_cast, B_cast, LiMEs_cast, shared_cast, gamma_cast = self._get_cached_params(compute_dtype)

        X = hidden_states
        if self.lora_dropout is not None and self.training:
            X = self.lora_dropout(X)
        
        delta = X @ A_cast.t()
        delta = delta @ B_cast.t()
        delta = delta * self._scaling_val

        H_slice = H_out[:, :, :E]
        delta_slice = delta[:, :, :E]
        
        if self.training and self.jitter_noise > 0:
            probs = _fused_routing_with_jitter(
                H_slice, delta_slice, 
                self._gamma_val, self._temperature_val,
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
        else:
            probs = _fused_routing_softmax(
                H_slice, delta_slice,
                self._gamma_val, self._temperature_val
            )
        
        # === MIXING LOGIC ===
        topk_idx = None
        selection_mask = None
        
        if self.sparse_mode:
            # ===== SPARSE MODE (MoE-LoRA style) =====
            if self.auto_topk:
                # Sparse auto top-k
                max_k = min(self.top_k * 2, E)  # Allow up to 2x top_k
                p_mix, topk_idx = _sparse_auto_topk(
                    probs, LiMEs_cast, self.auto_topk_threshold, max_k
                )
            elif self.top_k == 1:
                # Ultra-fast K=1 path
                p_mix, topk_idx = _sparse_topk_k1(probs, LiMEs_cast)
            else:
                # Sparse K>1 path
                p_mix, topk_idx = _sparse_topk(probs, LiMEs_cast, self.top_k)
        else:
            # ===== DENSE MODE (original) =====
            if self.auto_topk:
                weights, selection_mask = _fast_auto_topk_jit(
                    probs, self.auto_topk_threshold, self.soft_topk_temperature
                )
                p_mix = weights @ LiMEs_cast
            elif self.soft_topk:
                weights, topk_idx = _fast_soft_topk_jit(
                    probs, self.top_k, self.soft_topk_temperature
                )
                p_mix = weights @ LiMEs_cast
            else:
                # Hard top-k (already sparse in original)
                topk_vals, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)
                topk_sum = topk_vals.sum(dim=-1, keepdim=True).clamp_min_(1e-9)
                topk_vals = topk_vals / topk_sum
                p_mix = _sparse_expert_mix(topk_vals, topk_idx, LiMEs_cast)

        if self.track_aux:
            if self.training:
                aux = _moe_balance_losses_optimized(
                    probs, topk_idx, selection_mask, attention_mask
                )
                self._store_aux(aux)
            else:
                self._store_usage(probs.mean(dim=(0, 1)).detach())

        routed = delta * p_mix
        
        if self.use_shared_LiME and shared_cast is not None:
            routed = routed + (gamma_cast * (delta * shared_cast))
        
        out = H_out + routed

        if attention_mask is not None:
            msk = attention_mask.bool()
            if msk.dim() == 2:
                msk = msk.unsqueeze(-1)
            out = torch.where(msk, out, H_out)

        return out.squeeze(1) if is_cls else out

    def __repr__(self):
        mode = "auto" if self.auto_topk else ("soft" if self.soft_topk else "hard")
        sparse_str = ", sparse" if self.sparse_mode else ""
        return (f"LoRALiMELinear(in={self.linear.in_features}, "
                f"out={self.linear.out_features}, experts={self.num_experts}, "
                f"topk_mode={mode}{sparse_str})")


# =========================
# OPTIMIZED LoRALiMEEmbedding  
# =========================
class LoRALiMEEmbedding(nn.Module, _StoresAux):
    """
    Fully optimized embedding version with sparse mode support.
    """
    
    def __init__(
        self,
        embedding: nn.Embedding,
        num_experts: int = 8,
        rank: int = 8,
        alpha: float = 16.0,
        top_k: int = 2,
        temperature: float = 1.0,
        use_shared_LiME: bool = True,
        pad_token_id=None,
        track_aux: bool = True,
        rep_mode: str = "token",
        jitter_noise: float = 0.0,
        n_gram: int = 1,
        LiME_init_std: float = 0.1,
        gamma_routing: float = 0.5,
        soft_topk: bool = True,
        soft_topk_temperature: float = 0.5,
        auto_topk: bool = False,
        auto_topk_threshold: float = 0.5,
        peft_dtype: Optional[torch.dtype] = None,
        moe_dtype: Optional[torch.dtype] = None,
        # NEW: sparse mode
        sparse_mode: bool = False,
    ):
        super().__init__()
        assert isinstance(embedding, nn.Embedding)
        assert 1 <= top_k <= num_experts
        assert num_experts <= embedding.embedding_dim
        assert 0.0 < auto_topk_threshold <= 1.0

        self.embedding = embedding
        for p in self.embedding.parameters():
            p.requires_grad = False

        self.num_experts = int(num_experts)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.top_k = int(top_k)
        self.use_shared_LiME = bool(use_shared_LiME)
        self.pad_token_id = pad_token_id
        self.track_aux = bool(track_aux)
        self.rep_mode = rep_mode
        self.jitter_noise = float(jitter_noise)
        self.n_gram = int(n_gram)
        self._last_aux = None
        self._last_usage = None
        
        self.soft_topk = bool(soft_topk)
        self.soft_topk_temperature = float(soft_topk_temperature)
        self.auto_topk = bool(auto_topk)
        self.auto_topk_threshold = float(auto_topk_threshold)
        self.sparse_mode = bool(sparse_mode)

        H = embedding.embedding_dim
        r, E = self.rank, self.num_experts
        base_device = embedding.weight.device
        
        self.peft_dtype = peft_dtype if peft_dtype is not None else torch.float32
        self.moe_dtype = moe_dtype if moe_dtype is not None else torch.float32

        self._scaling_val = self.alpha / float(r)
        self._temperature_val = float(temperature)
        self._gamma_val = float(gamma_routing)

        with torch.no_grad():
            self.A = nn.Parameter(torch.empty(r, H, device=base_device, dtype=self.peft_dtype))
            self.B = nn.Parameter(torch.empty(H, r, device=base_device, dtype=self.peft_dtype))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)

        self.LiMEs = nn.Parameter(
            torch.empty(E, H, device=base_device, dtype=self.moe_dtype).uniform_(
                1.0 - LiME_init_std, 1.0 + LiME_init_std
            )
        )
        
        if use_shared_LiME:
            self.LiME_shared = nn.Parameter(torch.randn(H, device=base_device, dtype=self.moe_dtype) * 0.1)
            self.gamma = nn.Parameter(torch.zeros(1, device=base_device, dtype=self.moe_dtype))
        else:
            self.register_parameter("LiME_shared", None)
            self.gamma = None

    @property
    def gamma_routing(self):
        return self._gamma_val

    def _get_cached_params(self, compute_dtype: torch.dtype):
        A_cast = self.A.to(compute_dtype)
        B_cast = self.B.to(compute_dtype)
        LiMEs_cast = self.LiMEs.to(compute_dtype)
        
        shared_cast = None
        gamma_cast = None
        if self.LiME_shared is not None:
            shared_cast = self.LiME_shared.to(compute_dtype)
            gamma_cast = self.gamma.to(compute_dtype)
        
        return A_cast, B_cast, LiMEs_cast, shared_cast, gamma_cast

    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = getattr(self, "_broadcast_attention_mask", None)

        E_out = self.embedding(input_ids)
        B, T, H = E_out.shape
        E = self.num_experts
        compute_dtype = E_out.dtype

        if attention_mask is None and self.pad_token_id is not None:
            attention_mask = (input_ids != self.pad_token_id)

        A_cast, B_cast, LiMEs_cast, shared_cast, gamma_cast = self._get_cached_params(compute_dtype)

        delta = E_out @ A_cast.t()
        delta = delta @ B_cast.t()
        delta = delta * self._scaling_val

        E_slice = E_out[:, :, :E]
        delta_slice = delta[:, :, :E]
        
        if self.training and self.jitter_noise > 0:
            probs = _fused_routing_with_jitter(
                E_slice, delta_slice,
                self._gamma_val, self._temperature_val,
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
        else:
            probs = _fused_routing_softmax(
                E_slice, delta_slice,
                self._gamma_val, self._temperature_val
            )
        
        # === MIXING LOGIC ===
        topk_idx = None
        selection_mask = None
        
        if self.sparse_mode:
            # ===== SPARSE MODE (MoE-LoRA style) =====
            if self.auto_topk:
                max_k = min(self.top_k * 2, E)
                p_mix, topk_idx = _sparse_auto_topk(
                    probs, LiMEs_cast, self.auto_topk_threshold, max_k
                )
            elif self.top_k == 1:
                p_mix, topk_idx = _sparse_topk_k1(probs, LiMEs_cast)
            else:
                p_mix, topk_idx = _sparse_topk(probs, LiMEs_cast, self.top_k)
        else:
            # ===== DENSE MODE (original) =====
            if self.auto_topk:
                weights, selection_mask = _fast_auto_topk_jit(
                    probs, self.auto_topk_threshold, self.soft_topk_temperature
                )
                p_mix = weights @ LiMEs_cast
            elif self.soft_topk:
                weights, topk_idx = _fast_soft_topk_jit(
                    probs, self.top_k, self.soft_topk_temperature
                )
                p_mix = weights @ LiMEs_cast
            else:
                topk_vals, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)
                topk_sum = topk_vals.sum(dim=-1, keepdim=True).clamp_min_(1e-9)
                topk_vals = topk_vals / topk_sum
                p_mix = _sparse_expert_mix(topk_vals, topk_idx, LiMEs_cast)

        if self.track_aux:
            if self.training:
                aux = _moe_balance_losses_optimized(
                    probs, topk_idx, selection_mask, attention_mask
                )
                self._store_aux(aux)
            else:
                self._store_usage(probs.mean(dim=(0, 1)).detach())

        out_delta = delta * p_mix
        
        if self.use_shared_LiME and shared_cast is not None:
            out_delta = out_delta + (gamma_cast * (delta * shared_cast))
        
        out = E_out + out_delta

        if attention_mask is not None:
            msk = attention_mask.bool()
            if msk.dim() == 2:
                msk = msk.unsqueeze(-1)
            out = torch.where(msk, out, E_out)

        return out

    def __repr__(self):
        mode = "auto" if self.auto_topk else ("soft" if self.soft_topk else "hard")
        sparse_str = ", sparse" if self.sparse_mode else ""
        return (f"LoRALiMEEmbedding(dim={self.embedding.embedding_dim}, "
                f"experts={self.num_experts}, topk_mode={mode}{sparse_str})")


# ---------- Apply PEFT ----------
def _is_tied_with_linear(emb_mod, parent, child_name):
    try:
        w = emb_mod.weight
        for n, m in parent.named_children():
            if n != child_name and isinstance(m, nn.Linear) and getattr(m, "weight", None) is w:
                return True
    except:
        pass
    return False


def _replace_layers(
    parent, module_path="", targets=None,
    num_experts=8, rank=8, alpha=16.0, top_k=2, dropout=0.0,
    temperature=1.0, use_shared_LiME=True, preserve_tying=True,
    pad_token_id=None, track_aux=True, rep_mode="token",
    jitter_noise=0.0, n_gram=1, LiME_init_std=0.1,
    gamma_routing=0.5, soft_topk=True, soft_topk_temperature=0.5,
    auto_topk=False, auto_topk_threshold=0.5,
    peft_dtype=None, moe_dtype=None,
    sparse_mode=False,
):
    if targets is None:
        def wants(path, mod): return isinstance(mod, nn.Linear)
    elif callable(targets):
        wants = targets
    else:
        target_list = list(targets)
        def wants(path, mod): return isinstance(mod, nn.Linear) and any(path.endswith(t) for t in target_list)

    for child_name, child in list(parent.named_children()):
        path = f"{module_path}.{child_name}" if module_path else child_name

        if isinstance(child, (LoRALiMELinear, LoRALiMEEmbedding)):
            continue

        if isinstance(child, nn.Linear) and wants(path, child):
            wrapped = LoRALiMELinear(
                child, num_experts=num_experts, rank=rank, alpha=alpha, top_k=top_k,
                dropout=dropout, temperature=temperature, use_shared_LiME=use_shared_LiME,
                track_aux=track_aux, rep_mode=rep_mode, jitter_noise=jitter_noise,
                n_gram=n_gram, LiME_init_std=LiME_init_std, gamma_routing=gamma_routing,
                soft_topk=soft_topk, soft_topk_temperature=soft_topk_temperature,
                auto_topk=auto_topk, auto_topk_threshold=auto_topk_threshold,
                peft_dtype=peft_dtype, moe_dtype=moe_dtype,
                sparse_mode=sparse_mode,
            )
            setattr(parent, child_name, wrapped)

        elif isinstance(child, nn.Embedding) and wants(path, child):
            if preserve_tying and _is_tied_with_linear(child, parent, child_name):
                pass
            else:
                wrapped = LoRALiMEEmbedding(
                    child, num_experts=num_experts, rank=rank, alpha=alpha, top_k=top_k,
                    temperature=temperature, use_shared_LiME=use_shared_LiME,
                    pad_token_id=pad_token_id, track_aux=track_aux, rep_mode=rep_mode,
                    jitter_noise=jitter_noise, n_gram=n_gram, LiME_init_std=LiME_init_std,
                    gamma_routing=gamma_routing, soft_topk=soft_topk,
                    soft_topk_temperature=soft_topk_temperature,
                    auto_topk=auto_topk, auto_topk_threshold=auto_topk_threshold,
                    peft_dtype=peft_dtype, moe_dtype=moe_dtype,
                    sparse_mode=sparse_mode,
                )
                setattr(parent, child_name, wrapped)
        else:
            _replace_layers(
                child, path, targets, num_experts, rank, alpha, top_k, dropout,
                temperature, use_shared_LiME, preserve_tying, pad_token_id,
                track_aux, rep_mode, jitter_noise, n_gram, LiME_init_std,
                gamma_routing, soft_topk, soft_topk_temperature,
                auto_topk, auto_topk_threshold,
                peft_dtype, moe_dtype,
                sparse_mode,
            )


def apply_peft(
    model: nn.Module,
    targets=None,
    num_experts: int = 8,
    rank: int = 8,
    alpha: float = 16.0,
    top_k: int = 2,
    dropout: float = 0.0,
    temperature: float = 1.0,
    use_shared_LiME: bool = False,
    preserve_tying: bool = True,
    pad_token_id=None,
    tokenizer=None,
    track_aux: bool = True,
    rep_mode: str = "token",
    jitter_noise: float = 0.0,
    n_gram: int = 1,
    LiME_init_std: float = 0.1,
    gamma_routing: float = 0.5,
    soft_topk: bool = True,
    soft_topk_temperature: float = 0.5,
    auto_topk: bool = False,
    auto_topk_threshold: float = 0.5,
    peft_dtype: Optional[torch.dtype] = None,
    moe_dtype: Optional[torch.dtype] = None,
    use_compile: bool = False,
    sparse_mode: bool = False,  # NEW
) -> nn.Module:
    """Apply LoRA-LiME PEFT to model."""
    if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is not None:
        pad_token_id = tokenizer.pad_token_id

    for p in model.parameters():
        p.requires_grad = False

    _replace_layers(
        model, "", targets, num_experts, rank, alpha, top_k, dropout,
        temperature, use_shared_LiME, preserve_tying, pad_token_id,
        track_aux, rep_mode, jitter_noise, n_gram, LiME_init_std,
        gamma_routing, soft_topk, soft_topk_temperature,
        auto_topk, auto_topk_threshold,
        peft_dtype, moe_dtype,
        sparse_mode,
    )
    
    if use_compile and hasattr(torch, 'compile'):
        for m in model.modules():
            if isinstance(m, (LoRALiMELinear, LoRALiMEEmbedding)):
                m.forward = torch.compile(m.forward, mode="reduce-overhead")
    
    return model


# ---------- Mask broadcaster ----------
def enable_mask_broadcast(model: nn.Module):
    if getattr(model, "_peft_broadcast_wrapped", False):
        return
    orig_forward = model.forward
    def wrapped_forward(*args, **kwargs):
        mask = kwargs.get("attention_mask", None)
        for m in model.modules():
            if isinstance(m, (LoRALiMELinear, LoRALiMEEmbedding)):
                if mask is not None:
                    m._broadcast_attention_mask = mask
        try:
            return orig_forward(*args, **kwargs)
        finally:
            for m in model.modules():
                if hasattr(m, "_broadcast_attention_mask"):
                    delattr(m, "_broadcast_attention_mask")
    model.forward = wrapped_forward
    model._peft_broadcast_wrapped = True


# ---------- Collect aux losses (MULTI-GPU FIXED) ----------
def collect_and_zero_moe_aux(model: nn.Module, target_device: Optional[torch.device] = None) -> tuple:
    """
    Collect and sum aux losses from all MoE layers.
    MULTI-GPU FIX: Moves all tensors to target_device before summing.
    """
    totals = {"importance": 0.0, "load": 0.0, "kl_uniform": 0.0}
    count = 0
    device = target_device
    
    for m in model.modules():
        if hasattr(m, "_last_aux") and m._last_aux is not None:
            aux = m._last_aux
            
            if device is None and isinstance(aux["importance"], torch.Tensor):
                device = aux["importance"].device
            
            imp = aux["importance"]
            load = aux["load"]
            kl = aux["kl_uniform"]
            
            if device is not None:
                if isinstance(imp, torch.Tensor) and imp.device != device:
                    imp = imp.to(device)
                if isinstance(load, torch.Tensor) and load.device != device:
                    load = load.to(device)
                if isinstance(kl, torch.Tensor) and kl.device != device:
                    kl = kl.to(device)
            
            totals["importance"] = totals["importance"] + imp
            totals["load"] = totals["load"] + load
            totals["kl_uniform"] = totals["kl_uniform"] + kl
            m._last_aux = None
            count += 1
    
    if count == 0:
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        for k in totals:
            totals[k] = torch.tensor(0.0, device=device, requires_grad=True)
        return totals, 0
    
    return totals, count


def compute_balance_loss(model: nn.Module, importance_coef: float = 1.0, 
                         kl_coef: float = 0.1,
                         target_device: Optional[torch.device] = None) -> torch.Tensor:
    """Compute balance loss from all MoE layers."""
    totals, count = collect_and_zero_moe_aux(model, target_device=target_device)
    if count == 0:
        device = target_device if target_device else next(model.parameters()).device
        return torch.tensor(0.0, device=device, requires_grad=True)
    return (importance_coef * totals["importance"] + kl_coef * totals["kl_uniform"]) / count


def get_expert_usage_stats(model: nn.Module) -> Dict:
    """Get usage stats from lightweight cached values."""
    usages = []
    for m in model.modules():
        if hasattr(m, "_last_usage") and m._last_usage is not None:
            usages.append(m._last_usage.cpu())
    
    if not usages:
        for m in model.modules():
            if hasattr(m, "_last_aux") and m._last_aux is not None:
                aux = m._last_aux
                if "usage" in aux:
                    usages.append(aux["usage"].cpu())
    
    if not usages:
        return {"usage_per_expert": [], "entropy": 0.0, "imbalance_ratio": 1.0}
    
    avg_usage = torch.stack(usages).mean(dim=0)
    entropy = -(avg_usage * torch.log(avg_usage + 1e-9)).sum().item()
    
    return {
        "usage_per_expert": avg_usage.tolist(),
        "entropy": entropy,
        "max_usage": avg_usage.max().item(),
        "min_usage": avg_usage.min().item(),
        "imbalance_ratio": avg_usage.max().item() / (avg_usage.min().item() + 1e-9),
    }


def clear_aux(model: nn.Module):
    """Clear auxiliary data from all MoE layers."""
    for m in model.modules():
        if hasattr(m, "_last_aux"):
            m._last_aux = None
        if hasattr(m, "_last_usage"):
            m._last_usage = None


# ---------- Utilities ----------
def set_gamma_routing(model: nn.Module, gamma: float):
    for m in model.modules():
        if isinstance(m, (LoRALiMELinear, LoRALiMEEmbedding)):
            m._gamma_val = gamma


def set_temperature(model: nn.Module, temp: float):
    for m in model.modules():
        if isinstance(m, (LoRALiMELinear, LoRALiMEEmbedding)):
            m._temperature_val = temp


def set_soft_topk(model: nn.Module, soft_topk: bool, temperature: float = None):
    for m in model.modules():
        if isinstance(m, (LoRALiMELinear, LoRALiMEEmbedding)):
            m.soft_topk = soft_topk
            if temperature is not None:
                m.soft_topk_temperature = temperature


def set_auto_topk(model: nn.Module, auto_topk: bool, threshold: float = None, 
                  temperature: float = None):
    for m in model.modules():
        if isinstance(m, (LoRALiMELinear, LoRALiMEEmbedding)):
            m.auto_topk = auto_topk
            if threshold is not None:
                m.auto_topk_threshold = threshold
            if temperature is not None:
                m.soft_topk_temperature = temperature


def set_sparse_mode(model: nn.Module, sparse: bool):
    """Set sparse mode for all LiME layers."""
    for m in model.modules():
        if isinstance(m, (LoRALiMELinear, LoRALiMEEmbedding)):
            m.sparse_mode = sparse


def set_jitter(model: nn.Module, jitter: float):
    for m in model.modules():
        if isinstance(m, (LoRALiMELinear, LoRALiMEEmbedding)):
            m.jitter_noise = jitter


def set_peft_dtype(model: nn.Module, dtype: torch.dtype):
    for m in model.modules():
        if isinstance(m, (LoRALiMELinear, LoRALiMEEmbedding)):
            m.A.data = m.A.data.to(dtype)
            m.B.data = m.B.data.to(dtype)
            m.peft_dtype = dtype


def set_moe_dtype(model: nn.Module, dtype: torch.dtype):
    for m in model.modules():
        if isinstance(m, (LoRALiMELinear, LoRALiMEEmbedding)):
            m.LiMEs.data = m.LiMEs.data.to(dtype)
            if m.LiME_shared is not None:
                m.LiME_shared.data = m.LiME_shared.data.to(dtype)
            if m.gamma is not None:
                m.gamma.data = m.gamma.data.to(dtype)
            m.moe_dtype = dtype


def get_auto_topk_stats(model: nn.Module) -> Dict:
    stats = []
    for name, m in model.named_modules():
        if isinstance(m, (LoRALiMELinear, LoRALiMEEmbedding)) and m.auto_topk:
            if hasattr(m, "_last_usage") and m._last_usage is not None:
                usage = m._last_usage
                num_active = (usage > 1e-6).sum().item()
                stats.append({"layer": name, "active_experts": num_active})
    
    if not stats:
        return {"layers": [], "overall_avg": 0.0}
    
    overall_avg = sum(s.get("active_experts", 0) for s in stats) / len(stats)
    return {"layers": stats, "overall_avg": overall_avg}


def print_routing_summary(model: nn.Module):
    total_lora = 0
    total_LiMEs = 0
    peft_dtypes = set()
    moe_dtypes = set()
    topk_modes = set()
    sparse_modes = set()
    num_layers = 0
    
    for name, m in model.named_modules():
        if isinstance(m, LoRALiMELinear):
            total_lora += m.A.numel() + m.B.numel()
            total_LiMEs += m.LiMEs.numel()
            if m.LiME_shared is not None:
                total_LiMEs += m.LiME_shared.numel() + m.gamma.numel()
            peft_dtypes.add(m.A.dtype)
            moe_dtypes.add(m.LiMEs.dtype)
            if m.auto_topk:
                topk_modes.add(f"auto(θ={m.auto_topk_threshold})")
            elif m.soft_topk:
                topk_modes.add(f"soft(k={m.top_k})")
            else:
                topk_modes.add(f"hard(k={m.top_k})")
            sparse_modes.add("sparse" if m.sparse_mode else "dense")
            num_layers += 1
    
    print("=" * 60)
    print("LoRA-LiME (SPARSE MODE SUPPORT)")
    print("=" * 60)
    print(f"Wrapped layers:     {num_layers}")
    print(f"Routing parameters: 0")
    print(f"LoRA parameters:    {total_lora:,}")
    print(f"LiMEs parameters:   {total_LiMEs:,}")
    print(f"Total trainable:    {total_lora + total_LiMEs:,}")
    print(f"PEFT dtype(s):      {peft_dtypes}")
    print(f"MoE dtype(s):       {moe_dtypes}")
    print(f"Top-K mode(s):      {topk_modes}")
    print(f"Mixing mode(s):     {sparse_modes}")
    print("=" * 60)


def compile_moe_layers(model: nn.Module, mode: str = "reduce-overhead"):
    if not hasattr(torch, 'compile'):
        print("Warning: torch.compile not available (requires PyTorch 2.0+)")
        return
    
    count = 0
    for m in model.modules():
        if isinstance(m, (LoRALiMELinear, LoRALiMEEmbedding)):
            m.forward = torch.compile(m.forward, mode=mode)
            count += 1
    
    print(f"Compiled {count} MoE layers with mode='{mode}'")