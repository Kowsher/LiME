# slicefine_lime_row.py
# SliceFine + LiME: Optimized 2-Part ROW-mode implementation
# Full API compatibility with LoRA-LiME (n_gram, rep_mode, etc.)
# + SPARSE MODE SUPPORT
# + N-GRAM ROUTING SUPPORT (anchor + expand, consistent with other LiME variants)
#
# SliceFine ROW-MODE: W = [part_T]  (rank, Din)   <- trainable (top rows)
#                         [part_F]  (H-rank, Din) <- frozen (bottom rows)
#
# Forward: out_T = F.linear(x, part_T)  # (B, T, rank) - TRAINABLE contribution
#          out_F = F.linear(x, part_F)  # (B, T, H-rank) - FROZEN contribution
#          # MoE applied only to out_T, then cat at end
#          out = cat([out_T * p_mix[:,:,:rank], out_F], dim=-1)
#
# ADVANTAGES:
# 1. Input X stays CONTIGUOUS - no slicing along last dim!
# 2. No F.pad() allocation - MoE applied directly to out_T
# 3. Single cat() at end instead of cat() + pad()
# 4. Simpler H-only routing (no delta_slice needed)

import math
from typing import Optional, Dict, Callable, Iterable, Union

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
# Balance Loss
# -----------------------------
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
# JIT Helpers - Routing (ROW-MODE: H-only routing)
# -----------------------------
@torch.jit.script
def _routing_softmax(
    H_slice: torch.Tensor,
    temperature: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Simple routing using only hidden state (no delta).
    
    ROW-MODE uses this simpler routing because:
    - delta is zero-padded beyond rank, making delta_slice unreliable for E > rank
    - Hidden state alone provides sufficient routing signal
    - This is similar to standard MoE routers
    """
    H_scale = H_slice.abs().max().clamp_min(eps)
    logits = H_slice / H_scale
    logits = logits * (1.0 / max(temperature, eps))
    return F.softmax(logits, dim=-1)


@torch.jit.script
def _routing_softmax_with_jitter(
    H_slice: torch.Tensor,
    temperature: float,
    jitter_low: float,
    jitter_high: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Simple routing with multiplicative jitter noise (no delta)."""
    H_scale = H_slice.abs().max().clamp_min(eps)
    logits = H_slice / H_scale
    logits = logits * (1.0 / max(temperature, eps))
    jitter = torch.empty_like(logits).uniform_(jitter_low, jitter_high)
    logits = logits * jitter
    return F.softmax(logits, dim=-1)


# -----------------------------
# N-GRAM ROUTING HELPER (consistent with other LiME variants)
# -----------------------------
def _get_ngram_anchor_indices(T: int, n_gram: int, device: torch.device) -> torch.Tensor:
    """
    Get indices of anchor tokens (last token of each n-gram group).
    
    For n_gram=2 with T=6: returns [1, 3, 5] (0-indexed)
    For n_gram=2 with T=7: returns [1, 3, 5, 6] (last group has 1 token, uses itself)
    """
    num_full_groups = T // n_gram
    remainder = T % n_gram
    
    # Full group anchors
    anchors = torch.arange(n_gram - 1, num_full_groups * n_gram, n_gram, device=device)
    
    # If there's a remainder, the last partial group uses its last token
    if remainder > 0:
        last_anchor = torch.tensor([T - 1], device=device)
        anchors = torch.cat([anchors, last_anchor])
    
    return anchors


def _expand_ngram_probs(anchor_probs: torch.Tensor, T: int, n_gram: int) -> torch.Tensor:
    """
    Expand anchor probabilities back to full sequence length.
    
    Args:
        anchor_probs: (B, num_anchors, E) - routing probs for anchor tokens only
        T: original sequence length
        n_gram: tokens per group
    
    Returns:
        (B, T, E) with each anchor's probs repeated for its group
    """
    B, num_anchors, E = anchor_probs.shape
    num_full_groups = T // n_gram
    remainder = T % n_gram
    
    if remainder == 0:
        return anchor_probs.repeat_interleave(n_gram, dim=1)
    else:
        full_group_probs = anchor_probs[:, :num_full_groups, :]
        expanded_full = full_group_probs.repeat_interleave(n_gram, dim=1)
        
        last_anchor_probs = anchor_probs[:, -1:, :]
        expanded_last = last_anchor_probs.expand(B, remainder, E)
        
        return torch.cat([expanded_full, expanded_last], dim=1)


def _expand_ngram_values(anchor_values: torch.Tensor, T: int, n_gram: int) -> torch.Tensor:
    """
    Expand anchor values (e.g., p_mix) back to full sequence length.
    
    Args:
        anchor_values: (B, num_anchors, D) - values computed for anchor tokens only
        T: original sequence length
        n_gram: tokens per group
    
    Returns:
        (B, T, D) with each anchor's values repeated for its group
    """
    B, num_anchors, D = anchor_values.shape
    num_full_groups = T // n_gram
    remainder = T % n_gram
    
    if remainder == 0:
        return anchor_values.repeat_interleave(n_gram, dim=1)
    else:
        full_group_values = anchor_values[:, :num_full_groups, :]
        expanded_full = full_group_values.repeat_interleave(n_gram, dim=1)
        
        last_anchor_values = anchor_values[:, -1:, :]
        expanded_last = last_anchor_values.expand(B, remainder, D)
        
        return torch.cat([expanded_full, expanded_last], dim=1)


# -----------------------------
# DENSE JIT Helpers (soft masking)
# -----------------------------
@torch.jit.script
def _soft_topk(probs: torch.Tensor, k: int, temperature: float = 0.5):
    """Soft top-k selection with sigmoid masking."""
    topk_vals, topk_idx = torch.topk(probs, k=k, dim=-1)
    threshold = topk_vals[..., -1:]
    mask = torch.sigmoid((probs - threshold) * (1.0 / temperature))
    weights = probs * mask
    weight_sum = weights.sum(dim=-1, keepdim=True) + 1e-9
    weights = weights / weight_sum
    return weights, topk_idx


@torch.jit.script
def _auto_topk(probs: torch.Tensor, threshold: float, temperature: float = 0.5):
    """Automatic top-k based on threshold relative to max."""
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
    idx = torch.argmax(probs, dim=-1)
    p_mix = LiMEs[idx]
    return p_mix, idx.unsqueeze(-1)


@torch.jit.script
def _sparse_topk(probs: torch.Tensor, LiMEs: torch.Tensor, k: int):
    """Sparse top-k: gather K experts + weighted sum."""
    topk_vals, topk_idx = torch.topk(probs, k=k, dim=-1)
    topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    selected = LiMEs[topk_idx]
    p_mix = (topk_vals.unsqueeze(-1) * selected).sum(dim=-2)
    return p_mix, topk_idx


@torch.jit.script
def _sparse_auto_topk(probs: torch.Tensor, LiMEs: torch.Tensor, 
                      threshold: float, max_k: int):
    """Sparse auto top-k: select experts above threshold, up to max_k."""
    B, T, E = probs.shape
    
    topk_vals, topk_idx = torch.topk(probs, k=max_k, dim=-1)
    
    max_prob = probs.max(dim=-1, keepdim=True).values
    thresh_val = threshold * max_prob
    
    topk_mask = topk_vals >= thresh_val
    topk_vals = topk_vals * topk_mask.float()
    
    row_sums = topk_vals.sum(dim=-1, keepdim=True)
    zero_mask = row_sums < 1e-9
    
    first_val = probs.max(dim=-1, keepdim=True).values
    topk_vals = torch.where(
        zero_mask.expand_as(topk_vals),
        torch.cat([first_val, torch.zeros(B, T, max_k - 1, device=probs.device, dtype=probs.dtype)], dim=-1),
        topk_vals
    )
    
    topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    
    selected = LiMEs[topk_idx]
    p_mix = (topk_vals.unsqueeze(-1) * selected).sum(dim=-2)
    
    return p_mix, topk_idx


@torch.jit.script
def _sparse_expert_mix(
    topk_vals: torch.Tensor, topk_idx: torch.Tensor, LiMEs: torch.Tensor
) -> torch.Tensor:
    """Compute expert mixture for hard top-k."""
    expert_vecs = LiMEs[topk_idx]
    p_mix = (topk_vals.unsqueeze(-1) * expert_vecs).sum(dim=-2)
    return p_mix


# =============================================================================
# SliceFineLiMELinear - ROW MODE
# =============================================================================
class SliceFineLiMELinear(nn.Module, _StoresAux):
    """
    SliceFine + LiME Linear Layer (2-Part ROW mode).

    Partitions weight matrix by ROWS: W = [part_T]  (rank, Din)   <- trainable
                                          [part_F]  (H-rank, Din) <- frozen

    ADVANTAGE over column-mode: Input X stays CONTIGUOUS!
    - Column-mode slices X[..., :rank] which is non-contiguous
    - Row-mode does F.linear(X, part_T) where X is untouched

    ROUTING: Uses only base_out[:, :, :E] for routing (no delta_slice).
    This is simpler and avoids issues with zero-padded delta in row-mode.

    sparse_mode: If True, uses MoE-LoRA style sparse gather (faster).
                 If False, uses dense matmul (original behavior).
    
    n_gram: Number of tokens per routing group.
            n_gram=1: every token gets its own routing decision (default)
            n_gram=2: tokens 1,2 share routing from token 2; tokens 3,4 share from token 4; etc.
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
        soft_topk: bool = True,
        soft_topk_temperature: float = 0.5,
        auto_topk: bool = False,
        auto_topk_threshold: float = 0.5,
        peft_dtype: Optional[torch.dtype] = None,
        moe_dtype: Optional[torch.dtype] = None,
        sparse_mode: bool = False,
    ):
        super().__init__()
        assert isinstance(linear, nn.Linear)
        assert 1 <= top_k <= num_experts
        assert num_experts <= linear.out_features
        assert 0.0 < auto_topk_threshold <= 1.0
        assert rank > 0
        assert rep_mode in ("token", "last", "mean")
        assert n_gram >= 1, "n_gram must be >= 1"

        Din, H = linear.in_features, linear.out_features
        # ROW MODE: rank must be <= H (output features)
        assert rank <= H, f"rank ({rank}) must be <= out_features ({H})"

        self.in_features = Din
        self.out_features = H
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

        self._temperature_val = float(temperature)

        self.peft_dtype = peft_dtype if peft_dtype is not None else torch.float32
        self.moe_dtype = moe_dtype if moe_dtype is not None else torch.float32
        base_device = linear.weight.device

        # === Partition weights by ROWS (2-part) ===
        with torch.no_grad():
            W = linear.weight.detach()  # (H, Din)

            # ROW MODE: part_T is (rank, Din), part_F is (H-rank, Din)
            self.part_T = nn.Parameter(
                W[:rank, :].clone().contiguous().to(self.peft_dtype),
                requires_grad=True,
            )
            self.part_F = nn.Parameter(
                W[rank:, :].clone().contiguous().to(self.peft_dtype),
                requires_grad=False,
            )

        # Bias (trainable)
        if linear.bias is not None:
            self.bias = nn.Parameter(
                linear.bias.detach().to(self.peft_dtype).clone(),
                requires_grad=True,
            )
        else:
            self.register_parameter("bias", None)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # === LiME Expert Vectors ===
        self.LiMEs = nn.Parameter(
            torch.empty(num_experts, H, device=base_device, dtype=self.moe_dtype).uniform_(
                1.0 - LiME_init_std, 1.0 + LiME_init_std
            )
        )

        if use_shared_LiME:
            self.LiME_shared = nn.Parameter(
                torch.zeros(H, device=base_device, dtype=self.moe_dtype)
            )
            self.gamma = nn.Parameter(
                torch.zeros(1, device=base_device, dtype=self.moe_dtype)
            )
        else:
            self.register_parameter("LiME_shared", None)
            self.register_parameter("gamma", None)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstruct full weight matrix."""
        return torch.cat([self.part_T, self.part_F], dim=0)  # ROW MODE: cat along dim=0

    @property
    def temperature(self) -> float:
        return self._temperature_val

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = getattr(self, "_broadcast_attention_mask", None)

        is_2d = hidden_states.dim() == 2
        if is_2d:
            hidden_states = hidden_states.unsqueeze(1)

        B, T, _ = hidden_states.shape
        E = self.num_experts
        compute_dtype = hidden_states.dtype

        X = hidden_states
        if self.dropout is not None and self.training:
            X = self.dropout(X)

        part_T = self.part_T.to(compute_dtype)
        part_F = self.part_F.to(compute_dtype)
        LiMEs_cast = self.LiMEs.to(compute_dtype)
        bias = self.bias.to(compute_dtype) if self.bias is not None else None

        # === 2-Part ROW-MODE SliceFine Forward ===
        # X stays CONTIGUOUS - no slicing!
        out_T = F.linear(X, part_T)  # (B, T, rank) - trainable output
        out_F = F.linear(X, part_F)  # (B, T, H-rank) - frozen output

        # === Routing (use out_T for routing signal if E <= rank) ===
        # This avoids creating full base_out just for routing
        if E <= self.rank:
            H_slice = out_T[:, :, :E]  # No cat needed!
        else:
            # Need to include some frozen outputs for routing
            H_slice = torch.cat([out_T, out_F[:, :, :E - self.rank]], dim=-1)
        
        # === N-GRAM ROUTING: subsample BEFORE computing probs ===
        if self.n_gram > 1:
            anchor_idx = _get_ngram_anchor_indices(T, self.n_gram, out_T.device)
            H_slice = H_slice[:, anchor_idx, :]

        if self.training and self.jitter_noise > 0:
            probs = _routing_softmax_with_jitter(
                H_slice,
                self._temperature_val,
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise,
            )
        else:
            probs = _routing_softmax(H_slice, self._temperature_val)

        # === MIXING LOGIC ===
        topk_idx = None
        selection_mask = None

        if self.sparse_mode:
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
            if self.auto_topk:
                weights, selection_mask = _auto_topk(
                    probs, self.auto_topk_threshold, self.soft_topk_temperature
                )
                p_mix = weights @ LiMEs_cast
            elif self.soft_topk:
                weights, topk_idx = _soft_topk(
                    probs, self.top_k, self.soft_topk_temperature
                )
                p_mix = weights @ LiMEs_cast
            else:
                topk_vals, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)
                topk_sum = topk_vals.sum(dim=-1, keepdim=True).clamp_min_(1e-9)
                topk_vals = topk_vals / topk_sum
                p_mix = _sparse_expert_mix(topk_vals, topk_idx, LiMEs_cast)

        # === N-GRAM: expand p_mix and probs back to full sequence ===
        if self.n_gram > 1:
            p_mix = _expand_ngram_values(p_mix, T, self.n_gram)
            probs = _expand_ngram_probs(probs, T, self.n_gram)

        # Aux tracking
        if self.track_aux:
            if self.training:
                aux = _moe_balance_losses_optimized(probs, topk_idx, selection_mask, attention_mask)
                self._store_aux(aux)
            else:
                self._store_usage(probs.mean(dim=(0, 1)).detach())

        # === Apply MoE routing - OPTIMIZED ===
        # Since delta = [out_T | zeros], MoE only affects out_T:
        # out[:,:,:rank] = out_T + out_T * (p_mix[:,:,:rank] - 1) = out_T * p_mix[:,:,:rank]
        # out[:,:,rank:] = out_F (unchanged)
        #
        # This avoids creating full (B,T,H) delta tensor!
        
        p_mix_T = p_mix[:, :, :self.rank]  # (B, T, rank) - slice of p_mix for trainable part
        
        # Apply MoE to trainable output only
        out_T_routed = out_T * p_mix_T  # (B, T, rank)
        
        # Shared LiME contribution (only affects trainable part)
        if self.use_shared_LiME and self.LiME_shared is not None:
            shared_cast = self.LiME_shared.to(compute_dtype)
            gamma_cast = self.gamma.to(compute_dtype)
            shared_T = shared_cast[:self.rank]  # First rank dims of shared
            out_T_routed = out_T_routed + gamma_cast * out_T * shared_T
        
        # Concatenate final output (single cat, no intermediate delta)
        out = torch.cat([out_T_routed, out_F], dim=-1)  # (B, T, H)
        
        if bias is not None:
            out = out + bias

        # Mask handling
        if attention_mask is not None:
            msk = attention_mask.bool()
            if msk.dim() == 2:
                msk = msk.unsqueeze(-1)
            # For masked positions, use unrouted output
            base_out = torch.cat([out_T, out_F], dim=-1)
            if bias is not None:
                base_out = base_out + bias
            out = torch.where(msk, out, base_out)

        return out.squeeze(1) if is_2d else out

    def extra_repr(self) -> str:
        mode = "auto" if self.auto_topk else ("soft" if self.soft_topk else "hard")
        sparse_str = ", sparse" if self.sparse_mode else ""
        ngram_str = f", n_gram={self.n_gram}" if self.n_gram > 1 else ""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, experts={self.num_experts}, topk={self.top_k}, "
            f"mode={mode}{sparse_str}{ngram_str}, partition=row"
        )


# =============================================================================
# SliceFineLiMEEmbedding - ROW MODE
# =============================================================================
class SliceFineLiMEEmbedding(nn.Module, _StoresAux):
    """
    SliceFine + LiME Embedding Layer (2-Part ROW mode).

    Partitions embedding by ROWS (output dimensions):
        E = [part_T]  (V, rank)   <- trainable (first `rank` output dims)
            [part_F]  (V, H-rank) <- frozen (remaining output dims)

    Note: For embeddings, "rows" means output embedding dimensions.
    
    ROUTING: Uses only base_out[:, :, :E] for routing (no delta_slice).
    This is simpler and avoids issues with zero-padded delta in row-mode.
    
    sparse_mode: If True, uses MoE-LoRA style sparse gather (faster).
                 If False, uses dense matmul (original behavior).
    
    n_gram: Number of tokens per routing group.
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
        pad_token_id: Optional[int] = None,
        track_aux: bool = True,
        rep_mode: str = "token",
        jitter_noise: float = 0.0,
        n_gram: int = 1,
        LiME_init_std: float = 0.1,
        soft_topk: bool = True,
        soft_topk_temperature: float = 0.5,
        auto_topk: bool = False,
        auto_topk_threshold: float = 0.5,
        peft_dtype: Optional[torch.dtype] = None,
        moe_dtype: Optional[torch.dtype] = None,
        sparse_mode: bool = False,
    ):
        super().__init__()
        assert isinstance(embedding, nn.Embedding)
        assert 1 <= top_k <= num_experts
        assert num_experts <= embedding.embedding_dim
        assert 0.0 < auto_topk_threshold <= 1.0
        assert rank > 0
        assert rep_mode in ("token", "last", "mean")
        assert n_gram >= 1, "n_gram must be >= 1"

        V, H = embedding.num_embeddings, embedding.embedding_dim
        # ROW MODE: rank must be <= H (embedding dim)
        assert rank <= H, f"rank ({rank}) must be <= embedding_dim ({H})"

        self.num_embeddings = V
        self.embedding_dim = H
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

        self._temperature_val = float(temperature)

        self.peft_dtype = peft_dtype if peft_dtype is not None else torch.float32
        self.moe_dtype = moe_dtype if moe_dtype is not None else torch.float32
        base_device = embedding.weight.device

        self.padding_idx = embedding.padding_idx

        # === Partition embedding by output dimensions (ROW mode) ===
        with torch.no_grad():
            W = embedding.weight.detach()  # (V, H)

            # ROW MODE for embedding: part_T is (V, rank), part_F is (V, H-rank)
            # This partitions the OUTPUT dimensions of the embedding
            self.part_T = nn.Parameter(
                W[:, :rank].clone().contiguous().to(self.peft_dtype),
                requires_grad=True,
            )
            self.part_F = nn.Parameter(
                W[:, rank:].clone().contiguous().to(self.peft_dtype),
                requires_grad=False,
            )

        # === LiME Expert Vectors ===
        self.LiMEs = nn.Parameter(
            torch.empty(num_experts, H, device=base_device, dtype=self.moe_dtype).uniform_(
                1.0 - LiME_init_std, 1.0 + LiME_init_std
            )
        )

        if use_shared_LiME:
            self.LiME_shared = nn.Parameter(
                torch.zeros(H, device=base_device, dtype=self.moe_dtype)
            )
            self.gamma = nn.Parameter(
                torch.zeros(1, device=base_device, dtype=self.moe_dtype)
            )
        else:
            self.register_parameter("LiME_shared", None)
            self.register_parameter("gamma", None)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstruct full embedding matrix."""
        return torch.cat([self.part_T, self.part_F], dim=1)  # (V, H)

    @property
    def temperature(self) -> float:
        return self._temperature_val

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = getattr(self, "_broadcast_attention_mask", None)
        if attention_mask is None and self.pad_token_id is not None:
            attention_mask = input_ids != self.pad_token_id

        B, T = input_ids.shape
        E = self.num_experts
        compute_dtype = self.part_T.dtype

        LiMEs_cast = self.LiMEs.to(compute_dtype)

        # === 2-Part ROW-MODE SliceFine Embedding Forward ===
        emb_T = F.embedding(input_ids, self.part_T, padding_idx=self.padding_idx)  # (B, T, rank)
        emb_F = F.embedding(input_ids, self.part_F, padding_idx=self.padding_idx)  # (B, T, H-rank)

        # === Routing (use emb_T for routing signal if E <= rank) ===
        if E <= self.rank:
            H_slice = emb_T[:, :, :E]  # No cat needed!
        else:
            H_slice = torch.cat([emb_T, emb_F[:, :, :E - self.rank]], dim=-1)
        
        # === N-GRAM ROUTING: subsample BEFORE computing probs ===
        if self.n_gram > 1:
            anchor_idx = _get_ngram_anchor_indices(T, self.n_gram, emb_T.device)
            H_slice = H_slice[:, anchor_idx, :]

        if self.training and self.jitter_noise > 0:
            probs = _routing_softmax_with_jitter(
                H_slice,
                self._temperature_val,
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise,
            )
        else:
            probs = _routing_softmax(H_slice, self._temperature_val)

        # === MIXING LOGIC ===
        topk_idx = None
        selection_mask = None

        if self.sparse_mode:
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
            if self.auto_topk:
                weights, selection_mask = _auto_topk(
                    probs, self.auto_topk_threshold, self.soft_topk_temperature
                )
                p_mix = weights @ LiMEs_cast
            elif self.soft_topk:
                weights, topk_idx = _soft_topk(
                    probs, self.top_k, self.soft_topk_temperature
                )
                p_mix = weights @ LiMEs_cast
            else:
                topk_vals, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)
                topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min_(1e-9)
                p_mix = _sparse_expert_mix(topk_vals, topk_idx, LiMEs_cast)

        # === N-GRAM: expand p_mix and probs back to full sequence ===
        if self.n_gram > 1:
            p_mix = _expand_ngram_values(p_mix, T, self.n_gram)
            probs = _expand_ngram_probs(probs, T, self.n_gram)

        # Aux tracking
        if self.track_aux:
            if self.training:
                self._store_aux(
                    _moe_balance_losses_optimized(probs, topk_idx, selection_mask, attention_mask)
                )
            else:
                self._store_usage(probs.mean(dim=(0, 1)).detach())

        # === Apply MoE routing - OPTIMIZED ===
        # Since delta = [emb_T | zeros], MoE only affects emb_T:
        # out[:,:,:rank] = emb_T * p_mix[:,:,:rank]
        # out[:,:,rank:] = emb_F (unchanged)
        
        p_mix_T = p_mix[:, :, :self.rank]  # (B, T, rank)
        
        # Apply MoE to trainable embedding only
        emb_T_routed = emb_T * p_mix_T  # (B, T, rank)
        
        # Shared LiME contribution
        if self.use_shared_LiME and self.LiME_shared is not None:
            shared_cast = self.LiME_shared.to(compute_dtype)
            gamma_cast = self.gamma.to(compute_dtype)
            shared_T = shared_cast[:self.rank]
            emb_T_routed = emb_T_routed + gamma_cast * emb_T * shared_T
        
        # Concatenate final output (single cat, no intermediate delta)
        out = torch.cat([emb_T_routed, emb_F], dim=-1)  # (B, T, H)

        # Mask handling
        if attention_mask is not None:
            msk = attention_mask.bool()
            if msk.dim() == 2:
                msk = msk.unsqueeze(-1)
            base_out = torch.cat([emb_T, emb_F], dim=-1)
            out = torch.where(msk, out, base_out)

        return out

    def extra_repr(self) -> str:
        mode = "auto" if self.auto_topk else ("soft" if self.soft_topk else "hard")
        sparse_str = ", sparse" if self.sparse_mode else ""
        ngram_str = f", n_gram={self.n_gram}" if self.n_gram > 1 else ""
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"rank={self.rank}, experts={self.num_experts}, topk={self.top_k}, "
            f"mode={mode}{sparse_str}{ngram_str}, partition=row"
        )


# =============================================================================
# Apply PEFT
# =============================================================================
def _is_tied_with_linear(emb_mod, parent, child_name):
    """Check if embedding is tied with a linear layer."""
    try:
        w = emb_mod.weight
        for n, m in parent.named_children():
            if n != child_name and isinstance(m, nn.Linear):
                if getattr(m, "weight", None) is w:
                    return True
    except Exception:
        pass
    return False


def _replace_layers(
    parent: nn.Module,
    module_path: str = "",
    targets=None,
    num_experts: int = 8,
    rank: int = 8,
    alpha: float = 16.0,
    top_k: int = 2,
    dropout: float = 0.0,
    temperature: float = 1.0,
    use_shared_LiME: bool = True,
    preserve_tying: bool = True,
    pad_token_id: Optional[int] = None,
    track_aux: bool = True,
    rep_mode: str = "token",
    jitter_noise: float = 0.0,
    n_gram: int = 1,
    LiME_init_std: float = 0.1,
    soft_topk: bool = True,
    soft_topk_temperature: float = 0.5,
    auto_topk: bool = False,
    auto_topk_threshold: float = 0.5,
    peft_dtype: Optional[torch.dtype] = None,
    moe_dtype: Optional[torch.dtype] = None,
    sparse_mode: bool = False,
):
    """Recursively replace layers with SliceFine-LiME ROW-MODE versions."""
    if targets is None:
        def wants(path, mod):
            return isinstance(mod, nn.Linear)
    elif callable(targets):
        wants = targets
    else:
        target_list = list(targets)
        def wants(path, mod):
            return isinstance(mod, nn.Linear) and any(
                path.endswith(t) for t in target_list
            )

    for child_name, child in list(parent.named_children()):
        path = f"{module_path}.{child_name}" if module_path else child_name

        if isinstance(child, (SliceFineLiMELinear, SliceFineLiMEEmbedding)):
            continue

        if isinstance(child, nn.Linear) and wants(path, child):
            wrapped = SliceFineLiMELinear(
                child,
                num_experts=num_experts,
                rank=rank,
                alpha=alpha,
                top_k=top_k,
                dropout=dropout,
                temperature=temperature,
                use_shared_LiME=use_shared_LiME,
                track_aux=track_aux,
                rep_mode=rep_mode,
                jitter_noise=jitter_noise,
                n_gram=n_gram,
                LiME_init_std=LiME_init_std,
                soft_topk=soft_topk,
                soft_topk_temperature=soft_topk_temperature,
                auto_topk=auto_topk,
                auto_topk_threshold=auto_topk_threshold,
                peft_dtype=peft_dtype,
                moe_dtype=moe_dtype,
                sparse_mode=sparse_mode,
            )
            setattr(parent, child_name, wrapped)

        elif isinstance(child, nn.Embedding) and wants(path, child):
            if preserve_tying and _is_tied_with_linear(child, parent, child_name):
                continue
            wrapped = SliceFineLiMEEmbedding(
                child,
                num_experts=num_experts,
                rank=rank,
                alpha=alpha,
                top_k=top_k,
                temperature=temperature,
                use_shared_LiME=use_shared_LiME,
                pad_token_id=pad_token_id,
                track_aux=track_aux,
                rep_mode=rep_mode,
                jitter_noise=jitter_noise,
                n_gram=n_gram,
                LiME_init_std=LiME_init_std,
                soft_topk=soft_topk,
                soft_topk_temperature=soft_topk_temperature,
                auto_topk=auto_topk,
                auto_topk_threshold=auto_topk_threshold,
                peft_dtype=peft_dtype,
                moe_dtype=moe_dtype,
                sparse_mode=sparse_mode,
            )
            setattr(parent, child_name, wrapped)
        else:
            _replace_layers(
                child, path, targets, num_experts, rank, alpha, top_k, dropout,
                temperature, use_shared_LiME, preserve_tying, pad_token_id,
                track_aux, rep_mode, jitter_noise, n_gram, LiME_init_std,
                soft_topk, soft_topk_temperature,
                auto_topk, auto_topk_threshold, peft_dtype, moe_dtype,
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
    pad_token_id: Optional[int] = None,
    tokenizer=None,
    track_aux: bool = True,
    rep_mode: str = "token",
    jitter_noise: float = 0.0,
    n_gram: int = 1,
    gamma_routing=0.7,
    LiME_init_std: float = 0.1,
    soft_topk: bool = True,
    soft_topk_temperature: float = 0.5,
    auto_topk: bool = False,
    auto_topk_threshold: float = 0.5,
    peft_dtype: Optional[torch.dtype] = None,
    moe_dtype: Optional[torch.dtype] = None,
    use_compile: bool = False,
    sparse_mode: bool = False,
) -> nn.Module:
    """
    Apply SliceFine-LiME PEFT to a model (2-Part ROW-MODE version).

    ROW-MODE partitions by OUTPUT features:
    - Linear: W = [part_T (rank, Din); part_F (H-rank, Din)]
    - Embedding: E = [part_T (V, rank); part_F (V, H-rank)]
    
    ADVANTAGES:
    - Input tensors stay CONTIGUOUS - no slicing along last dim!
    - Simpler routing using only hidden state (no delta_slice needed)

    Args:
        n_gram: Number of tokens per routing group (1 = per-token routing).
        sparse_mode: If True, uses MoE-LoRA style sparse gather (faster).
    """
    if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is not None:
        pad_token_id = tokenizer.pad_token_id

    for p in model.parameters():
        p.requires_grad = False

    _replace_layers(
        model, "", targets, num_experts, rank, alpha, top_k, dropout,
        temperature, use_shared_LiME, preserve_tying, pad_token_id,
        track_aux, rep_mode, jitter_noise, n_gram, LiME_init_std,
        soft_topk, soft_topk_temperature,
        auto_topk, auto_topk_threshold, peft_dtype, moe_dtype,
        sparse_mode,
    )

    if use_compile and hasattr(torch, "compile"):
        for m in model.modules():
            if isinstance(m, (SliceFineLiMELinear, SliceFineLiMEEmbedding)):
                m.forward = torch.compile(m.forward, mode="reduce-overhead")

    return model


# =============================================================================
# Mask Broadcaster
# =============================================================================
def enable_mask_broadcast(model: nn.Module):
    """Enable attention mask broadcasting."""
    if getattr(model, "_peft_broadcast_wrapped", False):
        return

    orig_forward = model.forward

    def wrapped_forward(*args, **kwargs):
        mask = kwargs.get("attention_mask", None)
        for m in model.modules():
            if isinstance(m, (SliceFineLiMELinear, SliceFineLiMEEmbedding)):
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


# =============================================================================
# Aux Loss Collection
# =============================================================================
def collect_and_zero_moe_aux(
    model: nn.Module, target_device: Optional[torch.device] = None
) -> tuple:
    """Collect aux losses (multi-GPU compatible)."""
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


def compute_balance_loss(
    model: nn.Module,
    importance_coef: float = 1.0,
    kl_coef: float = 0.1,
    target_device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute balance loss."""
    totals, count = collect_and_zero_moe_aux(model, target_device=target_device)
    if count == 0:
        device = target_device if target_device else next(model.parameters()).device
        return torch.tensor(0.0, device=device, requires_grad=True)
    return (importance_coef * totals["importance"] + kl_coef * totals["kl_uniform"]) / count


def get_expert_usage_stats(model: nn.Module) -> Dict:
    """Get expert usage statistics."""
    usages = []
    for m in model.modules():
        if hasattr(m, "_last_usage") and m._last_usage is not None:
            usages.append(m._last_usage.cpu())

    if not usages:
        for m in model.modules():
            if hasattr(m, "_last_aux") and m._last_aux is not None:
                if "usage" in m._last_aux:
                    usages.append(m._last_aux["usage"].cpu())

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
    """Clear auxiliary data."""
    for m in model.modules():
        if hasattr(m, "_last_aux"):
            m._last_aux = None
        if hasattr(m, "_last_usage"):
            m._last_usage = None


# =============================================================================
# Utilities
# =============================================================================
def set_temperature(model: nn.Module, temp: float):
    for m in model.modules():
        if isinstance(m, (SliceFineLiMELinear, SliceFineLiMEEmbedding)):
            m._temperature_val = temp


def set_soft_topk(model: nn.Module, soft_topk: bool, temperature: Optional[float] = None):
    for m in model.modules():
        if isinstance(m, (SliceFineLiMELinear, SliceFineLiMEEmbedding)):
            m.soft_topk = soft_topk
            if temperature is not None:
                m.soft_topk_temperature = temperature


def set_auto_topk(
    model: nn.Module,
    auto_topk: bool,
    threshold: Optional[float] = None,
    temperature: Optional[float] = None,
):
    for m in model.modules():
        if isinstance(m, (SliceFineLiMELinear, SliceFineLiMEEmbedding)):
            m.auto_topk = auto_topk
            if threshold is not None:
                m.auto_topk_threshold = threshold
            if temperature is not None:
                m.soft_topk_temperature = temperature


def set_sparse_mode(model: nn.Module, sparse: bool):
    """Set sparse mode for all SliceFine-LiME layers."""
    for m in model.modules():
        if isinstance(m, (SliceFineLiMELinear, SliceFineLiMEEmbedding)):
            m.sparse_mode = sparse


def set_jitter(model: nn.Module, jitter: float):
    for m in model.modules():
        if isinstance(m, (SliceFineLiMELinear, SliceFineLiMEEmbedding)):
            m.jitter_noise = jitter


def set_n_gram(model: nn.Module, n_gram: int):
    """Set n_gram for all SliceFine-LiME layers."""
    assert n_gram >= 1, "n_gram must be >= 1"
    for m in model.modules():
        if isinstance(m, (SliceFineLiMELinear, SliceFineLiMEEmbedding)):
            m.n_gram = n_gram


def set_rep_mode(model: nn.Module, rep_mode: str):
    assert rep_mode in ("token", "last", "mean")
    for m in model.modules():
        if isinstance(m, (SliceFineLiMELinear, SliceFineLiMEEmbedding)):
            m.rep_mode = rep_mode


def set_peft_dtype(model: nn.Module, dtype: torch.dtype):
    for m in model.modules():
        if isinstance(m, (SliceFineLiMELinear, SliceFineLiMEEmbedding)):
            m.part_T.data = m.part_T.data.to(dtype)
            m.part_F.data = m.part_F.data.to(dtype)
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.data = m.bias.data.to(dtype)
            m.peft_dtype = dtype


def set_moe_dtype(model: nn.Module, dtype: torch.dtype):
    for m in model.modules():
        if isinstance(m, (SliceFineLiMELinear, SliceFineLiMEEmbedding)):
            m.LiMEs.data = m.LiMEs.data.to(dtype)
            if m.LiME_shared is not None:
                m.LiME_shared.data = m.LiME_shared.data.to(dtype)
            if m.gamma is not None:
                m.gamma.data = m.gamma.data.to(dtype)
            m.moe_dtype = dtype


def get_auto_topk_stats(model: nn.Module) -> Dict:
    stats = []
    for name, m in model.named_modules():
        if isinstance(m, (SliceFineLiMELinear, SliceFineLiMEEmbedding)) and m.auto_topk:
            if hasattr(m, "_last_usage") and m._last_usage is not None:
                usage = m._last_usage
                num_active = (usage > 1e-6).sum().item()
                stats.append({"layer": name, "active_experts": num_active})

    if not stats:
        return {"layers": [], "overall_avg": 0.0}

    overall_avg = sum(s.get("active_experts", 0) for s in stats) / len(stats)
    return {"layers": stats, "overall_avg": overall_avg}


def get_trainable_param_count(model: nn.Module) -> Dict:
    trainable = 0
    frozen = 0

    for m in model.modules():
        if isinstance(m, SliceFineLiMELinear):
            frozen += m.part_F.numel()
            trainable += m.part_T.numel()
            trainable += m.LiMEs.numel()
            if m.LiME_shared is not None:
                trainable += m.LiME_shared.numel()
            if m.gamma is not None:
                trainable += m.gamma.numel()
            if m.bias is not None:
                trainable += m.bias.numel()
        elif isinstance(m, SliceFineLiMEEmbedding):
            frozen += m.part_F.numel()
            trainable += m.part_T.numel()
            trainable += m.LiMEs.numel()
            if m.LiME_shared is not None:
                trainable += m.LiME_shared.numel()
            if m.gamma is not None:
                trainable += m.gamma.numel()

    total = trainable + frozen
    return {
        "trainable": trainable,
        "frozen": frozen,
        "total": total,
        "trainable_ratio": trainable / total if total > 0 else 0,
    }


def print_peft_summary(model: nn.Module):
    """Print summary of SliceFine-LiME ROW-MODE configuration."""
    total_frozen = 0
    total_trainable_slice = 0
    total_LiMEs = 0
    total_bias = 0
    peft_dtypes = set()
    moe_dtypes = set()
    topk_modes = set()
    sparse_modes = set()
    num_layers = 0
    ranks = set()
    n_grams = set()

    for name, m in model.named_modules():
        if isinstance(m, SliceFineLiMELinear):
            total_frozen += m.part_F.numel()
            total_trainable_slice += m.part_T.numel()
            total_LiMEs += m.LiMEs.numel()
            if m.LiME_shared is not None:
                total_LiMEs += m.LiME_shared.numel() + m.gamma.numel()
            if m.bias is not None:
                total_bias += m.bias.numel()
            peft_dtypes.add(m.part_T.dtype)
            moe_dtypes.add(m.LiMEs.dtype)
            ranks.add(m.rank)
            n_grams.add(m.n_gram)
            if m.auto_topk:
                topk_modes.add(f"auto(θ={m.auto_topk_threshold})")
            elif m.soft_topk:
                topk_modes.add(f"soft(k={m.top_k})")
            else:
                topk_modes.add(f"hard(k={m.top_k})")
            sparse_modes.add("sparse" if m.sparse_mode else "dense")
            num_layers += 1

        elif isinstance(m, SliceFineLiMEEmbedding):
            total_frozen += m.part_F.numel()
            total_trainable_slice += m.part_T.numel()
            total_LiMEs += m.LiMEs.numel()
            if m.LiME_shared is not None:
                total_LiMEs += m.LiME_shared.numel() + m.gamma.numel()
            peft_dtypes.add(m.part_T.dtype)
            moe_dtypes.add(m.LiMEs.dtype)
            ranks.add(m.rank)
            n_grams.add(m.n_gram)
            if m.auto_topk:
                topk_modes.add(f"auto(θ={m.auto_topk_threshold})")
            elif m.soft_topk:
                topk_modes.add(f"soft(k={m.top_k})")
            else:
                topk_modes.add(f"hard(k={m.top_k})")
            sparse_modes.add("sparse" if m.sparse_mode else "dense")
            num_layers += 1

    total_trainable = total_trainable_slice + total_LiMEs + total_bias

    print("=" * 60)
    print("SliceFine-LiME ROW-MODE Summary (MEMORY OPTIMIZED)")
    print("  - Input CONTIGUOUS (no slicing)")
    print("  - No F.pad() allocation")
    print("  - Single cat() at output")
    print("=" * 60)
    print(f"Wrapped layers:        {num_layers}")
    print(f"Slice ranks:           {sorted(ranks)}")
    print(f"N-gram(s):             {sorted(n_grams)}")
    print(f"Frozen parameters:     {total_frozen:,}")
    print(f"Trainable slice:       {total_trainable_slice:,}")
    print(f"LiME parameters:       {total_LiMEs:,}")
    print(f"Bias parameters:       {total_bias:,}")
    print(f"Total trainable:       {total_trainable:,}")
    print(f"PEFT dtype(s):         {peft_dtypes}")
    print(f"MoE dtype(s):          {moe_dtypes}")
    print(f"Top-K mode(s):         {topk_modes}")
    print(f"Mixing mode(s):        {sparse_modes}")
    print("=" * 60)


def print_LiME_summary(model: nn.Module):
    """Alias for print_peft_summary."""
    print_peft_summary(model)


def print_routing_summary(model: nn.Module):
    """Alias for print_peft_summary."""
    print_peft_summary(model)


def compile_layers(model: nn.Module, mode: str = "reduce-overhead"):
    if not hasattr(torch, "compile"):
        print("Warning: torch.compile not available")
        return

    count = 0
    for m in model.modules():
        if isinstance(m, (SliceFineLiMELinear, SliceFineLiMEEmbedding)):
            m.forward = torch.compile(m.forward, mode=mode)
            count += 1

    print(f"Compiled {count} SliceFine-LiME ROW-MODE layers with mode='{mode}'")


def compile_moe_layers(model: nn.Module, mode: str = "reduce-overhead"):
    """Alias for compile_layers."""
    compile_layers(model, mode)