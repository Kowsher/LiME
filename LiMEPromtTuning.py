# LiME for Prompt Tuning - MANUAL INTEGRATION VERSION
# ====================================================
# 
# This version does NOT auto-wrap model forward.
# Instead, use the helper functions to integrate manually.
#
# Usage:
#   prompt_module = apply_peft(model, num_virtual_tokens=50, wrap_forward=False)
#   
#   # In your training loop or collator:
#   inputs_embeds, attention_mask = prepare_inputs_with_prompts(
#       prompt_module, inputs_embeds, attention_mask
#   )
#   labels = adjust_labels_for_prompts(prompt_module, labels)

import math
import functools
from typing import Optional, Dict, List, Union, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Aux Storage Mixin
# =============================================================================

class _StoresAux:
    _last_aux: Optional[Dict] = None
    _last_usage: Optional[torch.Tensor] = None
    track_aux: bool = True
    
    def _store_aux(self, aux: Dict):
        self._last_aux = aux
    
    def _store_usage(self, usage: torch.Tensor):
        self._last_usage = usage


# =============================================================================
# JIT Helpers
# =============================================================================

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
                               jitter_low: float, jitter_high: float, eps: float = 1e-6) -> torch.Tensor:
    H_scale = H_slice.abs().max().clamp_min(eps)
    delta_scale = delta_slice.abs().max().clamp_min(eps)
    logits = (1.0 - gamma) * (H_slice / H_scale) + gamma * (delta_slice / delta_scale)
    logits = logits * (1.0 / max(temperature, eps))
    jitter = torch.empty_like(logits).uniform_(jitter_low, jitter_high)
    logits = logits * jitter
    return F.softmax(logits, dim=-1)


def _get_ngram_anchor_indices(T: int, n_gram: int, device: torch.device) -> torch.Tensor:
    num_full_groups = T // n_gram
    remainder = T % n_gram
    anchors = torch.arange(n_gram - 1, num_full_groups * n_gram, n_gram, device=device)
    if remainder > 0:
        anchors = torch.cat([anchors, torch.tensor([T - 1], device=device)])
    return anchors


def _expand_ngram_values(anchor_values: torch.Tensor, T: int, n_gram: int) -> torch.Tensor:
    *batch_dims, num_anchors, D = anchor_values.shape
    num_full_groups = T // n_gram
    remainder = T % n_gram
    if remainder == 0:
        return anchor_values.repeat_interleave(n_gram, dim=-2)
    full = anchor_values[..., :num_full_groups, :].repeat_interleave(n_gram, dim=-2)
    last = anchor_values[..., -1:, :].expand(*batch_dims, remainder, D)
    return torch.cat([full, last], dim=-2)


def _expand_ngram_probs(anchor_probs: torch.Tensor, T: int, n_gram: int) -> torch.Tensor:
    return _expand_ngram_values(anchor_probs, T, n_gram)


@torch.jit.script
def _sparse_topk_k1(probs: torch.Tensor, LiMEs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = torch.argmax(probs, dim=-1)
    return LiMEs[idx], idx.unsqueeze(-1)


@torch.jit.script
def _sparse_topk(probs: torch.Tensor, LiMEs: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    topk_vals, topk_idx = torch.topk(probs, k=k, dim=-1)
    topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    selected = LiMEs[topk_idx]
    p_mix = (topk_vals.unsqueeze(-1) * selected).sum(dim=-2)
    return p_mix, topk_idx


@torch.jit.script
def _sparse_auto_topk(probs: torch.Tensor, LiMEs: torch.Tensor, 
                      threshold: float, max_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    topk_vals, topk_idx = torch.topk(probs, k=max_k, dim=-1)
    max_prob = probs.max(dim=-1, keepdim=True).values
    thresh_val = threshold * max_prob
    topk_mask = topk_vals >= thresh_val
    topk_vals = topk_vals * topk_mask.float()
    row_sums = topk_vals.sum(dim=-1, keepdim=True)
    zero_mask = row_sums < 1e-9
    if zero_mask.any():
        fallback = torch.zeros_like(topk_vals)
        fallback[..., 0:1] = probs.max(dim=-1, keepdim=True).values
        topk_vals = torch.where(zero_mask.expand_as(topk_vals), fallback, topk_vals)
    topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    selected = LiMEs[topk_idx]
    p_mix = (topk_vals.unsqueeze(-1) * selected).sum(dim=-2)
    return p_mix, topk_idx


@torch.jit.script
def _sparse_expert_mix(topk_vals: torch.Tensor, topk_idx: torch.Tensor, 
                       LiMEs: torch.Tensor) -> torch.Tensor:
    return (topk_vals.unsqueeze(-1) * LiMEs[topk_idx]).sum(dim=-2)


@torch.jit.script
def _fast_soft_topk_jit(probs: torch.Tensor, k: int, 
                        temperature: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    topk_vals, topk_idx = torch.topk(probs, k=k, dim=-1)
    threshold = topk_vals[..., -1:]
    mask = torch.sigmoid((probs - threshold) * (1.0 / temperature))
    weights = probs * mask
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
    return weights, topk_idx


@torch.jit.script
def _fast_auto_topk_jit(probs: torch.Tensor, threshold: float, 
                        temperature: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    max_prob = probs.max(dim=-1, keepdim=True).values
    thresh_val = threshold * max_prob
    mask = torch.sigmoid((probs - thresh_val) * (1.0 / temperature))
    weights = probs * mask
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
    selection_mask = probs >= thresh_val
    return weights, selection_mask


def _moe_balance_losses(probs: torch.Tensor, topk_idx: Optional[torch.Tensor] = None,
                        selection_mask: Optional[torch.Tensor] = None,
                        mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    if probs.dim() == 1:
        probs = probs.unsqueeze(0).unsqueeze(0)
    elif probs.dim() == 2:
        probs = probs.unsqueeze(0)
        if topk_idx is not None:
            topk_idx = topk_idx.unsqueeze(0)
        if selection_mask is not None:
            selection_mask = selection_mask.unsqueeze(0)
    
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
    
    scv_importance = E_float * (p_bar * p_bar).sum() - 1.0
    
    if selection_mask is not None:
        load = selection_mask.float().sum(dim=(0, 1))
    elif topk_idx is not None:
        load = torch.zeros(E, device=device, dtype=probs.dtype)
        load.scatter_add_(0, topk_idx.reshape(-1), 
                          torch.ones(topk_idx.numel(), device=device, dtype=probs.dtype))
    else:
        load = probs.detach().sum(dim=(0, 1))
    
    load_norm = load / load.sum().clamp_min(1.0)
    scv_load = (E_float * (load_norm * load_norm).sum() - 1.0).detach()
    kl_uniform = (p_bar * (p_bar.clamp_min(1e-8).log() + math.log(E_float))).sum()
    
    return {"importance": scv_importance, "load": scv_load, 
            "kl_uniform": kl_uniform, "usage": p_bar.detach().clone()}


# =============================================================================
# LiME Prompt Tuning Module
# =============================================================================

class LiMEPromptTuning(nn.Module, _StoresAux):
    def __init__(
        self,
        num_virtual_tokens: int,
        hidden_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        learnable_base: bool = False,
        input_conditioned: bool = True,
        input_conditioning_mode: Literal["mean", "first", "last", "attention"] = "mean",
        conditioning_projection: bool = True,
        temperature: float = 1.0,
        gamma_routing: float = 0.5,
        jitter_noise: float = 0.0,
        n_gram: int = 1,
        soft_topk: bool = True,
        soft_topk_temperature: float = 0.5,
        auto_topk: bool = False,
        auto_topk_threshold: float = 0.5,
        sparse_mode: bool = True,
        use_shared_LiME: bool = True,
        init_from_vocab: bool = False,
        init_token_ids: Optional[List[int]] = None,
        embedding_layer: Optional[nn.Embedding] = None,
        LiME_init_std: float = 0.1,
        track_aux: bool = True,
        prompt_dropout: float = 0.0,
        rep_mode: str = "token",
        prompt_dtype: Optional[torch.dtype] = None,
        moe_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        assert num_virtual_tokens > 0
        assert num_experts > 0 and 1 <= top_k <= num_experts
        assert num_experts <= hidden_size
        assert 0.0 < auto_topk_threshold <= 1.0
        assert n_gram >= 1
        
        self.num_virtual_tokens = num_virtual_tokens
        self.hidden_size = hidden_size
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.learnable_base = learnable_base
        self.input_conditioned = input_conditioned
        self.input_conditioning_mode = input_conditioning_mode
        self._temperature_val = float(temperature)
        self._gamma_val = float(gamma_routing)
        self.jitter_noise = float(jitter_noise)
        self.n_gram = int(n_gram)
        self.soft_topk = bool(soft_topk)
        self.soft_topk_temperature = float(soft_topk_temperature)
        self.auto_topk = bool(auto_topk)
        self.auto_topk_threshold = float(auto_topk_threshold)
        self.sparse_mode = bool(sparse_mode)
        self.use_shared_LiME = bool(use_shared_LiME)
        self.track_aux = bool(track_aux)
        self.rep_mode = rep_mode
        self.prompt_dtype = prompt_dtype or torch.float32
        self.moe_dtype = moe_dtype or torch.float32
        self._last_aux = None
        self._last_usage = None
        
        base_init = torch.randn(num_virtual_tokens, hidden_size, dtype=self.prompt_dtype) * 0.02
        if learnable_base:
            self.prompt_base = nn.Parameter(base_init)
        else:
            self.register_buffer('prompt_base', base_init)
        
        self.prompt_delta = nn.Parameter(torch.zeros(num_virtual_tokens, hidden_size, dtype=self.prompt_dtype))
        
        if init_from_vocab and embedding_layer is not None and init_token_ids is not None:
            self._init_from_vocab(embedding_layer, init_token_ids)
        
        self.LiMEs = nn.Parameter(
            torch.empty(num_experts, hidden_size, dtype=self.moe_dtype).uniform_(1.0 - LiME_init_std, 1.0 + LiME_init_std)
        )
        
        if use_shared_LiME:
            self.LiME_shared = nn.Parameter(torch.randn(hidden_size, dtype=self.moe_dtype) * 0.1)
            self.gamma = nn.Parameter(torch.zeros(1, dtype=self.moe_dtype))
        else:
            self.register_parameter("LiME_shared", None)
            self.register_parameter("gamma", None)
        
        if input_conditioned and conditioning_projection:
            self.input_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            nn.init.eye_(self.input_proj.weight)
        else:
            self.input_proj = nn.Identity()
        
        self.dropout = nn.Dropout(prompt_dropout) if prompt_dropout > 0 else nn.Identity()
    
    @property
    def gamma_routing(self) -> float:
        return self._gamma_val
    
    @property
    def temperature(self) -> float:
        return self._temperature_val
    
    def _init_from_vocab(self, embedding_layer: nn.Embedding, token_ids: List[int]):
        with torch.no_grad():
            token_ids = list(token_ids)[:self.num_virtual_tokens]
            if len(token_ids) < self.num_virtual_tokens:
                token_ids = (token_ids * ((self.num_virtual_tokens // len(token_ids)) + 1))[:self.num_virtual_tokens]
            token_ids_tensor = torch.tensor(token_ids, device=embedding_layer.weight.device)
            init_embeds = embedding_layer(token_ids_tensor).to(self.prompt_dtype)
            if isinstance(self.prompt_base, nn.Parameter):
                self.prompt_base.data.copy_(init_embeds)
            else:
                self.prompt_base.copy_(init_embeds)
            self.prompt_delta.data.zero_()
    
    def _get_cached_params(self, compute_dtype: torch.dtype):
        LiMEs_cast = self.LiMEs.to(compute_dtype)
        shared_cast = self.LiME_shared.to(compute_dtype) if self.LiME_shared is not None else None
        gamma_cast = self.gamma.to(compute_dtype) if self.gamma is not None else None
        return LiMEs_cast, shared_cast, gamma_cast
    
    def _get_input_representation(self, input_embeds: torch.Tensor, 
                                   attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.input_conditioning_mode == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                return self.input_proj((input_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0))
            return self.input_proj(input_embeds.mean(dim=1))
        elif self.input_conditioning_mode == "first":
            return self.input_proj(input_embeds[:, 0, :])
        elif self.input_conditioning_mode == "last":
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1).long() - 1
                batch_idx = torch.arange(input_embeds.shape[0], device=input_embeds.device)
                return self.input_proj(input_embeds[batch_idx, lengths.clamp_min(0), :])
            return self.input_proj(input_embeds[:, -1, :])
        elif self.input_conditioning_mode == "attention":
            query = input_embeds.mean(dim=1, keepdim=True)
            scores = (input_embeds @ query.transpose(-1, -2)).squeeze(-1)
            if attention_mask is not None:
                scores = scores.masked_fill(~attention_mask.bool(), float('-inf'))
            weights = F.softmax(scores, dim=-1)
            return self.input_proj((weights.unsqueeze(-1) * input_embeds).sum(dim=1))
        raise ValueError(f"Unknown mode: {self.input_conditioning_mode}")
    
    def _get_routing_probs(self, H: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        E = self.num_experts
        H_slice, delta_slice = H[..., :E], delta[..., :E]
        if self.training and self.jitter_noise > 0:
            return _fused_routing_with_jitter(H_slice, delta_slice, self._gamma_val, self._temperature_val,
                                              1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        return _fused_routing_softmax(H_slice, delta_slice, self._gamma_val, self._temperature_val)
    
    def _mix_experts(self, probs: torch.Tensor, LiMEs_cast: torch.Tensor):
        topk_idx, selection_mask = None, None
        if self.sparse_mode:
            if self.auto_topk:
                p_mix, topk_idx = _sparse_auto_topk(probs, LiMEs_cast, self.auto_topk_threshold, 
                                                     min(self.top_k * 2, self.num_experts))
            elif self.top_k == 1:
                p_mix, topk_idx = _sparse_topk_k1(probs, LiMEs_cast)
            else:
                p_mix, topk_idx = _sparse_topk(probs, LiMEs_cast, self.top_k)
        else:
            if self.auto_topk:
                weights, selection_mask = _fast_auto_topk_jit(probs, self.auto_topk_threshold, self.soft_topk_temperature)
                p_mix = weights @ LiMEs_cast
            elif self.soft_topk:
                weights, topk_idx = _fast_soft_topk_jit(probs, self.top_k, self.soft_topk_temperature)
                p_mix = weights @ LiMEs_cast
            else:
                topk_vals, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)
                topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min_(1e-9)
                p_mix = _sparse_expert_mix(topk_vals, topk_idx, LiMEs_cast)
        return p_mix, topk_idx, selection_mask
    
    def _compute_prompt_embeddings_static(self, batch_size: int):
        H, delta, T = self.prompt_base, self.prompt_delta, self.num_virtual_tokens
        LiMEs_cast, shared_cast, gamma_cast = self._get_cached_params(delta.dtype)
        
        if self.n_gram > 1:
            anchor_idx = _get_ngram_anchor_indices(T, self.n_gram, H.device)
            H_routing, delta_routing = H[anchor_idx, :], delta[anchor_idx, :]
        else:
            H_routing, delta_routing = H, delta
        
        probs = self._get_routing_probs(H_routing, delta_routing)
        p_mix, topk_idx, selection_mask = self._mix_experts(probs, LiMEs_cast)
        
        if self.n_gram > 1:
            p_mix = _expand_ngram_values(p_mix.unsqueeze(0), T, self.n_gram).squeeze(0)
            probs = _expand_ngram_probs(probs.unsqueeze(0), T, self.n_gram).squeeze(0)
        
        routed_delta = delta * p_mix
        if self.use_shared_LiME and shared_cast is not None:
            routed_delta = routed_delta + (gamma_cast * (delta * shared_cast))
        
        prompt_embeds = H + routed_delta
        
        if self.track_aux:
            if self.training:
                self._store_aux(_moe_balance_losses(probs, topk_idx, selection_mask, None))
            else:
                self._store_usage(probs.mean(dim=0).detach())
        
        return prompt_embeds.unsqueeze(0).expand(batch_size, -1, -1).clone(), probs
    
    def _compute_prompt_embeddings_conditioned(self, input_embeds: torch.Tensor, 
                                                attention_mask: Optional[torch.Tensor] = None):
        batch_size, T = input_embeds.shape[0], self.num_virtual_tokens
        input_rep = self._get_input_representation(input_embeds, attention_mask)
        H = self.prompt_base.unsqueeze(0).expand(batch_size, -1, -1)
        delta = self.prompt_delta.unsqueeze(0).expand(batch_size, -1, -1)
        H_conditioned = H + input_rep.unsqueeze(1)
        
        LiMEs_cast, shared_cast, gamma_cast = self._get_cached_params(delta.dtype)
        
        if self.n_gram > 1:
            anchor_idx = _get_ngram_anchor_indices(T, self.n_gram, H.device)
            H_routing, delta_routing = H_conditioned[:, anchor_idx, :], delta[:, anchor_idx, :]
        else:
            H_routing, delta_routing = H_conditioned, delta
        
        probs = self._get_routing_probs(H_routing, delta_routing)
        p_mix, topk_idx, selection_mask = self._mix_experts(probs, LiMEs_cast)
        
        if self.n_gram > 1:
            p_mix = _expand_ngram_values(p_mix, T, self.n_gram)
            probs = _expand_ngram_probs(probs, T, self.n_gram)
        
        routed_delta = delta * p_mix
        if self.use_shared_LiME and shared_cast is not None:
            routed_delta = routed_delta + (gamma_cast * (delta * shared_cast))
        
        prompt_embeds = H + routed_delta
        
        if self.track_aux:
            if self.training:
                self._store_aux(_moe_balance_losses(probs, topk_idx, selection_mask, None))
            else:
                self._store_usage(probs.mean(dim=(0, 1)).detach())
        
        return prompt_embeds, probs
    
    def get_prompt_embeddings(self, batch_size: int = 1, input_embeds: Optional[torch.Tensor] = None,
                               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.input_conditioned and input_embeds is not None:
            prompt_embeds, _ = self._compute_prompt_embeddings_conditioned(input_embeds, attention_mask)
        else:
            if input_embeds is not None:
                batch_size = input_embeds.shape[0]
            prompt_embeds, _ = self._compute_prompt_embeddings_static(batch_size)
        return self.dropout(prompt_embeds) if self.training else prompt_embeds
    
    def forward(self, input_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                return_prompt_only: bool = False):
        batch_size = input_embeds.shape[0]
        prompt_embeds = self.get_prompt_embeddings(batch_size, 
                                                    input_embeds if self.input_conditioned else None,
                                                    attention_mask if self.input_conditioned else None)
        if prompt_embeds.dtype != input_embeds.dtype:
            prompt_embeds = prompt_embeds.to(input_embeds.dtype)
        
        if return_prompt_only:
            return prompt_embeds
        
        combined_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
        combined_mask = None
        if attention_mask is not None:
            prompt_mask = torch.ones(batch_size, self.num_virtual_tokens, 
                                     dtype=attention_mask.dtype, device=attention_mask.device)
            combined_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        return combined_embeds, combined_mask
    
    def get_aux_loss(self, importance_coef: float = 1.0, kl_coef: float = 0.1) -> torch.Tensor:
        if self._last_aux is None:
            return torch.tensor(0.0, device=self.prompt_delta.device, requires_grad=True)
        aux = self._last_aux
        loss = importance_coef * aux["importance"] + kl_coef * aux["kl_uniform"]
        self._last_aux = None
        return loss
    
    def get_usage_stats(self) -> Dict:
        usage = self._last_usage.cpu() if self._last_usage is not None else (
            self._last_aux["usage"].cpu() if self._last_aux and "usage" in self._last_aux else None)
        if usage is None:
            return {"usage_per_expert": [], "entropy": 0.0, "max_usage": 0.0, "min_usage": 0.0, "imbalance_ratio": 1.0}
        entropy = -(usage * torch.log(usage + 1e-9)).sum().item()
        return {"usage_per_expert": usage.tolist(), "entropy": entropy, "max_usage": usage.max().item(),
                "min_usage": usage.min().item(), "imbalance_ratio": usage.max().item() / (usage.min().item() + 1e-9)}
    
    def clear_aux(self):
        self._last_aux = self._last_usage = None
    
    def __repr__(self) -> str:
        mode = "auto" if self.auto_topk else ("soft" if self.soft_topk else "hard")
        return (f"LiMEPromptTuning(tokens={self.num_virtual_tokens}, hidden={self.hidden_size}, "
                f"experts={self.num_experts}, top_k={self.top_k}, mode={mode})")


# =============================================================================
# Helper Functions for Manual Integration
# =============================================================================

def prepare_inputs_with_prompts(
    prompt_module: LiMEPromptTuning,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Prepend soft prompts to inputs_embeds and adjust attention_mask.
    
    Use this in your training loop or data collator.
    """
    return prompt_module(inputs_embeds, attention_mask)


def adjust_labels_for_prompts(
    prompt_module: LiMEPromptTuning,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Prepend -100 (ignore_index) to labels to account for prompt tokens.
    
    Use this in your training loop or data collator.
    """
    batch_size = labels.shape[0]
    num_prompt_tokens = prompt_module.num_virtual_tokens
    prompt_labels = torch.full(
        (batch_size, num_prompt_tokens),
        -100,
        dtype=labels.dtype,
        device=labels.device
    )
    return torch.cat([prompt_labels, labels], dim=1)


# =============================================================================
# Trainer Integration
# =============================================================================

PROMPT_TUNING_PEFT_PATTERNS = ['prompt_delta', 'input_proj']
PROMPT_TUNING_MOE_PATTERNS = ['LiMEs', 'LiME_shared', '.gamma']


def get_trainer_patterns() -> Tuple[List[str], List[str]]:
    return PROMPT_TUNING_PEFT_PATTERNS.copy(), PROMPT_TUNING_MOE_PATTERNS.copy()


def make_prompt_tuning_param_groups(prompt_module: LiMEPromptTuning, moe_lr: float = 1e-3,
                                     peft_lr: float = 2e-4, weight_decay: float = 0.01) -> List[Dict]:
    moe_params, peft_params = [], []
    for name, param in prompt_module.named_parameters():
        if not param.requires_grad:
            continue
        if 'LiMEs' in name or 'LiME_shared' in name or name == 'gamma':
            moe_params.append(param)
        else:
            peft_params.append(param)
    
    groups = []
    if moe_params:
        groups.append({'params': moe_params, 'lr': moe_lr, 'weight_decay': 0.0, 'name': 'moe'})
    if peft_params:
        groups.append({'params': peft_params, 'lr': peft_lr, 'weight_decay': weight_decay, 'name': 'peft'})
    return groups


def create_prompt_tuning_optimizer(prompt_module: LiMEPromptTuning, moe_lr: float = 1e-3,
                                    peft_lr: float = 2e-4, weight_decay: float = 0.01):
    return torch.optim.AdamW(make_prompt_tuning_param_groups(prompt_module, moe_lr, peft_lr, weight_decay))


# =============================================================================
# Auto-detection Utilities
# =============================================================================

def _auto_detect_hidden_size(model: nn.Module) -> Optional[int]:
    for attr in ['config.text_config.hidden_size', 'config.hidden_size', 'config.d_model', 'config.n_embd']:
        try:
            obj = model
            for part in attr.split('.'):
                obj = getattr(obj, part)
            if isinstance(obj, int) and obj > 0:
                return obj
        except:
            pass
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) and 'vision' not in name.lower():
            return module.embedding_dim
    return None


def _find_embedding_layer(model: nn.Module) -> Optional[nn.Embedding]:
    paths = ['model.language_model.embed_tokens', 'language_model.embed_tokens', 
             'model.embed_tokens', 'transformer.wte', 'shared']
    for path in paths:
        try:
            obj = model
            for part in path.split('.'):
                obj = getattr(obj, part)
            if isinstance(obj, nn.Embedding):
                return obj
        except:
            pass
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) and 'vision' not in name.lower():
            if 'embed_tokens' in name or 'wte' in name:
                return module
    return None


def _find_language_model(model: nn.Module):
    for path in ['model.language_model', 'language_model', 'model.model', 'model']:
        try:
            obj = model
            for part in path.split('.'):
                obj = getattr(obj, part)
            if hasattr(obj, 'embed_tokens') or hasattr(obj, 'wte'):
                return obj, path
        except:
            pass
    return None, None


# =============================================================================
# Forward Wrapping (Optional)
# =============================================================================

def _wrap_model_forward(model: nn.Module, prompt_module: LiMEPromptTuning):
    """
    Wrap model forward to auto-inject prompts and adjust labels.
    
    This wraps:
    1. Language model forward: inject prompts into inputs_embeds
    2. Main model forward: adjust labels
    """
    language_model, lm_path = _find_language_model(model)
    if language_model is None:
        print("WARNING: Could not find language model. Use manual integration.")
        return
    
    num_prompt_tokens = prompt_module.num_virtual_tokens
    original_lm_forward = language_model.forward
    
    @functools.wraps(original_lm_forward)
    def wrapped_lm_forward(input_ids=None, attention_mask=None, position_ids=None, 
                           past_key_values=None, inputs_embeds=None, labels=None,
                           use_cache=None, output_attentions=None, output_hidden_states=None,
                           return_dict=None, cache_position=None, **kwargs):
        # Skip if generation continuation
        if past_key_values is not None:
            return original_lm_forward(input_ids=input_ids, attention_mask=attention_mask,
                                       position_ids=position_ids, past_key_values=past_key_values,
                                       inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                                       output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                                       return_dict=return_dict, cache_position=cache_position, **kwargs)
        
        # Get embeddings
        if inputs_embeds is not None:
            batch_size, device, dtype = inputs_embeds.shape[0], inputs_embeds.device, inputs_embeds.dtype
        elif input_ids is not None:
            batch_size, device = input_ids.shape[0], input_ids.device
            if hasattr(language_model, 'embed_tokens'):
                inputs_embeds = language_model.embed_tokens(input_ids)
            elif hasattr(language_model, 'wte'):
                inputs_embeds = language_model.wte(input_ids)
            else:
                return original_lm_forward(input_ids=input_ids, attention_mask=attention_mask,
                                           position_ids=position_ids, past_key_values=past_key_values,
                                           inputs_embeds=inputs_embeds, labels=labels, **kwargs)
            dtype = inputs_embeds.dtype
            input_ids = None
        else:
            return original_lm_forward(input_ids=input_ids, attention_mask=attention_mask,
                                       position_ids=position_ids, past_key_values=past_key_values,
                                       inputs_embeds=inputs_embeds, labels=labels, **kwargs)
        
        # Prepend prompts
        prompt_embeds = prompt_module.get_prompt_embeddings(
            batch_size=batch_size,
            input_embeds=inputs_embeds if prompt_module.input_conditioned else None,
        ).to(dtype)
        inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)
        
        # Adjust attention mask
        if attention_mask is not None:
            prompt_mask = torch.ones(batch_size, num_prompt_tokens, dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        
        # Adjust position ids
        if position_ids is not None:
            prompt_positions = torch.arange(num_prompt_tokens, device=device).unsqueeze(0).expand(batch_size, -1)
            position_ids = torch.cat([prompt_positions, position_ids + num_prompt_tokens], dim=1)
        
        # Adjust cache position
        if cache_position is not None:
            prompt_cache_pos = torch.arange(num_prompt_tokens, device=device)
            cache_position = torch.cat([prompt_cache_pos, cache_position + num_prompt_tokens], dim=0)
        
        # DO NOT adjust labels here - done in main forward
        
        return original_lm_forward(input_ids=None, attention_mask=attention_mask,
                                   position_ids=position_ids, past_key_values=past_key_values,
                                   inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                                   output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                                   return_dict=return_dict, cache_position=cache_position, **kwargs)
    
    language_model.forward = wrapped_lm_forward
    language_model._original_forward = original_lm_forward
    
    # Wrap main model forward to adjust labels
    original_main_forward = model.forward
    
    @functools.wraps(original_main_forward)
    def wrapped_main_forward(input_ids=None, pixel_values=None, pixel_values_videos=None,
                             image_sizes=None, attention_mask=None, position_ids=None,
                             past_key_values=None, inputs_embeds=None, labels=None,
                             use_cache=None, output_attentions=None, output_hidden_states=None,
                             return_dict=None, cache_position=None, **kwargs):
        # Skip if generation
        if past_key_values is not None:
            return original_main_forward(input_ids=input_ids, pixel_values=pixel_values,
                                         pixel_values_videos=pixel_values_videos, image_sizes=image_sizes,
                                         attention_mask=attention_mask, position_ids=position_ids,
                                         past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                                         labels=labels, use_cache=use_cache, output_attentions=output_attentions,
                                         output_hidden_states=output_hidden_states, return_dict=return_dict,
                                         cache_position=cache_position, **kwargs)
        
        # Adjust labels HERE (single place)
        if labels is not None:
            batch_size = labels.shape[0]
            prompt_labels = torch.full((batch_size, num_prompt_tokens), -100, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([prompt_labels, labels], dim=1)
        
        return original_main_forward(input_ids=input_ids, pixel_values=pixel_values,
                                     pixel_values_videos=pixel_values_videos, image_sizes=image_sizes,
                                     attention_mask=attention_mask, position_ids=position_ids,
                                     past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                                     labels=labels, use_cache=use_cache, output_attentions=output_attentions,
                                     output_hidden_states=output_hidden_states, return_dict=return_dict,
                                     cache_position=cache_position, **kwargs)
    
    model.forward = wrapped_main_forward
    model._original_forward = original_main_forward
    model._prompt_tuning_enabled = True
    
    print(f"✓ Prompt tuning integrated: {num_prompt_tokens} tokens")
    print(f"  - LM forward ({lm_path}): prompts injected")
    print(f"  - Main forward: labels adjusted")


# =============================================================================
# Main Entry Point
# =============================================================================

def apply_peft(
    model: nn.Module,
    num_virtual_tokens: int,
    hidden_size: Optional[int] = None,
    num_experts: int = 8,
    top_k: int = 2,
    temperature: float = 1.0,
    gamma_routing: float = 0.5,
    jitter_noise: float = 0.0,
    n_gram: int = 1,
    soft_topk: bool = True,
    soft_topk_temperature: float = 0.5,
    auto_topk: bool = False,
    auto_topk_threshold: float = 0.5,
    sparse_mode: bool = True,
    use_shared_LiME: bool = True,
    input_conditioned: bool = True,
    input_conditioning_mode: str = "mean",
    init_from_vocab: bool = True,
    init_text: Optional[str] = None,
    tokenizer = None,
    LiME_init_std: float = 0.1,
    track_aux: bool = True,
    freeze_model: bool = True,
    prompt_dropout: float = 0.0,
    wrap_forward: bool = True,
    prompt_dtype: Optional[torch.dtype] = None,
    moe_dtype: Optional[torch.dtype] = None,
    **kwargs
) -> LiMEPromptTuning:
    """
    Create and attach LiME prompt tuning to a model.
    
    Args:
        wrap_forward: If True, auto-wrap model forward. If False, use manual integration
                      with prepare_inputs_with_prompts() and adjust_labels_for_prompts().
    """
    if hidden_size is None:
        hidden_size = _auto_detect_hidden_size(model)
        if hidden_size is None:
            raise ValueError("Could not auto-detect hidden_size. Please provide explicitly.")
    
    embedding_layer = _find_embedding_layer(model) if init_from_vocab else None
    init_token_ids = None
    if embedding_layer is not None:
        if embedding_layer.embedding_dim != hidden_size:
            print(f"WARNING: Embedding dim mismatch. Disabling init_from_vocab.")
            embedding_layer = None
        elif tokenizer is not None and init_text is not None:
            init_token_ids = tokenizer.encode(init_text, add_special_tokens=False)
        else:
            start_idx = min(100, embedding_layer.num_embeddings - num_virtual_tokens)
            init_token_ids = list(range(start_idx, start_idx + num_virtual_tokens))
    
    prompt_module = LiMEPromptTuning(
        num_virtual_tokens=num_virtual_tokens, hidden_size=hidden_size, num_experts=num_experts,
        top_k=top_k, temperature=temperature, gamma_routing=gamma_routing, jitter_noise=jitter_noise,
        n_gram=n_gram, soft_topk=soft_topk, soft_topk_temperature=soft_topk_temperature,
        auto_topk=auto_topk, auto_topk_threshold=auto_topk_threshold, sparse_mode=sparse_mode,
        use_shared_LiME=use_shared_LiME, input_conditioned=input_conditioned,
        input_conditioning_mode=input_conditioning_mode,
        init_from_vocab=init_from_vocab and embedding_layer is not None,
        init_token_ids=init_token_ids, embedding_layer=embedding_layer,
        LiME_init_std=LiME_init_std, track_aux=track_aux, prompt_dropout=prompt_dropout,
        prompt_dtype=prompt_dtype, moe_dtype=moe_dtype, **kwargs
    )
    
    if freeze_model:
        for param in model.parameters():
            param.requires_grad = False
    
    model.prompt_tuning = prompt_module
    
    try:
        device = next(model.parameters()).device
        prompt_module.to(device)
    except StopIteration:
        pass
    
    for param in prompt_module.parameters():
        param.requires_grad = True
    
    if wrap_forward:
        _wrap_model_forward(model, prompt_module)
    
    return prompt_module


apply_lime_prompt_tuning = apply_peft


# =============================================================================
# Setters
# =============================================================================

def set_gamma_routing(m: LiMEPromptTuning, v: float): m._gamma_val = v
def set_temperature(m: LiMEPromptTuning, v: float): m._temperature_val = v
def set_jitter(m: LiMEPromptTuning, v: float): m.jitter_noise = v
def set_n_gram(m: LiMEPromptTuning, v: int): m.n_gram = v
def set_sparse_mode(m: LiMEPromptTuning, v: bool): m.sparse_mode = v
def set_soft_topk(m: LiMEPromptTuning, v: bool, t: float = None): 
    m.soft_topk = v
    if t is not None: m.soft_topk_temperature = t
def set_auto_topk(m: LiMEPromptTuning, v: bool, threshold: float = None, t: float = None):
    m.auto_topk = v
    if threshold is not None: m.auto_topk_threshold = threshold
    if t is not None: m.soft_topk_temperature = t


def get_prompt_tuning_params(model: nn.Module) -> List[nn.Parameter]:
    if hasattr(model, 'prompt_tuning'):
        return [p for p in model.prompt_tuning.parameters() if p.requires_grad]
    return []


def print_prompt_tuning_summary(m: LiMEPromptTuning):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"LiMEPromptTuning: {m.num_virtual_tokens} tokens, {m.hidden_size} hidden, "
          f"{m.num_experts} experts, top_k={m.top_k}")
    print(f"  Total params: {total:,}, Trainable: {trainable:,}")


if __name__ == "__main__":
    print("Testing LiME Prompt Tuning...")
    pt = LiMEPromptTuning(8, 64, 8, 2, use_shared_LiME=True, jitter_noise=0.1, n_gram=2)
    x = torch.randn(4, 16, 64)
    pt.train()
    out, mask = pt(x, torch.ones(4, 16))
    assert out.shape == (4, 24, 64)
    print(f"✓ Forward: {x.shape} -> {out.shape}")
    loss = pt.get_aux_loss()
    print(f"✓ Aux loss: {loss.item():.4f}")
    print("All tests passed!")