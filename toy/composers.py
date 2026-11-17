"""Attention-DDQN strict source-entry composer and explicit component ablations."""
from repaired.paper_aligned_attention import PaperAlignedAttention
from repaired.paper_aligned_run import train_upper, evaluate_attention
from repaired.evidence_ablations import (EntryGreedy, SelfOnlyAttention,
    NoSFAttention, NoTeammateContextSF, make_ablation_model,
    train_ablation_upper, train_ablation_lower, evaluate_ablation)
__all__=['PaperAlignedAttention','train_upper','evaluate_attention','EntryGreedy',
    'SelfOnlyAttention','NoSFAttention','NoTeammateContextSF','make_ablation_model',
    'train_ablation_upper','train_ablation_lower','evaluate_ablation']
