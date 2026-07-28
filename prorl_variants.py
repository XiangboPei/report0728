from collections import OrderedDict

import torch


METHOD_MODES = (
    'none',
    'pareto_advantage',
    'elite_memory',
    'difficulty_curriculum',
    'robust_normalization',
)


def pareto_rollout_scores(sequence_ioi, sequence_ior, sequence_ctr, valid_mask, eps=1e-8):
    """Pairwise Pareto dominance score for rollouts from the same input."""
    objectives = torch.stack([sequence_ioi, sequence_ior, sequence_ctr], dim=-1)
    left = objectives.unsqueeze(2)
    right = objectives.unsqueeze(1)

    weakly_better = (left >= right - eps).all(dim=-1)
    strictly_better = (left > right + eps).any(dim=-1)
    dominates = weakly_better & strictly_better

    num_samples = objectives.shape[1]
    off_diagonal = ~torch.eye(num_samples, dtype=torch.bool, device=objectives.device).unsqueeze(0)
    pair_mask = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1) & off_diagonal
    dominates = dominates & pair_mask

    dominated_count = dominates.float().sum(dim=2)
    dominating_count = dominates.transpose(1, 2).float().sum(dim=2)
    comparison_count = pair_mask.float().sum(dim=2).clamp(min=1.0)
    scores = (dominated_count - dominating_count) / comparison_count

    scores = scores * valid_mask.float()
    valid_count = valid_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
    scores = scores - (scores.sum(dim=1, keepdim=True) / valid_count)
    return scores * valid_mask.float()


def pareto_front_scores(sequence_ioi, sequence_ior, sequence_ctr, valid_mask, eps=1e-8):
    """Standard non-dominated front ranks, centered for policy gradients."""
    objectives = torch.stack([sequence_ioi, sequence_ior, sequence_ctr], dim=-1)
    left = objectives.unsqueeze(2)
    right = objectives.unsqueeze(1)
    weakly_better = (left >= right - eps).all(dim=-1)
    strictly_better = (left > right + eps).any(dim=-1)
    dominates = weakly_better & strictly_better

    num_samples = objectives.shape[1]
    off_diagonal = ~torch.eye(
        num_samples, dtype=torch.bool, device=objectives.device
    ).unsqueeze(0)
    pair_mask = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1) & off_diagonal
    dominates = dominates & pair_mask

    remaining = valid_mask.clone()
    front_ranks = sequence_ioi.new_zeros(sequence_ioi.shape)
    for front_index in range(num_samples):
        active_pairs = remaining.unsqueeze(2) & remaining.unsqueeze(1)
        is_dominated = (dominates & active_pairs).any(dim=1)
        current_front = remaining & ~is_dominated
        if not current_front.any():
            break
        front_ranks = torch.where(
            current_front,
            torch.full_like(front_ranks, float(front_index)),
            front_ranks,
        )
        remaining = remaining & ~current_front

    scores = -front_ranks * valid_mask.float()
    valid_count = valid_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
    scores = scores - scores.sum(dim=1, keepdim=True) / valid_count
    scores = scores * valid_mask.float()
    scale = scores.abs().amax(dim=1, keepdim=True).clamp(min=1.0)
    return scores / scale


def masked_pearson_correlation(left, right, valid_mask, eps=1e-8):
    left_values = left[valid_mask].float()
    right_values = right[valid_mask].float()
    if left_values.numel() < 2:
        return left_values.new_zeros(())
    left_centered = left_values - left_values.mean()
    right_centered = right_values - right_values.mean()
    denominator = (
        left_centered.square().sum().sqrt() *
        right_centered.square().sum().sqrt()
    ).clamp(min=eps)
    return (left_centered * right_centered).sum() / denominator


def normalized_target_quality(normalized_rewards, item_mask):
    target_steps = normalized_rewards['ioi'] + normalized_rewards['ior']
    return (target_steps * item_mask.float()).sum(dim=-1)


def curriculum_sample_weights(initial_ranks, valid_mask, epoch, total_epochs,
                              start_competence=0.5, temperature=0.25):
    """Easy-to-hard weights based on the initial target-rank percentile in a batch."""
    if not 0 < start_competence <= 1:
        raise ValueError('curriculum_start must be in (0, 1]')
    if temperature <= 0:
        raise ValueError('curriculum_temperature must be positive')

    context_valid = valid_mask.any(dim=1)
    positive_rank = initial_ranks > 0
    fallback = torch.full_like(initial_ranks, float('inf'))
    context_rank = torch.where(positive_rank, initial_ranks, fallback).min(dim=1).values
    context_rank = torch.where(context_valid, context_rank, torch.zeros_like(context_rank))

    batch_size = context_rank.shape[0]
    if batch_size <= 1:
        quantiles = torch.zeros_like(context_rank)
    else:
        order = torch.argsort(context_rank)
        quantiles = torch.empty_like(context_rank)
        quantiles[order] = torch.linspace(
            0.0, 1.0, steps=batch_size, device=context_rank.device, dtype=context_rank.dtype
        )

    denominator = max(int(total_epochs) - 1, 1)
    progress = min(max(float(epoch) / denominator, 0.0), 1.0)
    competence = start_competence + (1.0 - start_competence) * progress
    weights = torch.exp(-torch.relu(quantiles - competence) / temperature)
    weights = weights * context_valid.float()
    if context_valid.any():
        weights = weights / weights[context_valid].mean().clamp(min=1e-8)
    return weights, quantiles, competence


def bounded_normalization(normalized_rewards, scale=2.5):
    """Smoothly winsorize z-scores while preserving their local ordering."""
    if scale <= 0:
        raise ValueError('robust_scale must be positive')
    return {
        name: scale * torch.tanh(values / scale)
        for name, values in normalized_rewards.items()
    }


class EliteTrajectoryMemory:
    """CPU-backed best-trajectory memory keyed by the exact model input."""

    def __init__(self, capacity=4096):
        if capacity < 1:
            raise ValueError('memory_capacity must be positive')
        self.capacity = int(capacity)
        self.entries = OrderedDict()

    @staticmethod
    def _keys(input_ids):
        return [tuple(row) for row in input_ids.detach().cpu().tolist()]

    def fetch(self, input_ids, seq_len, device):
        sequences = []
        batch_indices = []
        for batch_idx, key in enumerate(self._keys(input_ids)):
            entry = self.entries.get(key)
            if entry is None:
                continue
            sequence = entry['sequence']
            if sequence.numel() < seq_len:
                sequence = torch.cat([
                    sequence,
                    torch.zeros(seq_len - sequence.numel(), dtype=sequence.dtype),
                ])
            sequences.append(sequence[:seq_len])
            batch_indices.append(batch_idx)
            self.entries.move_to_end(key)

        if not sequences:
            return None, None
        return (
            torch.stack(sequences).to(device=device, dtype=torch.long),
            torch.tensor(batch_indices, device=device, dtype=torch.long),
        )

    def update(self, input_ids, generated_sequences, quality, valid_mask):
        keys = self._keys(input_ids)
        quality_cpu = quality.detach().cpu()
        valid_cpu = valid_mask.detach().cpu()
        sequences_cpu = generated_sequences.detach().cpu()

        updates = 0
        for batch_idx, key in enumerate(keys):
            candidate_scores = quality_cpu[batch_idx].clone()
            candidate_scores[~valid_cpu[batch_idx]] = float('-inf')
            best_score, best_rollout = candidate_scores.max(dim=0)
            if not torch.isfinite(best_score):
                continue

            score = float(best_score.item())
            previous = self.entries.get(key)
            if previous is None or score > previous['score']:
                self.entries[key] = {
                    'score': score,
                    'sequence': sequences_cpu[batch_idx, int(best_rollout.item())].clone(),
                }
                self.entries.move_to_end(key)
                updates += 1

        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return updates

    def __len__(self):
        return len(self.entries)
