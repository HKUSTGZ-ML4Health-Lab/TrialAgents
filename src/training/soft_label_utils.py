"""
Soft Label Utility Functions

Migrated from m-agents/utils/soft_label_utils.py
"""
import torch
import numpy as np


def build_soft_labels(labels, method='direct', **kwargs):
    """
    Build soft labels from various input formats.

    Args:
        labels: Input labels (multiple formats supported)
        method: Construction method ('direct', 'confidence', 'smoothing', 'temperature')
        **kwargs: Method-specific parameters

    Returns:
        torch.Tensor: Soft labels [N, num_classes]
    """
    if method == 'direct':
        if labels.dim() == 2:
            return labels / labels.sum(dim=1, keepdim=True)
        else:
            return hard_to_onehot(labels, kwargs.get('num_classes', 2))

    elif method == 'confidence':
        hard_labels = kwargs.get('hard_labels', labels)
        confidences = kwargs.get('confidences', torch.ones_like(hard_labels) * 0.9)
        return confidence_to_soft_labels(hard_labels, confidences)

    elif method == 'smoothing':
        epsilon = kwargs.get('epsilon', 0.1)
        num_classes = kwargs.get('num_classes', 2)
        return label_smoothing(labels, epsilon, num_classes)

    elif method == 'temperature':
        temperature = kwargs.get('temperature', 1.0)
        return temperature_scaling(labels, temperature)

    else:
        raise ValueError(f"Unknown method: {method}")


def hard_to_onehot(labels, num_classes=2):
    """Convert hard labels to one-hot encoding."""
    if labels.dim() == 2:
        return labels.float()
    onehot = torch.zeros(len(labels), num_classes)
    onehot.scatter_(1, labels.long().unsqueeze(1), 1)
    return onehot


def confidence_to_soft_labels(hard_labels, confidences):
    """Build soft labels from hard labels and confidences."""
    batch_size = len(hard_labels)
    soft_labels = torch.zeros(batch_size, 2)
    for i in range(batch_size):
        if hard_labels[i] == 0:
            soft_labels[i] = torch.tensor([confidences[i], 1 - confidences[i]])
        else:
            soft_labels[i] = torch.tensor([1 - confidences[i], confidences[i]])
    return soft_labels


def label_smoothing(labels, epsilon=0.1, num_classes=2):
    """Apply label smoothing."""
    if labels.dim() == 2:
        return (1 - epsilon) * labels + epsilon / num_classes
    smooth_labels = torch.zeros(len(labels), num_classes)
    smooth_labels.fill_(epsilon / num_classes)
    smooth_labels.scatter_(1, labels.long().unsqueeze(1), 1 - epsilon + epsilon / num_classes)
    return smooth_labels


def temperature_scaling(logits, temperature=1.0):
    """Apply temperature scaling."""
    return torch.softmax(logits / temperature, dim=1)


def continuous_to_soft_labels(continuous_values, method='direct'):
    """Convert continuous values to soft labels."""
    if method == 'direct':
        pos_prob = torch.clamp(continuous_values, 0, 1)
        neg_prob = 1 - pos_prob
        return torch.stack([neg_prob, pos_prob], dim=1)

    elif method == 'sigmoid':
        pos_prob = torch.sigmoid(continuous_values)
        neg_prob = 1 - pos_prob
        return torch.stack([neg_prob, pos_prob], dim=1)

    elif method == 'threshold':
        threshold = 0.5
        distances = torch.abs(continuous_values - threshold)
        confidences = 1 - distances
        hard_labels = (continuous_values > threshold).long()
        return confidence_to_soft_labels(hard_labels, confidences)


def mix_labels(hard_labels, soft_labels, mixing_ratio=0.8):
    """Mix hard labels and soft labels."""
    if hard_labels.dim() == 1:
        hard_onehot = hard_to_onehot(hard_labels)
    else:
        hard_onehot = hard_labels
    return mixing_ratio * soft_labels + (1 - mixing_ratio) * hard_onehot


def validate_soft_labels(soft_labels, tolerance=1e-6):
    """Validate soft labels are proper probability distributions."""
    sums = soft_labels.sum(dim=1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=tolerance):
        print(f"Warning: soft labels not valid probability distributions. Sum range: {sums.min():.6f} - {sums.max():.6f}")
        return False
    if (soft_labels < 0).any():
        print("Warning: soft labels contain negative values")
        return False
    return True
