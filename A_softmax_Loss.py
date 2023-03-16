import torch
import torch.nn.functional as F
from math import cos

def compute_loss(logits, f, y, m):
    y = y.view(-1, 1)  # reshape y

    # Get the target logits (cosine values)
    target_logits = torch.gather(logits, 1, y)

    # Calculate the angle between logits and their labels
    theta = torch.acos(target_logits)

    # Add the margin to the angle
    target_theta_plus_m = theta + m

    # Calculate the target logits with added margin
    target_logits_with_margin = torch.cos(target_theta_plus_m)

    # Keep the original logits for non-target classes
    final_logits = logits * (1 - y) + target_logits_with_margin * y

    # Calculate the cross-entropy loss
    loss = F.cross_entropy(final_logits, y.view(-1))

    return loss

