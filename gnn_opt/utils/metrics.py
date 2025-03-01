"""Accuracy, timing, and training step utilities."""
import torch
import torch.nn.functional as F


def train_step_baseline(model, data, optimizer, split_idx):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[split_idx['train']], data.y[split_idx['train']].squeeze())
    loss.backward()
    optimizer.step()
    return loss.item()


def train_step_optimized(model, data, optimizer, split_idx):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index_csr)
    loss = F.cross_entropy(out[split_idx['train']], data.y[split_idx['train']].squeeze())
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, split_idx, use_csr=False):
    model.eval()
    if use_csr:
        out = model(data.x, data.edge_index_csr)
    else:
        out = model(data.x, data.edge_index)
    y_pred = out.argmax(dim=-1)
    accs = {}
    for key in ['train', 'valid', 'test']:
        if key not in split_idx:
            continue
        mask = split_idx[key]
        accs[key] = (y_pred[mask] == data.y[mask].squeeze()).float().mean().item()
    return accs
