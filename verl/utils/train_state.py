"""Per-rank FSDP training state for resuming a run.

Each rank saves its local flat-parameter shards, its optimizer state, the LR scheduler and its
RNG states to <path>/rank<r>.pt, so a resumed run on the same number of GPUs continues from
exactly the saved state. A checksum over parameters and optimizer state is stored with the
state and re-checked after loading.
"""

import os

import torch


def _checksum(module, optimizer) -> float:
    total = 0.0
    with torch.no_grad():
        for _, p in module.named_parameters():
            total += float(p.detach().double().sum().item())
        for state in optimizer.state.values():
            for v in state.values():
                if isinstance(v, torch.Tensor):
                    total += float(v.detach().double().sum().item())
    return total


def save_fsdp_train_state(path, rank, module, optimizer, lr_scheduler):
    """Call with the parameters in place (on GPU or CPU, as training keeps them)."""
    os.makedirs(path, exist_ok=True)
    state = {
        'params': {n: p.detach().to('cpu', copy=True) for n, p in module.named_parameters()},
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
        'cpu_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state(),
        'checksum': _checksum(module, optimizer),
    }
    tmp = os.path.join(path, f'rank{rank}.pt.tmp')
    torch.save(state, tmp)
    os.replace(tmp, os.path.join(path, f'rank{rank}.pt'))


def load_fsdp_train_state(path, rank, module, optimizer, lr_scheduler):
    """Call with the parameters on GPU and the optimizer state wherever training keeps it;
    the optimizer state is placed on the parameters' device. The learning rate stays the
    configured one, so a run can resume with a changed optim.lr."""
    configured_lrs = list(lr_scheduler.base_lrs) if lr_scheduler is not None else None
    state = torch.load(os.path.join(path, f'rank{rank}.pt'), map_location='cpu')
    with torch.no_grad():
        for n, p in module.named_parameters():
            src = state['params'][n]
            assert p.data.shape == src.shape, f'{n}: {tuple(p.data.shape)} vs saved {tuple(src.shape)}'
            p.data.copy_(src)
            shard = getattr(p, '_local_shard', None)
            if shard is not None and shard.data_ptr() != p.data.data_ptr() and shard.shape == src.shape:
                shard.copy_(src)
    optimizer.load_state_dict(state['optimizer'])
    if lr_scheduler is not None and state['lr_scheduler'] is not None:
        lr_scheduler.load_state_dict(state['lr_scheduler'])
    if configured_lrs is not None and list(lr_scheduler.base_lrs) != configured_lrs:
        if rank == 0:
            print(f'[resume] learning rate {lr_scheduler.base_lrs} -> configured {configured_lrs}')
        lr_scheduler.base_lrs = configured_lrs
        for group, base, lr in zip(optimizer.param_groups, configured_lrs, lr_scheduler.get_lr()):
            group['initial_lr'] = base
            group['lr'] = lr
    torch.set_rng_state(state['cpu_rng'])
    torch.cuda.set_rng_state(state['cuda_rng'])
    got = _checksum(module, optimizer)
    want = state['checksum']
    assert abs(got - want) <= 1e-6 * max(1.0, abs(want)), f'train state checksum {got} != saved {want}'
    return got
