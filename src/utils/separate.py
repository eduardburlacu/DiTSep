import torch

def shuffle_sources(x):
    """
    Shuffle along the second dimension with a different permutation for each
    batch entry
    """
    if x.ndim <= 1:
        return x

    n_extra_dim = x.ndim - 2

    # generate a random permutation per batch entry
    c = x.new_zeros(x.shape[:2]).uniform_()
    idx = torch.argsort(c, dim=1)
    idx = torch.broadcast_to(idx[(...,) + (None,) * n_extra_dim], x.shape)

    # re-order the target tensor
    x = torch.gather(x, dim=1, index=idx)

    return x


def select_elem_at_random(x, dim=-1, batch_dim=0):
    x = x.moveaxis(dim, -1)
    select = torch.randint(x.shape[-1], size=(x.shape[batch_dim],), device=x.device)
    select = torch.broadcast_to(
        select[(...,) + (None,) * (x.ndim - 1)], x.shape[:-1] + (1,)
    )
    x = torch.gather(x, dim=-1, index=select)
    x = x.moveaxis(-1, dim)
    return x


def power_order_sources(x):
    """
    Shuffle along the second dimension with a different permutation for each
    batch entry
    """
    if x.ndim <= 1:
        return x

    n_extra_dim = x.ndim - 2

    # generate a random permutation per batch entry
    c = torch.var(x, dim=-1)
    idx = torch.argsort(c, dim=1)
    idx = torch.broadcast_to(idx[(...,) + (None,) * n_extra_dim], x.shape)

    # re-order the target tensor
    x = torch.gather(x, dim=1, index=idx)

    return x


def normalize_batch(batch):
    mix, tgt = batch
    mean = mix.mean(dim=(1, 2), keepdim=True)
    std = mix.std(dim=(1, 2), keepdim=True).clamp(min=1e-5)
    mix = (mix - mean) / std
    if tgt is not None:
        tgt = (tgt - mean) / std
    return (mix, tgt), mean, std


def denormalize_batch(x, mean, std):
    return x * std + mean
