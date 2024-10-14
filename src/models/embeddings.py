"""
https://github.com/facebookresearch/DiT/blob/main/models.py#L27
"""
import torch
from torch import nn
import math

class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, freq_dim:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(freq_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.embedding_dim = embedding_dim
        self.freq_dim = freq_dim

    @staticmethod
    def pos_encod(t, dim:int ,max_period:int = 10_000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output := D
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """

        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, 1, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args),torch.cos(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        emb = self.pos_encod(t, self.freq_dim)
        emb = self.net(emb)
        return emb




if __name__=="__main__":
    te = TimeEmbedding(10, 10)
    te.pos_encod(torch.tensor([1,2,3,4,5,6,7,8,9,10]), 10, 10_000)