"""embed2bit.py -- a 2-bit embedding table, so the comparison has a real reference.

The last run exposed a hole in my own test rather than in the quantiser: convert()
only rewrites nn.Linear, and embedding tables are nn.Embedding, so the
"include_emb" variant quantised exactly the same tensors as the other one and
produced a bit-identical loss. The study it was being checked against quantised every
ndim>=2 tensor, so the two were never comparable.

With an embedding path, the comparison becomes meaningful:

    study (all ndim>=2, embeddings included)   2 bit nf  ->  5.6206
    this module, Linears only                           ->  5.5538
    this module, Linears + embeddings                   ->  should reach 5.6206

and the difference between the last two is the cost of quantising embeddings, which
the study implies is about 0.067 against 0.050 for the Linear layers. Embeddings are
lookup tables, not matmuls, so it is plausible they deserve more precision -- and
that is a decision worth having a number for.
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn

sys.path.insert(0, r'D:\work\bnb-quant')
sys.stdout.reconfigure(encoding='utf-8')

from quant2bit import BLOCK, NF2, dequantize, quantize  # noqa: E402


class Embedding2bit(nn.Module):
    """A lookup table stored at 2 bits per entry.

    forward() dequantises the whole table, which is wasteful -- a lookup touches
    only the requested rows -- but keeps this a single clear code path while the
    point is to validate the format, not to optimise the access pattern.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, table=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._table = NF2 if table is None else table
        # packed/absmax are registered in from_embedding; do not pre-assign them
        # here, because register_buffer refuses to overwrite an existing attribute.

    @classmethod
    def from_embedding(cls, emb: nn.Embedding, table=None):
        e = cls(emb.num_embeddings, emb.embedding_dim, table)
        packed, absmax, n, pad, shape = quantize(emb.weight.detach().float(), table)
        e.register_buffer('packed', packed)
        e.register_buffer('absmax', absmax)
        e._n, e._pad, e._shape = n, pad, shape
        return e

    def weight(self) -> torch.Tensor:
        return dequantize(self.packed, self.absmax, self._n, self._pad, self._shape,
                          self._table)

    def forward(self, idx):
        return torch.nn.functional.embedding(idx, self.weight())

    def extra_repr(self):
        return '%d x %d, packed %d bytes' % (self.num_embeddings, self.embedding_dim,
                                             self.packed.numel())
