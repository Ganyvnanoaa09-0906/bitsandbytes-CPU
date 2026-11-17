"""NF-codebook low-bit weight storage (2/3-bit packed) for CPU inference and frozen bases.

Why this module exists
----------------------
``q3bit`` already stores *uniform* 2/3-bit codes, and the 2-bit path there is explicitly
experimental. The nf3/nf2 study (``quantize_model_eval.py`` on the 32x32 AR checkpoint)
measured that at 2 bits a **normal mid-quantile** codebook beats uniform spacing by ~11%
of the quantisation loss (5.6206 vs 5.6354 at the same block size), while at 3 bits the
two are equivalent. That codebook had no packed storage format in the package: the
research scripts quantised into an 8-bit index tensor, which saves nothing.

This module provides the missing format:

* ``packed``: 4 codes/byte (2-bit) or 8 codes/3 bytes (3-bit), first code in the low bits;
* ``absmax``: one fp32 scale per ``block`` weights, so the payload is
  ``bits + 32/block`` bits per weight (2.5 bit/weight at ``block=64`` for 2-bit,
  3.375 bit/weight for 3-bit);
* optional per-layer LoRA adapters, so a frozen low-bit base can still be adapted.

Honest boundaries (measured, not aspirational)
----------------------------------------------
* **Storage only.** There is no native 2/3-bit GEMM on CPU: ``gemm`` and the module
  forward dequantise into an fp32 temporary and call ``F.linear``. That is a memory win,
  not a speed win; account for the temporary with :meth:`QuantLinearNF.forward_peak_bytes`.
* **Packed + frozen base + LoRA is not a cheap training recipe** at 2 bits: on Qwen1.7B
  the conversion costs +9.5 nats and 60 LoRA steps only recovered part of it, while the
  step time grew ~38% (see the dev12 notes). For *training* a low-bit base, use QAT+STE.
* The two quantisation knobs that matter are the codebook (NF vs uniform) and the block
  size; both are arguments, and both are included in the checkpoint round-trip tests.

Layout is deliberately identical to ``q3bit`` where the formats overlap, so a packed
tensor produced here can be handed to ``q3bit.unpack_3bit`` / the C kernels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .q3bit import pack_3bit, unpack_3bit

__all__ = [
    "NF2_TABLE",
    "NF3_TABLE",
    "UNIFORM2_TABLE",
    "UNIFORM3_TABLE",
    "QuantLinearNF",
    "convert_linears_nf",
    "dequantize",
    "gemm",
    "pack_codes",
    "pack_state_dict",
    "packed_state_bytes",
    "quantize",
    "storage_bytes",
    "unpack_codes",
    "unpack_state_dict",
]

# Normal mid-quantiles, normalised so the extreme levels are exactly +-1. Computed with
# statistics.NormalDist and written down here so the codebook is stable across Python
# versions (the values are the whole format; a drifting table would silently change
# every checkpoint).
NF2_TABLE: tuple[float, ...] = (-1.0, -0.276993554654, 0.276993554654, 1.0)
NF3_TABLE: tuple[float, ...] = (
    -1.0,
    -0.578276956322,
    -0.318603654005,
    -0.102541280207,
    0.102541280207,
    0.318603654005,
    0.578276956322,
    1.0,
)
UNIFORM2_TABLE: tuple[float, ...] = (-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0)
UNIFORM3_TABLE: tuple[float, ...] = (
    -1.0,
    -5.0 / 7.0,
    -3.0 / 7.0,
    -1.0 / 7.0,
    1.0 / 7.0,
    3.0 / 7.0,
    5.0 / 7.0,
    1.0,
)
_TABLES = {
    (2, "nf"): NF2_TABLE,
    (3, "nf"): NF3_TABLE,
    (2, "uniform"): UNIFORM2_TABLE,
    (3, "uniform"): UNIFORM3_TABLE,
}


def _check_bits(bits: int) -> int:
    if bits not in (2, 3):
        raise ValueError("only 2- and 3-bit storage are supported")
    return int(bits)


def _block_align(bits: int) -> int:
    return 4 if bits == 2 else 8


def _table(bits: int, kind: str = "nf") -> torch.Tensor:
    if (bits, kind) not in _TABLES:
        raise ValueError("unknown codebook %r for %d bits" % (kind, bits))
    return torch.tensor(_TABLES[(bits, kind)], dtype=torch.float32)


def _resolve_table(bits: int, table) -> torch.Tensor:
    if table is None:
        return _table(bits, "nf")
    if isinstance(table, str):
        return _table(bits, table)
    t = torch.as_tensor(table, dtype=torch.float32).reshape(-1)
    if t.numel() != (1 << bits):
        raise ValueError(
            "codebook must have %d entries, got %d" % (1 << bits, t.numel())
        )
    if not bool(torch.all(t[1:] > t[:-1])):
        raise ValueError("codebook must be strictly increasing")
    return t


def _bytes_per_block(bits: int, block: int) -> int:
    return block // 4 if bits == 2 else block // 8 * 3


def _midpoints(table: torch.Tensor) -> torch.Tensor:
    return (table[:-1] + table[1:]) * 0.5


def pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack ``(nblocks, block)`` uint8 codes into bytes (first code in the low bits)."""
    _check_bits(bits)
    if codes.dim() != 2 or codes.shape[1] % _block_align(bits) != 0:
        raise ValueError(
            "codes must be (nblocks, block) with block aligned for %d bits" % bits
        )
    if codes.numel() and int(codes.max()) > (1 << bits) - 1:
        raise ValueError("code %d does not fit in %d bits" % (int(codes.max()), bits))
    if codes.numel() == 0:
        bpb = _bytes_per_block(bits, codes.shape[1])
        return torch.empty((codes.shape[0], bpb), dtype=torch.uint8)
    if bits == 3:
        return pack_3bit(codes)
    c = codes.view(codes.shape[0], codes.shape[1] // 4, 4).to(torch.int32)
    v = c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)
    return v.to(torch.uint8).reshape(codes.shape[0], -1)


def unpack_codes(packed: torch.Tensor, bits: int, block: int) -> torch.Tensor:
    """Inverse of :func:`pack_codes`; ``packed`` may be flat or per-row 2-D."""
    _check_bits(bits)
    if block % _block_align(bits) != 0:
        raise ValueError(
            "block must be a multiple of %d for %d-bit storage"
            % (_block_align(bits), bits)
        )
    bpb = _bytes_per_block(bits, block)
    if packed.numel() % bpb != 0:
        raise ValueError(
            "packed size %d is not a multiple of bytes-per-block %d"
            % (packed.numel(), bpb)
        )
    if bits == 3:
        return unpack_3bit(packed.reshape(-1, bpb), block)
    p = packed.reshape(-1, bpb)
    groups = block // 4
    nb = int(p.numel()) // groups
    v = p.reshape(nb, groups).to(torch.int32)
    return (
        torch.stack([(v >> (2 * i)) & 3 for i in range(4)], dim=-1)
        .reshape(nb, block)
        .to(torch.uint8)
    )


def quantize(
    w: torch.Tensor, bits: int = 2, block: int = 64, table=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blockwise NF (or custom) quantisation into ``(packed_uint8, absmax_float32)``.

    ``block`` must be a multiple of 4 (2-bit) or 8 (3-bit). A 2-D weight whose last
    dimension is block-aligned keeps the per-row layout expected by :func:`gemm`.
    """
    bits = _check_bits(bits)
    if block <= 0 or block % _block_align(bits) != 0:
        raise ValueError(
            "block must be a positive multiple of %d for %d-bit storage"
            % (_block_align(bits), bits)
        )
    t = _resolve_table(bits, table)
    flat = w.detach().float().reshape(-1).contiguous()
    n = flat.numel()
    pad = (-n) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blk = flat.view(-1, block)
    absmax = blk.abs().amax(dim=1)
    absmax = torch.where(absmax == 0, torch.ones_like(absmax), absmax)
    x = blk / absmax.unsqueeze(1)
    idx = torch.bucketize(x, _midpoints(t)).clamp_(0, t.numel() - 1).to(torch.uint8)
    packed = pack_codes(idx, bits)
    absmax = absmax.float()
    if w.dim() == 2 and w.shape[1] % block == 0:
        nb_per_row = w.shape[1] // block
        bpb = packed.shape[1]
        packed = packed.reshape(w.shape[0], nb_per_row * bpb)
        absmax = absmax.reshape(w.shape[0], nb_per_row)
    return packed, absmax


def _numel(shape) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def dequantize(
    packed: torch.Tensor,
    absmax: torch.Tensor,
    shape,
    bits: int = 2,
    block: int = 64,
    table=None,
) -> torch.Tensor:
    """Unpack packed codes and apply the per-block scales, returning ``shape``."""
    bits = _check_bits(bits)
    t = _resolve_table(bits, table)
    n = _numel(shape)
    a = absmax.reshape(-1).float()
    codes = unpack_codes(packed, bits, block)
    if codes.shape[0] != a.numel():
        raise ValueError(
            "absmax count %d does not match packed blocks %d"
            % (a.numel(), codes.shape[0])
        )
    vals = t[codes.long()] * a.unsqueeze(1)
    return vals.reshape(-1)[:n].reshape(tuple(shape))


def _check_gemm(
    x: torch.Tensor, packed: torch.Tensor, absmax: torch.Tensor, bits: int, block: int
) -> tuple[int, int]:
    if x.dim() != 2:
        raise ValueError("x must be 2-D (M, K)")
    if packed.dim() != 2:
        raise ValueError("packed weight must be 2-D")
    bpb = _bytes_per_block(bits, block)
    if packed.shape[1] % bpb != 0:
        raise ValueError(
            "packed row width %d is not a multiple of bytes-per-block %d"
            % (packed.shape[1], bpb)
        )
    nb_per_row = packed.shape[1] // bpb
    K = nb_per_row * block
    if x.shape[1] != K:
        raise ValueError("x.shape[1] (%d) != K (%d)" % (x.shape[1], K))
    if int(absmax.numel()) != packed.shape[0] * nb_per_row:
        raise ValueError("absmax count does not match packed blocks")
    return packed.shape[0], K


def gemm(
    x: torch.Tensor,
    packed: torch.Tensor,
    absmax: torch.Tensor,
    bits: int = 2,
    block: int = 64,
    table=None,
) -> torch.Tensor:
    """``x @ dequant(packed).T`` — memory win only (no native 2/3-bit GEMM)."""
    N, K = _check_gemm(x, packed, absmax, bits, block)
    w = dequantize(packed, absmax, (N, K), bits=bits, block=block, table=table)
    return x.float() @ w.t()


def storage_bytes(packed: torch.Tensor, absmax: torch.Tensor) -> int:
    """Payload bytes (codes + fp32 scales), excluding any dequantisation temporary."""
    return int(packed.numel()) + int(absmax.numel()) * 4


def pack_state_dict(
    state_dict,
    bits: int = 2,
    block: int = 64,
    table=None,
    min_numel: int = 4096,
    exclude_prefixes=(),
):
    """Quantise every 2-D float tensor of a state dict into packed NF storage.

    Returns a plain dict that ``torch.save`` can write: quantised entries are
    ``{"kind": "nfbit", "packed", "absmax", "shape"}``; everything else (1-D biases,
    norms, non-float tensors) is copied verbatim. ``__nfbit_meta__`` records the format
    so :func:`unpack_state_dict` needs no extra arguments. This is a *storage* format:
    keep the fp32 weights and use QAT+STE if you need to train a low-bit base.
    """
    bits = _check_bits(bits)
    t = _resolve_table(bits, table)
    meta = {
        "version": 1,
        "bits": bits,
        "block": int(block),
        "table": "custom"
        if not isinstance(table, (str, type(None)))
        else (table or "nf"),
    }
    if meta["table"] == "custom":
        meta["table_values"] = t.tolist()
    excluded = tuple(exclude_prefixes)
    out = {"__nfbit_meta__": meta}
    for k, v in state_dict.items():
        if excluded and k.startswith(excluded):
            out[k] = v
        elif (
            torch.is_tensor(v)
            and v.dtype == torch.float32
            and v.ndim >= 2
            and v.numel() >= int(min_numel)
        ):
            packed, absmax = quantize(v, bits=bits, block=block, table=t)
            out[k] = {
                "kind": "nfbit",
                "packed": packed,
                "absmax": absmax,
                "shape": tuple(v.shape),
            }
        else:
            out[k] = v
    return out


def unpack_state_dict(packed_state_dict):
    """Inverse of :func:`pack_state_dict`; returns fp32 tensors (dequantised)."""
    sd = dict(packed_state_dict)
    meta = sd.pop("__nfbit_meta__", None)
    if not meta:
        raise ValueError("not an nfbit-packed state dict (missing __nfbit_meta__)")
    bits = _check_bits(int(meta["bits"]))
    block = int(meta["block"])
    kind = meta.get("table", "nf")
    table = meta.get("table_values") if kind == "custom" else kind
    t = _resolve_table(bits, table)
    out = {}
    for k, v in sd.items():
        if isinstance(v, dict) and v.get("kind") == "nfbit":
            out[k] = dequantize(
                v["packed"], v["absmax"], v["shape"], bits=bits, block=block, table=t
            )
        else:
            out[k] = v
    return out


def packed_state_bytes(packed_state_dict) -> int:
    """Bytes held by a packed state dict (codes + scales + untouched tensors)."""
    total = 0
    for k, v in packed_state_dict.items():
        if k == "__nfbit_meta__":
            continue
        if isinstance(v, dict) and v.get("kind") == "nfbit":
            total += storage_bytes(v["packed"], v["absmax"])
        elif torch.is_tensor(v):
            total += v.numel() * v.element_size()
    return int(total)


class QuantLinearNF(nn.Module):
    """A frozen NF-coded ``nn.Linear`` with trainable LoRA adapters.

    The base weight lives as packed NF codes plus one fp32 scale per ``block``
    (2.5 bit/weight for 2-bit at block=64); only ``lora_A``/``lora_B`` train. Forward
    dequantises into an fp32 temporary (there is no fused 2/3-bit GEMM), so this is a
    storage/checkpoint win rather than a speed win — see :meth:`forward_peak_bytes`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 2,
        block: int = 64,
        lora_r: int = 8,
        lora_alpha: int = 16,
        table=None,
        dtype: torch.dtype = torch.float32,
        device=None,
    ):
        super().__init__()
        bits = _check_bits(bits)
        if block <= 0 or block % _block_align(bits) != 0:
            raise ValueError(
                "block must be a positive multiple of %d for %d-bit storage"
                % (_block_align(bits), bits)
            )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bits = bits
        self.block = int(block)
        self.table_kind = table if isinstance(table, str) or table is None else "custom"
        self.register_buffer("packed", None)
        self.register_buffer("absmax", None)
        if self.table_kind == "custom":
            self.register_buffer("codebook", _resolve_table(bits, table))
        else:
            self.codebook = None
        self.register_buffer("bias", None)
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(self.out_features, dtype=dtype, device=device),
                requires_grad=False,
            )
        self.lora_A = nn.Parameter(
            torch.empty(lora_r, self.in_features, dtype=dtype, device=device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, lora_r, dtype=dtype, device=device)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.scale = lora_alpha / lora_r

    def _table(self) -> torch.Tensor:
        if self.table_kind == "custom":
            return self.codebook
        return _table(self.bits, "nf" if self.table_kind is None else self.table_kind)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 2,
        block: int = 64,
        lora_r: int = 8,
        lora_alpha: int = 16,
        table=None,
    ) -> QuantLinearNF:
        m = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            bits=bits,
            block=block,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            table=table,
            dtype=linear.weight.dtype,
            device=linear.weight.device,
        )
        m.load_from_linear(linear)
        return m

    def load_from_linear(self, linear: nn.Linear) -> None:
        packed, absmax = quantize(
            linear.weight.detach(),
            bits=self.bits,
            block=self.block,
            table=self._table(),
        )
        self.packed = packed
        self.absmax = absmax
        if linear.bias is not None and self.bias is not None:
            with torch.no_grad():
                self.bias.copy_(linear.bias.detach())

    def weight(self) -> torch.Tensor:
        if self.packed is None:
            raise RuntimeError("QuantLinearNF has no packed weight; use from_linear()")
        return dequantize(
            self.packed,
            self.absmax,
            (self.out_features, self.in_features),
            bits=self.bits,
            block=self.block,
            table=self._table(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.packed is None:
            raise RuntimeError("QuantLinearNF has no packed weight; use from_linear()")
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        bias = None if self.bias is None else self.bias.to(x.dtype)
        base = F.linear(x2, self.weight().to(x.dtype), bias)
        lora = (
            (x2 @ self.lora_A.to(x2.dtype).t())
            @ self.lora_B.to(x2.dtype).t()
            * self.scale
        )
        return (base + lora).to(x.dtype).reshape(*lead, self.out_features)

    def storage_bytes(self) -> int:
        """Bytes actually held: packed codes + scales + LoRA + bias."""
        n = storage_bytes(self.packed, self.absmax) if self.packed is not None else 0
        n += self.lora_A.numel() * self.lora_A.element_size()
        n += self.lora_B.numel() * self.lora_B.element_size()
        if self.bias is not None:
            n += self.bias.numel() * self.bias.element_size()
        return int(n)

    def forward_peak_bytes(self, batch: int = 1) -> int:
        """Storage plus the fp32 weight temporary and activations of one forward."""
        temp = self.out_features * self.in_features * 4
        act = batch * (self.in_features + self.out_features) * 4
        return int(self.storage_bytes() + temp + act)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load a checkpoint, including the lazily-created ``packed``/``absmax`` buffers.

        ``packed``/``absmax`` are registered as ``None`` until ``from_linear`` fills
        them, so the base loader would report them as unexpected keys. They are popped
        and assigned here; the rest (bias / LoRA / custom codebook) goes through the
        normal loader.
        """
        sd = dict(state_dict)
        packed = sd.pop("packed", None)
        absmax = sd.pop("absmax", None)
        result = super().load_state_dict(sd, strict=strict, assign=assign)
        if packed is not None:
            self.packed = packed
        if absmax is not None:
            self.absmax = absmax
        return result

    def extra_repr(self) -> str:
        return (
            "in_features=%d, out_features=%d, bits=%d, block=%d, table=%s, lora_r=%d"
            % (
                self.in_features,
                self.out_features,
                self.bits,
                self.block,
                self.table_kind,
                self.lora_A.shape[0],
            )
        )


def convert_linears_nf(
    model: nn.Module,
    bits: int = 2,
    block: int = 64,
    lora_r: int = 8,
    lora_alpha: int = 16,
    table=None,
    skip_names: tuple[str, ...] = ("lm_head",),
    name_prefix: str = "",
) -> int:
    """Replace ``nn.Linear`` children with :class:`QuantLinearNF` (in place)."""
    n = 0
    for name, child in list(model.named_children()):
        full = f"{name_prefix}.{name}" if name_prefix else name
        if isinstance(child, nn.Linear):
            if name in skip_names:
                continue
            setattr(
                model,
                name,
                QuantLinearNF.from_linear(
                    child,
                    bits=bits,
                    block=block,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    table=table,
                ),
            )
            n += 1
        else:
            n += convert_linears_nf(
                child,
                bits=bits,
                block=block,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                table=table,
                skip_names=skip_names,
                name_prefix=full,
            )
    return n
