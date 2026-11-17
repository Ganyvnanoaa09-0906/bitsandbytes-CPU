"""Tests for the NF2/NF3 packed low-bit storage format (no native kernels needed)."""

import pytest
import torch
from torch import nn

from bitsandbytes import nfbit


def _rel_err(a, b):
    return float((a - b).norm()) / (float(b.norm()) or 1.0)


def test_codebooks():
    assert nfbit.NF2_TABLE == (-1.0, -0.276993554654, 0.276993554654, 1.0)
    for table in (
        nfbit.NF2_TABLE,
        nfbit.NF3_TABLE,
        nfbit.UNIFORM2_TABLE,
        nfbit.UNIFORM3_TABLE,
    ):
        assert len(table) in (4, 8)
        assert all(table[i] < table[i + 1] for i in range(len(table) - 1))
        assert table[0] == -1.0 and table[-1] == 1.0
    assert len(nfbit.NF3_TABLE) == 8


def test_pack_bit_order_2bit():
    # byte 0b11_10_01_00 -> codes 0,1,2,3 (first code in the low bits)
    codes = torch.tensor([[0, 1, 2, 3]], dtype=torch.uint8)
    packed = nfbit.pack_codes(codes, bits=2)
    assert packed.tolist() == [[0b11100100]]
    back = nfbit.unpack_codes(packed, bits=2, block=4)
    assert back.tolist() == [[0, 1, 2, 3]]


def test_pack_roundtrip_3bit():
    g = torch.Generator().manual_seed(0)
    codes = torch.randint(0, 8, (5, 64), generator=g, dtype=torch.uint8)
    packed = nfbit.pack_codes(codes, bits=3)
    assert packed.shape == (5, 24)
    assert nfbit.unpack_codes(packed, bits=3, block=64).equal(codes)


def test_nf_beats_uniform_at_2bit():
    g = torch.Generator().manual_seed(1)
    w = torch.randn(256, 256, generator=g)
    nf_p, nf_a = nfbit.quantize(w, bits=2, block=64)
    un_p, un_a = nfbit.quantize(w, bits=2, block=64, table="uniform")
    nf_err = _rel_err(nfbit.dequantize(nf_p, nf_a, w.shape, bits=2, block=64), w)
    un_err = _rel_err(
        nfbit.dequantize(un_p, un_a, w.shape, bits=2, block=64, table="uniform"), w
    )
    assert nf_err < un_err, (nf_err, un_err)
    assert nf_err < 0.55


def test_3bit_error_below_2bit():
    g = torch.Generator().manual_seed(2)
    w = torch.randn(128, 256, generator=g)
    e2 = _rel_err(nfbit.dequantize(*nfbit.quantize(w, bits=2, block=64), w.shape), w)
    e3 = _rel_err(
        nfbit.dequantize(
            *nfbit.quantize(w, bits=3, block=64), w.shape, bits=3, block=64
        ),
        w,
    )
    assert e3 < e2


def test_storage_size():
    w = torch.randn(128, 256)
    packed, absmax = nfbit.quantize(w, bits=2, block=64)
    bpp = nfbit.storage_bytes(packed, absmax) / w.numel()
    assert abs(bpp - (2 + 32.0 / 64) / 8.0) < 1e-6
    packed3, absmax3 = nfbit.quantize(w, bits=3, block=64)
    bpp3 = nfbit.storage_bytes(packed3, absmax3) / w.numel()
    assert abs(bpp3 - (3 + 32.0 / 64) / 8.0) < 1e-6


def test_gemm_matches_dequant():
    g = torch.Generator().manual_seed(3)
    w = torch.randn(64, 128, generator=g)
    x = torch.randn(7, 128, generator=g)
    packed, absmax = nfbit.quantize(w, bits=2, block=64)
    out = nfbit.gemm(x, packed, absmax, bits=2, block=64)
    ref = x @ nfbit.dequantize(packed, absmax, w.shape, bits=2, block=64).t()
    assert torch.allclose(out, ref, atol=1e-6)


def test_quant_linear_forward_grad_and_save(tmp_path):
    torch.manual_seed(4)
    lin = nn.Linear(128, 96)
    ql = nfbit.QuantLinearNF.from_linear(lin, bits=2, block=64, lora_r=4)
    x = torch.randn(3, 128)
    # lora_B is zero-initialised: forward must equal the quantised frozen base
    y = ql(x)
    ref = nn.functional.linear(
        x, nfbit.dequantize(ql.packed, ql.absmax, (96, 128), bits=2, block=64), lin.bias
    )
    assert torch.allclose(y, ref, atol=1e-5)

    loss = y.square().sum()
    loss.backward()
    assert ql.lora_A.grad is not None  # zero while lora_B is zero-initialised
    assert ql.lora_B.grad is not None and ql.lora_B.grad.abs().sum() > 0
    assert ql.packed.grad is None and ql.absmax.grad is None
    assert ql.bias is not None and ql.bias.requires_grad is False

    # storage: 2.5 bit/weight payload + small LoRA, far below fp32
    payload = nfbit.storage_bytes(ql.packed, ql.absmax)
    assert payload < 0.4 * 96 * 128
    assert ql.storage_bytes() > payload
    assert ql.forward_peak_bytes(batch=3) > ql.storage_bytes()

    # checkpoint round-trip must be bit-identical
    p = tmp_path / "nf.pt"
    torch.save(ql.state_dict(), p)
    ql2 = nfbit.QuantLinearNF(128, 96, bias=True, bits=2, block=64, lora_r=4)
    ql2.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
    assert torch.equal(ql(x), ql2(x))
    assert torch.equal(ql.packed, ql2.packed)


def test_convert_nested_and_skip():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(32, 32)
            self.inner = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8))
            self.lm_head = nn.Linear(8, 4)

        def forward(self, x):
            return self.lm_head(self.inner(self.a(x)))

    m = M()
    n = nfbit.convert_linears_nf(m, bits=2, block=32, lora_r=2)
    assert n == 3
    assert isinstance(m.a, nfbit.QuantLinearNF)
    assert isinstance(m.inner[0], nfbit.QuantLinearNF)
    assert isinstance(m.lm_head, nn.Linear)
    assert m(torch.randn(2, 32)).shape == (2, 4)


def test_invalid_args():
    with pytest.raises(ValueError):
        nfbit.quantize(torch.randn(8, 8), bits=4)
    with pytest.raises(ValueError):
        nfbit.quantize(torch.randn(8, 8), bits=2, block=6)
    with pytest.raises(ValueError):
        nfbit.gemm(
            torch.randn(2, 63),
            *nfbit.quantize(torch.randn(8, 64), bits=2),
            bits=2,
            block=64,
        )


def test_state_dict_pack_roundtrip(tmp_path):
    torch.manual_seed(7)
    sd = {
        "w": torch.randn(96, 128),
        "b": torch.randn(96),
        "emb": torch.randn(64, 96),
        "int": torch.arange(8, dtype=torch.int64),
    }
    packed = nfbit.pack_state_dict(sd, bits=2, block=32, min_numel=1024)
    assert packed["__nfbit_meta__"]["bits"] == 2
    assert packed["w"]["kind"] == "nfbit" and packed["b"] is sd["b"]
    assert packed["emb"]["kind"] == "nfbit"
    f = tmp_path / "model.nf2"
    torch.save(packed, f)
    loaded = nfbit.unpack_state_dict(
        torch.load(f, map_location="cpu", weights_only=False)
    )
    ref = nfbit.dequantize(
        *nfbit.quantize(sd["w"], bits=2, block=32), sd["w"].shape, bits=2, block=32
    )
    assert torch.equal(loaded["w"], ref)
    assert torch.equal(loaded["b"], sd["b"])
    assert torch.equal(loaded["int"], sd["int"])
    raw = sum(v.numel() * v.element_size() for v in sd.values())
    assert nfbit.packed_state_bytes(packed) < 0.45 * raw
