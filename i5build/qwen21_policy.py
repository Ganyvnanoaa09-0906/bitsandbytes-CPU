"""qwen21_policy.py -- the single source of truth for what gets quantized.

Why this module exists
----------------------
The quantizer decides which parameters to convert; the loader decides which module
to rebuild for each converted parameter. Written independently they drift, and the
drift is silent: the quantizer happily converts a tensor the loader has no case for,
the checkpoint writes fine, and the failure surfaces much later as a missing key or
a shape mismatch during load.

That is not hypothetical here. The first version of the rule was "2-D parameter
whose name ends in .weight", which sounds like "a Linear weight" and is almost
true -- every nn.Linear in QwenImage21Transformer2DModel matches it. But the
text encoder's token embedding is also 2-D and also named .weight
(model.language_model.embed_tokens.weight, [151936, 4096], the single largest
tensor in the whole checkpoint at 1.16 GB). It is an nn.Embedding, so a loader that
only handles nn.Linear would build the wrong module for it.

Both sides import from here, and the loader asserts that its replacement count
agrees with the component's metadata, so a divergence is loud.

Verified against the vendored source rather than assumed:
  transformer_qwenimage21.py: img_in, to_q/to_k/to_v/to_out.0, img_mlp.{proj,out,
    gate_layer}, modulation.1, timestep_embedder.linear_{1,2},
    txt_in.{in_layer,out_layer}, proj_out are all nn.Linear(..., bias=False)
  modeling_qwen3_vl.py:755  self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
"""

# Parameter-key suffix for a Linear weight.
WEIGHT_SUFFIX = ".weight"

# Module-path fragments that identify an embedding rather than a linear layer.
EMBEDDING_MARKERS = ("embed_tokens",)

QUANTIZED_KIND_LINEAR = "linear"
QUANTIZED_KIND_EMBEDDING = "embedding"


def is_quantizable(name, tensor):
    """True when this parameter is a weight bnb can quantize.

    bnb quantizes nn.Linear and nn.Embedding. Both store their weight as a 2-D
    parameter named *.weight, and in this model nothing else is 2-D, so the shape
    and name together identify the set exactly.
    """
    return tensor.dim() == 2 and name.endswith(WEIGHT_SUFFIX)


def kind_of(name):
    """Which quantized module class a converted parameter belongs to.

    Recorded in the checkpoint metadata so the loader does not have to re-derive it
    from the key name -- the metadata is the record of what the quantizer actually
    did, and inferring it again at load time is how the two sides drift apart.
    """
    if any(m in name for m in EMBEDDING_MARKERS):
        return QUANTIZED_KIND_EMBEDDING
    return QUANTIZED_KIND_LINEAR


def quantized_module_for(kind):
    """(module_class, name) from bitsandbytes for a quantized kind.

    Imported lazily so this module stays importable without bitsandbytes present,
    which matters because it is also read by inspection scripts.
    """
    import bitsandbytes as bnb

    if kind == QUANTIZED_KIND_EMBEDDING:
        return bnb.nn.Embedding4bit, "Embedding4bit"
    if kind == QUANTIZED_KIND_LINEAR:
        return bnb.nn.Linear4bit, "Linear4bit"
    raise ValueError(f"unknown quantized kind {kind!r}")
