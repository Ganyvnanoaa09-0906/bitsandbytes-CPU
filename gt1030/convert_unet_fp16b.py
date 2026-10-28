"""Convert sd15 fp32 UNet to all-fp16 internals with explicit float32 IO Cast nodes."""
from __future__ import annotations

import copy
import os
import shutil

import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxconverter_common import float16

SRC = r'D:\work\sd15_onnx'
DST = r'D:\work\sd15_onnx_fp16b'


def main():
    m = onnx.load(os.path.join(SRC, 'unet', 'model.onnx'), load_external_data=True)
    m16 = float16.convert_float_to_float16(m, keep_io_types=False, disable_shape_infer=True)
    g = m16.graph
    # inputs: float32 graph input -> Cast -> fp16 internal
    for inp in list(g.input):
        old = inp.name
        inp.type.tensor_type.elem_type = TensorProto.FLOAT
        cast_out = old + '_f16'
        for n in g.node:
            for i, name in enumerate(n.input):
                if name == old:
                    n.input[i] = cast_out
        g.node.insert(0, helper.make_node('Cast', [old], [cast_out], to=TensorProto.FLOAT16,
                                          name=old + '_cast'))
    # output: fp16 internal -> Cast -> float32 graph output
    for out in list(g.output):
        old = out.name
        f16_name = old + '_f16'
        for n in g.node:
            for i, name in enumerate(n.output):
                if name == old:
                    n.output[i] = f16_name
        out.type.tensor_type.elem_type = TensorProto.FLOAT
        g.node.append(helper.make_node('Cast', [f16_name], [old], to=TensorProto.FLOAT,
                                       name=old + '_cast_out'))

    # convert any float32 Constant attribute tensors left by the converter
    import numpy as np
    from onnx import numpy_helper as _nh
    nfix = 0
    for n in g.node:
        for a in n.attribute:
            if a.type == onnx.AttributeProto.TENSOR and a.t.data_type == TensorProto.FLOAT:
                arr = _nh.to_array(a.t).astype(np.float16)
                a.t.CopyFrom(_nh.from_array(arr, a.t.name))
                nfix += 1
    print('converted constant tensors', nfix, flush=True)

    # keep only IO casts float32; internal Cast-to-float32 -> fp16
    io_cast_names = {'sample_cast', 'timestep_cast', 'encoder_hidden_states_cast', 'out_sample_cast_out'}
    ncast = 0
    for n in g.node:
        if n.op_type == 'Cast' and n.name not in io_cast_names:
            for a in n.attribute:
                if a.name == 'to' and a.i == TensorProto.FLOAT:
                    a.i = TensorProto.FLOAT16
                    ncast += 1
    print('internal Cast->fp16', ncast, flush=True)

    # Resize requires float32 roi/scales: convert producers back
    init_by_name = {i.name: i for i in g.initializer}
    node_by_out = {}
    for n in g.node:
        for o in n.output:
            node_by_out[o] = n
    nres = 0
    for n in g.node:
        if n.op_type != 'Resize':
            continue
        for idx in (1, 2):
            if idx >= len(n.input) or not n.input[idx]:
                continue
            nm = n.input[idx]
            if nm in init_by_name:
                ini = init_by_name[nm]
                arr = _nh.to_array(ini).astype(np.float32)
                ini.CopyFrom(_nh.from_array(arr, ini.name))
                nres += 1
            elif nm in node_by_out and node_by_out[nm].op_type == 'Constant':
                for a in node_by_out[nm].attribute:
                    if a.name == 'value' and a.t.data_type == TensorProto.FLOAT16:
                        arr = _nh.to_array(a.t).astype(np.float32)
                        a.t.CopyFrom(_nh.from_array(arr, a.t.name))
                        nres += 1
    print('resize float32 restored', nres, flush=True)
    outp = os.path.join(DST, 'unet', 'model.onnx')
    onnx.save(m16, outp, save_as_external_data=True, all_tensors_to_one_file=True,
              location='model.onnx_data', size_threshold=1024)
    print('saved', outp, flush=True)


if __name__ == '__main__':
    main()
