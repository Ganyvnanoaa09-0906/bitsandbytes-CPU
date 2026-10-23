"""Convert fp32 SD1.5 UNet ONNX to fp16 (keep float32 IO) for DML 512 test.

Creates D:/work/sd15_onnx_fp16b with fp32 text encoder + VAE and fp16 UNet.
"""
from __future__ import annotations

import os
import shutil

import onnx
from onnxconverter_common import float16

SRC = r'D:\work\sd15_onnx'
DST = r'D:\work\sd15_onnx_fp16b'


def copy_component(name):
    src = os.path.join(SRC, name); dst = os.path.join(DST, name)
    if os.path.isdir(src):
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def main():
    os.makedirs(DST, exist_ok=True)
    for f in ('model_index.json',):
        shutil.copy2(os.path.join(SRC, f), os.path.join(DST, f))
    for c in ('scheduler', 'tokenizer', 'text_encoder', 'vae_decoder', 'vae_encoder'):
        copy_component(c)
    os.makedirs(os.path.join(DST, 'unet'), exist_ok=True)
    for f in ('config.json',):
        shutil.copy2(os.path.join(SRC, 'unet', f), os.path.join(DST, 'unet', f))
    print('loading fp32 unet', flush=True)
    m = onnx.load(os.path.join(SRC, 'unet', 'model.onnx'), load_external_data=True)
    print('converting to fp16 keep_io_types', flush=True)
    m16 = float16.convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=False)
    out = os.path.join(DST, 'unet', 'model.onnx')
    print('saving', out, flush=True)
    onnx.save(m16, out, save_as_external_data=True, all_tensors_to_one_file=True,
              location='model.onnx_data', size_threshold=1024)
    print('done', flush=True)


if __name__ == '__main__':
    main()
