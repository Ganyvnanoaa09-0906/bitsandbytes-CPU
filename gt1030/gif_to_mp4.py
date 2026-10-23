"""Convert animated GIF frames to H.264 mp4 with PyAV."""
from __future__ import annotations

import sys

import av
from PIL import Image, ImageSequence

src, dst = sys.argv[1], sys.argv[2]
frames = [f.convert('RGB') for f in ImageSequence.Iterator(Image.open(src))]
w, h = frames[0].size
c = av.open(dst, 'w')
s = c.add_stream('libx264', rate=8)
s.width, s.height, s.pix_fmt = w, h, 'yuv420p'
for im in frames:
    frame = av.VideoFrame.from_image(im)
    for packet in s.encode(frame):
        c.mux(packet)
for packet in s.encode(None):
    c.mux(packet)
c.close()
print(dst, len(frames))
