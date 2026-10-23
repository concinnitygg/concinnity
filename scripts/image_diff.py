"""Score PNG captures pairwise: for every file name two directories share,
print the whole-image RMSE (0..1), the pixels differing by more than two
levels in any channel, and the largest difference.

    image_diff.py DIR_A DIR_B [DIR_A2 DIR_B2 ...]

Standard library only, so it runs on any host; numpy speeds it up when it is
importable.
"""

import os
import struct
import sys
import zlib


def read_png(path):
    """(width, height, channels, bytes) of an 8-bit non-interlaced PNG."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path}: not a PNG")
    pos, idat, header = 8, [], None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat.append(body)
        pos += 12 + length
    width, height, depth, color, _, _, interlace = header
    if depth != 8 or interlace != 0:
        raise ValueError(f"{path}: only 8-bit non-interlaced PNGs")
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color]
    raw = zlib.decompress(b"".join(idat))
    return width, height, channels, unfilter(raw, width, height, channels)


def unfilter(raw, width, height, bpp):
    stride = width * bpp
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError:
        np = None
    out = bytearray(stride * height)
    prev = bytearray(stride)
    for y in range(height):
        kind = raw[y * (stride + 1)]
        line = bytearray(raw[y * (stride + 1) + 1 : (y + 1) * (stride + 1)])
        if kind == 1:
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 0xFF
        elif kind == 2:
            if np is not None:
                line = bytearray((np.frombuffer(line, np.uint8) + np.frombuffer(prev, np.uint8)).tobytes())
            else:
                for i in range(stride):
                    line[i] = (line[i] + prev[i]) & 0xFF
        elif kind == 3:
            for i in range(stride):
                left = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif kind == 4:
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                line[i] = (line[i] + pred) & 0xFF
        out[y * stride : (y + 1) * stride] = line
        prev = line
    return bytes(out)


def score(a_path, b_path):
    wa, ha, ca, a = read_png(a_path)
    wb, hb, cb, b = read_png(b_path)
    if (wa, ha, ca) != (wb, hb, cb):
        return None
    color = min(ca, 3)
    try:
        import numpy as np  # noqa: PLC0415

        x = np.frombuffer(a, np.uint8).reshape(-1, ca)[:, :color].astype(np.int32)
        y = np.frombuffer(b, np.uint8).reshape(-1, cb)[:, :color].astype(np.int32)
        d = np.abs(x - y)
        rmse = float(np.sqrt(np.mean((d / 255.0) ** 2)))
        return rmse, int(np.count_nonzero(d.max(axis=1) > 2)), int(d.max())
    except ImportError:
        sq, over, most = 0, 0, 0
        for p in range(0, len(a), ca):
            worst = 0
            for c in range(color):
                diff = abs(a[p + c] - b[p + c])
                sq += diff * diff
                worst = max(worst, diff)
            over += worst > 2
            most = max(most, worst)
        n = (len(a) // ca) * color
        return (sq / n) ** 0.5 / 255.0, over, most


def main(args):
    if len(args) < 2 or len(args) % 2:
        sys.exit(__doc__)
    for a_dir, b_dir in zip(args[::2], args[1::2]):
        print(f"== {a_dir} -> {b_dir}")
        names = sorted(set(os.listdir(a_dir)) & set(os.listdir(b_dir)))
        for name in (n for n in names if n.endswith(".png")):
            result = score(os.path.join(a_dir, name), os.path.join(b_dir, name))
            if result is None:
                print(f"{name:24s} size differs")
                continue
            rmse, over, most = result
            print(f"{name:24s} rmse {rmse:.4f} px>2 {over:8d} max {most:3d}")


if __name__ == "__main__":
    main(sys.argv[1:])
