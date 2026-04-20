#!/usr/bin/env python

import numpy as np
from PIL import Image
from imagehash import ImageHash
from numba import cuda
import math

from FINd import FINDHasher


@cuda.jit
def _boxFilter_cuda(input, output, rows, cols, halfRowWin, halfColWin):
    """Each CUDA thread computes one output pixel's box filter value."""
    i, j = cuda.grid(2)
    if i >= rows or j >= cols:
        return
    xmin = max(0, i - halfRowWin)
    xmax = min(rows, i + halfRowWin)
    ymin = max(0, j - halfColWin)
    ymax = min(cols, j + halfColWin)
    s = 0.0
    for k in range(xmin, xmax):
        for l in range(ymin, ymax):
            s += input[k * cols + l]
    output[i * cols + j] = s / ((xmax - xmin) * (ymax - ymin))


class FINDHasherGPU(FINDHasher):
    """
    GPU-accelerated FINd hasher using NVIDIA CUDA via Numba.

    The box filter kernel runs entirely on GPU — each CUDA thread computes
    one output pixel, giving massive parallelism over the ~262K pixel grid.
    Luma, decimation, DCT and hash remain in numpy (CPU) as they are small
    enough that GPU transfer overhead would dominate.
    """

    _DCT_np = None

    def _get_dct_np(self):
        if FINDHasherGPU._DCT_np is None:
            FINDHasherGPU._DCT_np = np.array(self.DCT_matrix)
        return FINDHasherGPU._DCT_np

    def fromFile(self, filepath):
        img = Image.open(filepath)
        return self._fromImage(img)

    def fromImage(self, img):
        return self._fromImage(img.copy())

    def _fromImage(self, img):
        if max(img.size) > 512:
            img.thumbnail((512, 512))
        numCols, numRows = img.size

        # Luma via numpy
        if img.mode != "RGB":
            img = img.convert("RGB")
        rgb = np.array(img, dtype=np.float64)
        luma = (
            self.LUMA_FROM_R_COEFF * rgb[:, :, 0]
            + self.LUMA_FROM_G_COEFF * rgb[:, :, 1]
            + self.LUMA_FROM_B_COEFF * rgb[:, :, 2]
        ).ravel()

        # Box filter on GPU
        windowSizeAlongRows = self.computeBoxFilterWindowSize(numCols)
        windowSizeAlongCols = self.computeBoxFilterWindowSize(numRows)
        halfRowWin = (windowSizeAlongRows + 2) // 2
        halfColWin = (windowSizeAlongCols + 2) // 2

        d_input = cuda.to_device(luma)
        d_output = cuda.device_array_like(d_input)

        threads = (16, 16)
        blocks = (math.ceil(numRows / threads[0]), math.ceil(numCols / threads[1]))
        _boxFilter_cuda[blocks, threads](d_input, d_output, numRows, numCols,
                                         halfRowWin, halfColWin)
        filtered_2d = d_output.copy_to_host().reshape((numRows, numCols))

        # Decimate via numpy fancy indexing
        i_idx = ((np.arange(64) + 0.5) * numRows / 64).astype(int)
        j_idx = ((np.arange(64) + 0.5) * numCols / 64).astype(int)
        dec64 = filtered_2d[np.ix_(i_idx, j_idx)]

        # DCT via numpy matmul
        D = self._get_dct_np()
        dct16 = D @ dec64 @ D.T

        # Hash via vectorised median comparison
        median = np.median(dct16)
        bits = (dct16 > median).astype(int)
        flipped = np.flip(bits, axis=(0, 1))
        return ImageHash(flipped.reshape(256))
