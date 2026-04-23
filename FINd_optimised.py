#!/usr/bin/env python

import numpy as np
from PIL import Image
from imagehash import ImageHash
from multiprocessing import Pool
import os

from numba import njit, prange

from FINd import FINDHasher


@njit(parallel=True, fastmath=True, cache=True)
def _boxFilter_numba(input, output, rows, cols, rowWin, colWin):
    halfColWin = (colWin + 2) // 2
    halfRowWin = (rowWin + 2) // 2
    for i in prange(rows):
        xmin = max(0, i - halfRowWin)
        xmax = min(rows, i + halfRowWin)
        for j in range(cols):
            ymin = max(0, j - halfColWin)
            ymax = min(cols, j + halfColWin)
            s = 0.0
            for k in range(xmin, xmax):
                for l in range(ymin, ymax):
                    s += input[k * cols + l]
            output[i * cols + j] = s / ((xmax - xmin) * (ymax - ymin))


def _hash_one(path):
    hasher = FINDHasherOptimised()
    return (path, str(hasher.fromFile(path)))


class FINDHasherOptimised(FINDHasher):
    """
    Optimised version of FINDHasher with four targeted improvements:

    1. Luma conversion: numpy vectorised array op replaces per-pixel getpixel() loop.

    2. boxFilter: Numba @njit(parallel=True, fastmath=True) compiles the nested
       loops to native machine code and parallelises rows across CPU cores via
       prange. Preserves exact boundary behaviour. Bug fix: original used k*rows
       instead of k*cols for row-major indexing.

    3. Full numpy pipeline in fromImage: decimation, DCT, and hash generation all
       stay in numpy arrays — no copy-back loops into MatrixUtil list-of-lists.
       - Decimation via fancy indexing (one C-level read).
       - DCT via numpy matmul (BLAS): D @ A @ D.T.
       - Hash via vectorised comparison against median (no Python loops).

    4. fromFiles: parallel hashing across images via multiprocessing.Pool.
    """

    _DCT_np = None

    def _get_dct_np(self):
        if FINDHasherOptimised._DCT_np is None:
            FINDHasherOptimised._DCT_np = np.array(self.DCT_matrix)
        return FINDHasherOptimised._DCT_np

    def fromFile(self, filepath):
        img = Image.open(filepath)
        return self._fromImage(img)

    def fromImage(self, img):
        return self._fromImage(img.copy())

    def _fromImage(self, img):
        if max(img.size) > 512:
            img.thumbnail((512, 512))
        numCols, numRows = img.size

        # Optimisation 1: luma via numpy vectorised weighted sum
        if img.mode != "RGB":
            img = img.convert("RGB")
        rgb = np.array(img, dtype=np.float64)
        luma_2d = (
            self.LUMA_FROM_R_COEFF * rgb[:, :, 0]
            + self.LUMA_FROM_G_COEFF * rgb[:, :, 1]
            + self.LUMA_FROM_B_COEFF * rgb[:, :, 2]
        )

        # Optimisation 2: box filter via parallel Numba JIT
        windowSizeAlongRows = self.computeBoxFilterWindowSize(numCols)
        windowSizeAlongCols = self.computeBoxFilterWindowSize(numRows)
        flat_input = luma_2d.ravel()
        flat_output = np.empty_like(flat_input)
        _boxFilter_numba(flat_input, flat_output, numRows, numCols,
                         windowSizeAlongRows, windowSizeAlongCols)
        filtered_2d = flat_output.reshape((numRows, numCols))

        # Optimisation 3a: decimate via numpy fancy indexing
        i_idx = ((np.arange(64) + 0.5) * numRows / 64).astype(int)
        j_idx = ((np.arange(64) + 0.5) * numCols / 64).astype(int)
        dec64 = filtered_2d[np.ix_(i_idx, j_idx)]

        # Optimisation 3b: DCT via numpy matmul (BLAS)
        D = self._get_dct_np()
        dct16 = D @ dec64 @ D.T

        # Optimisation 3c: hash via vectorised median comparison
        return self._dct_to_hash_numpy(dct16)

    @staticmethod
    def _dct_to_hash_numpy(dct16):
        median = np.median(dct16)
        bits = (dct16 > median).astype(int)
        flipped = np.flip(bits, axis=(0, 1))
        return ImageHash(flipped.reshape(256))

    def fromFiles(self, paths, n_workers=None):
        """Hash multiple images in parallel using multiprocessing."""
        if n_workers is None:
            n_workers = os.cpu_count()
        with Pool(processes=n_workers) as pool:
            results = pool.map(_hash_one, paths)
        return dict(results)


if __name__ == "__main__":
    import sys
    hasher = FINDHasherOptimised()
    for filename in sys.argv[1:]:
        h = hasher.fromFile(filename)
        print("{},{}".format(h, filename))
