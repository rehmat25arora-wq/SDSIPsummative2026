#!/usr/bin/env python

import os
from multiprocessing import Pool

import numpy as np
from PIL import Image
from imagehash import ImageHash
from numba import njit, prange

from FINd import FINDHasher


# ---------------------------------------------------------------------------
# Fast box filter using an integral image (summed-area table)
# Complexity: O(rows * cols) instead of O(rows * cols * rowWin * colWin)
# ---------------------------------------------------------------------------

@njit(parallel=True, fastmath=True, cache=True)
def _boxFilter_integral_numba(input_flat, output_flat, rows, cols, rowWin, colWin):
    img = input_flat.reshape((rows, cols))

    # Summed-area table with 1-pixel padding
    integral = np.zeros((rows + 1, cols + 1), dtype=np.float32)

    # Build integral image
    for i in range(rows):
        row_sum = 0.0
        for j in range(cols):
            row_sum += img[i, j]
            integral[i + 1, j + 1] = integral[i, j + 1] + row_sum

    halfRowWin = (rowWin + 2) // 2
    halfColWin = (colWin + 2) // 2

    # Apply box filter using four-lookups from the integral image
    for i in prange(rows):
        xmin = max(0, i - halfRowWin)
        xmax = min(rows, i + halfRowWin)

        for j in range(cols):
            ymin = max(0, j - halfColWin)
            ymax = min(cols, j + halfColWin)

            s = (
                integral[xmax, ymax]
                - integral[xmin, ymax]
                - integral[xmax, ymin]
                + integral[xmin, ymin]
            )

            output_flat[i * cols + j] = s / ((xmax - xmin) * (ymax - ymin))


# ---------------------------------------------------------------------------
# Multiprocessing helper
# ---------------------------------------------------------------------------

def _hash_one(path):
    hasher = FINDHasherOptimised()
    return path, str(hasher.fromFile(path))


# ---------------------------------------------------------------------------
# Optimised FIND Hasher
# ---------------------------------------------------------------------------

class FINDHasherOptimised(FINDHasher):

    _DCT_np = None

    def _get_dct_np(self):
        if FINDHasherOptimised._DCT_np is None:
            FINDHasherOptimised._DCT_np = np.array(
                self.DCT_matrix,
                dtype=np.float32
            )
        return FINDHasherOptimised._DCT_np

    def fromFile(self, filepath):
        with Image.open(filepath) as img:
            return self._fromImage(img)

    def fromImage(self, img):
        return self._fromImage(img.copy())

    def _fromImage(self, img):
        # Resize large images for bounded runtime
        if max(img.size) > 512:
            img.thumbnail((512, 512))

        numCols, numRows = img.size

        # Ensure RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Vectorised RGB -> luma conversion
        rgb = np.array(img, dtype=np.float32)

        luma_2d = (
            self.LUMA_FROM_R_COEFF * rgb[:, :, 0]
            + self.LUMA_FROM_G_COEFF * rgb[:, :, 1]
            + self.LUMA_FROM_B_COEFF * rgb[:, :, 2]
        ).astype(np.float32)

        # Adaptive window sizes
        windowSizeAlongRows = self.computeBoxFilterWindowSize(numRows)
        windowSizeAlongCols = self.computeBoxFilterWindowSize(numCols)

        # Integral-image box filter
        flat_input = luma_2d.ravel()
        flat_output = np.empty_like(flat_input)

        _boxFilter_integral_numba(
            flat_input,
            flat_output,
            numRows,
            numCols,
            windowSizeAlongRows,
            windowSizeAlongCols,
        )

        filtered_2d = flat_output.reshape((numRows, numCols))

        # Decimate to 64x64
        i_idx = ((np.arange(64) + 0.5) * numRows / 64).astype(np.int32)
        j_idx = ((np.arange(64) + 0.5) * numCols / 64).astype(np.int32)

        dec64 = filtered_2d[np.ix_(i_idx, j_idx)]

        # DCT using cached matrix
        D = self._get_dct_np()
        dct16 = D @ dec64 @ D.T

        return self._dct_to_hash_numpy(dct16)

    @staticmethod
    def _dct_to_hash_numpy(dct16):
        median = np.median(dct16)

        bits = (dct16 > median).astype(np.uint8)

        # Match original orientation
        flipped = np.flip(bits, axis=(0, 1))

        return ImageHash(flipped.reshape(256))

    def fromFiles(self, paths, n_workers=None):
        """
        Hash multiple files in parallel.
        """
        if n_workers is None:
            n_workers = os.cpu_count()

        with Pool(processes=n_workers) as pool:
            results = pool.map(_hash_one, paths)

        return dict(results)


# ---------------------------------------------------------------------------
# Command-line usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    hasher = FINDHasherOptimised()

    for filename in sys.argv[1:]:
        h = hasher.fromFile(filename)
        print(f"{h},{filename}")