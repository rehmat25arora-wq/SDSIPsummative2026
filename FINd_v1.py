#!/usr/bin/env python

"""
FINd v1 — loop-level optimisation only (no numpy).

Changes from original FINd.py:
1. boxFilter: fixed stride bug (k*rows -> k*cols), precomputed xmin/xmax
   outside inner loop to avoid redundant recalculation per j iteration.
2. fillFloatLumaFromBufferImage: cached coefficient lookups as locals to
   reduce attribute access overhead in tight loop.
"""

from FINd import FINDHasher


class FINDHasherV1(FINDHasher):

    def fillFloatLumaFromBufferImage(self, img, luma):
        numCols, numRows = img.size
        rgb_image = img.convert("RGB")
        r_coeff = self.LUMA_FROM_R_COEFF
        g_coeff = self.LUMA_FROM_G_COEFF
        b_coeff = self.LUMA_FROM_B_COEFF
        for i in range(numRows):
            for j in range(numCols):
                r, g, b = rgb_image.getpixel((j, i))
                luma[i * numCols + j] = r_coeff * r + g_coeff * g + b_coeff * b

    @classmethod
    def boxFilter(cls, input, output, rows, cols, rowWin, colWin):
        halfColWin = (colWin + 2) // 2
        halfRowWin = (rowWin + 2) // 2
        for i in range(rows):
            xmin = max(0, i - halfRowWin)
            xmax = min(rows, i + halfRowWin)
            for j in range(cols):
                ymin = max(0, j - halfColWin)
                ymax = min(cols, j + halfColWin)
                s = 0.0
                for k in range(xmin, xmax):
                    for l in range(ymin, ymax):
                        s += input[k * cols + l]   # bug fix: cols not rows
                output[i * cols + j] = s / ((xmax - xmin) * (ymax - ymin))
