import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from FINd_optimised import FINDHasherOptimised

app = FastAPI()
hasher = FINDHasherOptimised()


def _hamming(h1: str, h2: str) -> int:
    bits1 = bin(int(h1, 16))[2:].zfill(len(h1) * 4)
    bits2 = bin(int(h2, 16))[2:].zfill(len(h2) * 4)
    return sum(b1 != b2 for b1, b2 in zip(bits1, bits2))


@app.post("/compare")
async def compare(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    img1 = Image.open(io.BytesIO(await image1.read()))
    img2 = Image.open(io.BytesIO(await image2.read()))

    hash1 = str(hasher.fromImage(img1))
    hash2 = str(hasher.fromImage(img2))

    return {
        "image1_hash": hash1,
        "image2_hash": hash2,
        "distance": _hamming(hash1, hash2),
    }
