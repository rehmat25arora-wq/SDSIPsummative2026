import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io

from FINd_optimised import FINDHasherOptimised

app = FastAPI(title="FINd Image Hashing API")
hasher = FINDHasherOptimised()


@app.post("/compare")
async def compare(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    """
    Compare two images using the FINd algorithm.
    Returns the FINd hash of each image and the Hamming distance between them.
    """
    try:
        img1 = Image.open(io.BytesIO(await image1.read()))
        img2 = Image.open(io.BytesIO(await image2.read()))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read images: {e}")

    hash1 = hasher.fromImage(img1)
    hash2 = hasher.fromImage(img2)

    return {
        "image1_hash": str(hash1),
        "image2_hash": str(hash2),
        "distance": int(hash1 - hash2),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8945)