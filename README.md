# FINd Image Hashing Library

A Python library for perceptual image hashing using the FINd algorithm, with a RESTful API for comparing images at scale.

---

## What it does

FINd converts images into 256-bit perceptual hashes. Images that look similar will have hashes with a small Hamming distance. This makes it fast to find near-duplicate images without comparing pixels directly.

---

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Hash a single image
```
from FINd_optimised import FINDHasherOptimised

hasher = FINDHasherOptimised()
h = hasher.fromFile("image.jpg")
print(h)
```

### Compare two images
```
h1 = hasher.fromFile("image1.jpg")
h2 = hasher.fromFile("image2.jpg")
print(h1 - h2)  # Hamming distance

```

### Hash multiple images in parallel
```
hashes = hasher.fromFiles(image_paths, n_workers=4)
```

## Running the API

### With Python
```
python -m uvicorn api.main:app --host 0.0.0.0 --port 8945
```

### With Docker
```
docker build -t find-api -f api/Dockerfile .
docker run -p 8945:8945 find-api
```
### API usage
```
curl -X POST "http://127.0.0.1:8945/compare" \
  -F "image1=@meme_images/0660_23034755.jpg" \
  -F "image2=@meme_images/0012_12173443.jpg"

Response:
{
  "image1_hash": "20f7881c9e863231efca8d383cf4c35fc69cd844b1e13a69be787c9a661d7382",
  "image2_hash": "9a4e135b6c9e61330e37c49393e31e4e7cd9618c692c92b1c6b438f17963b6ac",
  "distance": 134
}
```

## Running tests
```
python -m pytest tests/test_find.py -v
```

## Repository structure
```
api/                        # FastAPI server
tests/                      # Unit tests
legacy/                     # Original and early optimisation

FINd_optimised.py           # Optimised version 
FINd_gpu.py                 # GPU implementation ```

benchmark.py                # Benchmark script
benchmark_gpu.py            # GPU benchmarking
Benchmarking and Profiling.ipynb  # Profiling and analysis

matrix.py                   # Matrix utilities
requirements.txt            # Dependencies
```
