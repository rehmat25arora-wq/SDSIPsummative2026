# FINd Image Hashing Library

A Python library for perceptual image hashing using the FINd algorithm, with a RESTful API for comparing images at scale.

## What it does

FINd converts images into 256-bit perceptual hashes. Images that look similar will have hashes with a small Hamming distance. This makes it fast to find near-duplicate images without comparing pixels directly.

## Installation

pip install -r requirements.txt

## Usage

### Hash a single image

from FINd_optimised import FINDHasherOptimised

hasher = FINDHasherOptimised()
h = hasher.fromFile("image.jpg")
print(h)

### Compare two images

h1 = hasher.fromFile("image1.jpg")
h2 = hasher.fromFile("image2.jpg")
print(h1 - h2)  # Hamming distance

### Hash multiple images in parallel

hashes = hasher.fromFiles(image_paths, n_workers=4)

## Running the API

### With Python

python -m uvicorn api.main:app --host 0.0.0.0 --port 8945

### With Docker

docker build -t find-api -f api/Dockerfile .
docker run -p 8945:8945 find-api

### API usage

curl -X POST "http://127.0.0.1:8945/compare" \
  -F "image1=@image1.jpg" \
  -F "image2=@image2.jpg"

Response:
{"image1_hash": "393b246d...","image2_hash": "18ab6c6f...","distance": 40}

## Running tests

python -m pytest tests/test_find.py -v

## Repository structure

FINd.py                 # Original FINd algorithm
FINd_optimised.py       # Optimised version (numpy + multiprocessing)
matrix.py               # Matrix utilities
api/
    main.py             # FastAPI server
    Dockerfile          # Docker container
tests/
    test_find.py        # Unit tests
benchmarking.ipynb      # Profiling and accuracy analysis
requirements.txt        # Dependencies
