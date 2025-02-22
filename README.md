# SnapHound

SnapHound is an AI-powered image and video search tool that enables efficient retrieval using deep learning models.

## Features
- AI-driven image and text-based search using SIGCLIP
- SQLite-based storage for embeddings

## Installation
```sh
pip install -e .
```
or
```sh
pip install git+https://github.com/jebin2/SnapHoundPy.git
```

## Usage
```python
from snaphoundpy import SnapHound

# Initialize with paths to index
sh = SnapHound(paths=['/path/to/images'])

# Search with text
results = sh.search_with_text("A dog playing in the park")
print(results)

# Search with an image
results = sh.search_with_image("/path/to/query.jpg")
print(results)
```

## Requirements
- Python 3.8+
- torch
- transformers
- numpy
- faiss-cpu
- pillow
- python-dotenv