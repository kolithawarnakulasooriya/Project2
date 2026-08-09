# FashionMNIST CNN (Project2)

A compact PyTorch pipeline for training, evaluating, and serving a CNN on the FashionMNIST dataset.

## Requirements

- Python 3.8+
- See `requirements.txt` for exact pinned dependencies

## Quick start

1. Create & activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. (Optional) Prepare the dataset directory if you want to provide local data: place files under `dataset/`.

## Project structure

- `src/train.py` — training entrypoint (downloads/uses FashionMNIST, saves model weights)
- `src/evaluation.py` — evaluation script that loads saved weights and reports metrics
- `src/service.py` — FastAPI app for serving predictions
- `src/lib/model.py` — model definition
- `src/lib/transform.py` — image preprocessing pipeline
- `src/lib/data.py` — dataset helpers and loaders
- `src/lib/utils.py` — utility functions
- `requirements.txt` — Python dependencies
- `dataset/` — optional local dataset directory (gitignored)

## Usage

Train the model:

```bash
python src/train.py
```

Evaluate the model:

```bash
python src/evaluation.py
```

Run the API server (development):

```bash
uvicorn src.service:app --reload --host 0.0.0.0 --port 8000
```

Then POST an image file to the `/predict` endpoint to receive a classification.

## Development notes

- Model weights are saved/loaded by the training and evaluation scripts; check `src/train.py` for the exact filename and location.
- Use `dataset/` for local dataset overrides — the code falls back to downloading FashionMNIST if data is not present.
- Add or update dependencies in `requirements.txt` and re-run `pip install -r requirements.txt`.

If you'd like, I can also:

- add a short example `curl` request for the `/predict` endpoint,
- or run the test/train script to verify everything works in your environment.

---

Updated to match the current `src/` layout and usage commands.
