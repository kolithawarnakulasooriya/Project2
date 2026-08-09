# FashionMNIST CNN Project

This repository contains a simple PyTorch-based FashionMNIST classification workflow with training, evaluation, and deployment steps.

## Project Structure

- `step_model_training.py` - trains the `CNNModel` on FashionMNIST and saves weights to `fashin.pth`
- `step_model_evaluation.py` - evaluates the trained model on the FashionMNIST test set
- `step_model_deploy.py` - FastAPI app for serving predictions using the trained weights
- `model.py` - CNN model definition
- `transform.py` - image preprocessing pipeline
- `requirements.txt` - Python dependencies
- `.github/workflows/train.yml` - GitHub Actions workflow for running model training
- `.gitignore` - ignores dataset directory, `fashin.pth`, and virtual environment files

## Setup

1. Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Install the app server if you want to run deployment locally:

```bash
pip install "uvicorn[standard]"
```

## Usage

### Train the model

```bash
python step_model_training.py
```

This will download FashionMNIST, train the model, and save the weights to `fashin.pth`.

### Evaluate the model

```bash
python step_model_evaluation.py
```

This will load `fashin.pth` and evaluate model accuracy on the test dataset.

### Deploy the model

```bash
uvicorn step_model_deploy:app --reload --host 0.0.0.0 --port 8000
```

Then send a POST request to `/predict` with an image file to get a prediction.

## GitHub Actions

The repository includes a workflow at `.github/workflows/train.yml` that runs `step_model_training.py` on pushes to `main` and supports manual dispatch.

## Notes

- The dataset directory is ignored by `.gitignore`.
- The trained model file `fashin.pth` is also ignored.
- If you need to stop tracking files already committed, use `git rm --cached <path>`.
