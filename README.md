# Trojan Traffic ML

This repository contains a small PyTorch pipeline for classifying network traffic as `Trojan` or `Benign`.

## Important attribution note

This project folder was initialized from a downloaded archive named `TROJAN_TRAFFIC_ML-main`. Before publishing this repository publicly, add the original author, source URL, and license information if this code came from someone else.

If you do not have permission or a license that allows redistribution, do not publish it publicly as your own original work.

## Project structure

- `data/Trojan_Detection_sample.csv`: sample dataset
- `src/preprocess.py`: data cleaning, splitting, and scaling
- `src/model.py`: MLP classifier
- `src/train.py`: training entry point
- `src/evaluate.py`: evaluation script
- `test_preprocess.py`: preprocessing pipeline test

## Setup

```bash
pip install -r requirements.txt
```

## Run training

```bash
python -m src.train
```

## Run evaluation

```bash
python -m src.evaluate
```

## Run tests

```bash
pytest
```

## Run with Docker

```bash
docker build -t trojan-traffic-ml .
docker run --rm trojan-traffic-ml
```
