
## Installation

```bash
conda create -n allennlp python=3.7
conda activate allennlp
pip install allennlp==1.3.0
```

## Download English CoNLL

```bash
get_data.sh
```

## Training

See `scripts/sweep.sh`

```bash
export LEARNING_RATE=2e-05
allennlp train configs/ner-bert.json -s experiments/exp-${LEARNING_RATE} --include-package mylib
```

## Prediction

```bash
allennlp predict experiments/exp-2e-05/ conll2003/eng.testa --use-dataset-reader --cuda-device 0 --output-file out.txt
```
