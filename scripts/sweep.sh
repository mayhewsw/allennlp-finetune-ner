for LR in 5e-05 3e-05 2e-05; do
    export LEARNING_RATE=${LR}
    allennlp train configs/ner-bert.json -s experiments/exp-${LEARNING_RATE}
done

