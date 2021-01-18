

# Set this to the number of times to repeat
END=5

# Set this to the winning value from sweep.sh
LR=3e-05

for run in $(seq 1 $END); do
    export LEARNING_RATE=${LR}
    allennlp train configs/ner-bert.json -s experiments/exp-${LEARNING_RATE}-run-${run}
done
