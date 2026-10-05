#!/bin/bash

TSV_FILE="experiments.tsv"

# Number of lines in the TSV
N=$(wc -l < "$TSV_FILE")

echo "Running $N tasks from $TSV_FILE"

for ((i=0; i<N; i++)); do
    echo "Running task $i / $((N-1))"

    python task_train_sbert.py "$TSV_FILE" "$i"

    # Stop if the Python script fails
    if [ $? -ne 0 ]; then
        echo "Task $i failed. Stopping."
        exit 1
    fi
done

echo "All tasks completed."