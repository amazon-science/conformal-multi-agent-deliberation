#!/bin/bash

# Define variables
CATEGORY="engineering"
AGENT=("claude-haiku" "deepseek-r1" "qwen-80b" "nova-pro" "qwen-32b")
MODE="greedy"
NUM_SAMPLES=10

for AGENT in "${AGENT[@]}"; do
    # for CATEGORY in "${CATEGORY[@]}"; do
    echo "Running evaluation for agent: ${AGENT}/${CATEGORY}"
    
    mkdir -p "results/${AGENT}/${CATEGORY}"

    # Run evaluation
    python -u run_evaluation.py \
        --mode ${MODE} \
        --agent ${AGENT} \
        --category ${CATEGORY} \
        --num-samples ${NUM_SAMPLES} \
        --save-file "results/${AGENT}/${CATEGORY}/${MODE}_${AGENT}.json" \
        2>&1 | tee "logs/${CATEGORY}_${MODE}_${AGENT}_results.out"
done