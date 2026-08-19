#!/bin/bash

# Debate with majority voting on 5 MMLU-Pro categories
# Agents: claude-haiku, deepseek-r1, qwen-32b

AGENTS="claude-haiku deepseek-r1 qwen-32b"
NUM_ROUNDS=5
CATEGORIES=("engineering" "chemistry" "law" "math" "physics")
RESULTS_DIR="results/haiku-deepseek-qwen-32b"

mkdir -p "$RESULTS_DIR"
mkdir -p logs

echo "Debate Majority Voting - MMLU-Pro"
echo "Agents: $AGENTS"
echo "Rounds: $NUM_ROUNDS"
echo "Categories: ${CATEGORIES[*]}"
echo "========================================"
echo ""

for CATEGORY in "${CATEGORIES[@]}"; do
    echo "Running $CATEGORY..."
    echo "----------------------------------------"

    python -u run_evaluation.py \
        --mode debate-majority \
        --agents $AGENTS \
        --dataset mmlu-pro \
        --category $CATEGORY \
        --rounds $NUM_ROUNDS \
        --save-file "$RESULTS_DIR/${CATEGORY}_debate_majority_haiku_deepseek_qwen-32b.json" \
        2>&1 | tee "logs/${CATEGORY}_debate_majority_haiku_deepseek_qwen-32b.out"

    echo ""
done

echo "========================================"
echo "All categories completed!"
echo "Results saved to: $RESULTS_DIR/"
