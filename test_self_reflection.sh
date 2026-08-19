#!/bin/bash

# Test script for self-reflection mode
# Single agent iteratively reflects on its own responses

echo "Testing Self-Reflection Mode"
echo "============================="
echo ""
echo "This mode prompts a single agent to:"
echo "1. Provide an initial answer"
echo "2. Reflect on its answer k times"
echo "3. Update confidence distribution after each reflection"
echo ""

# Configuration
AGENT="claude-haiku"
NUM_REFLECTIONS=3
NUM_SAMPLES=20
CATEGORY="engineering" #"math"

# Run self-reflection evaluation
python run_evaluation.py \
    --mode self-reflection \
    --agent $AGENT \
    --rounds $NUM_REFLECTIONS \
    --num-samples $NUM_SAMPLES \
    --category $CATEGORY \
    --use-subset \
    --save-file results/self_reflection_${AGENT}_${NUM_REFLECTIONS}reflections_${CATEGORY}_${NUM_SAMPLES}_results.json

echo ""
echo "Results saved to: results/self_reflection_${AGENT}_${NUM_REFLECTIONS}reflections_${CATEGORY}_${NUM_SAMPLES}_results.json"