#!/bin/bash

#########=====================================================================================================
## HAIKU + DEEPSEEK + QWEN-32B - Entropy Weighting - All Categories with alpha 0.01, 0.05 and 0.1
#########=====================================================================================================

LOGDIR=logs/haiku-deepseek-qwen32b-conformal-entropy
RESULTSDIR=results/haiku-deepseek-qwen-32b
CATEGORIES=(engineering chemistry law math physics economics health psychology)
ALPHAS=(0.05 0.1) #0.01
EARLY_STOP=true #false  # set to true to enable early stopping

mkdir -p "$LOGDIR"

echo ""
echo "Starting conformal prediction tests (ENTROPY weighting) for HAIKU + DEEPSEEK + QWEN-32B..."
echo "Categories: ${CATEGORIES[*]}"
echo "Alpha values: ${ALPHAS[*]}"
echo "Early stop: ${EARLY_STOP}"
echo "Weight strategy: entropy"
echo "========================================================================"
echo ""

for category in "${CATEGORIES[@]}"; do
    echo "Testing $category category (entropy weighting)..."
    echo "----------------------------------------"

    for alpha in "${ALPHAS[@]}"; do
        alpha_str=$(echo $alpha | tr -d '.')

        if [ "$EARLY_STOP" = true ]; then
            echo "$category - Alpha $alpha (with early stop)"
            python -u run_evaluation.py --mode debate-conformal \
                --results-file ${RESULTSDIR}/${category}_debate_majority_haiku_deepseek_qwen-32b.json \
                --alpha ${alpha} --weight-strategy entropy --early-stop \
                2>&1 | tee ${LOGDIR}/${category}_debate_conformal_haiku_deepseek_qwen-32b_alpha_${alpha_str}_early_stop.out
        else
            echo "$category - Alpha $alpha (no early stop)"
            python -u run_evaluation.py --mode debate-conformal \
                --results-file ${RESULTSDIR}/${category}_debate_majority_haiku_deepseek_qwen-32b.json \
                --alpha ${alpha} --weight-strategy entropy \
                2>&1 | tee ${LOGDIR}/${category}_debate_conformal_haiku_deepseek_qwen-32b_alpha_${alpha_str}.out
        fi
    done

    echo ""
done

echo "========================================================================"
echo "All entropy-weighted conformal prediction tests completed!"
echo "========================================================================"
