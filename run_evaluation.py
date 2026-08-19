"""
Unified Evaluation Script for Conformal Social Choice Framework

Combines all evaluation modes:
1. Greedy Single Agent - Top-1 prediction from single agent
2. Self-Reflection - Single agent with iterative self-reflection (k rounds)
3. Majority Voting - Multi-agent majority vote with debate
4. Multi-Agent Conformal - Conformal prediction with existing debate results

Supported Datasets:
- MMLU-Pro: Multiple categories (math, physics, chemistry, law, engineering, business, economics, health, psychology)
- GPQA: Graduate-level questions (gpqa_main, gpqa_diamond, gpqa_extended)
- ARC: AI2 Reasoning Challenge (ARC-Challenge, ARC-Easy)
- MuSR: Multi-Step Reasoning (murder_mysteries, object_placements, team_allocation)

Usage:
    # MMLU-Pro examples
    python run_evaluation.py --mode greedy --agent qwen --dataset mmlu-pro --category math --num-samples 100
    python run_evaluation.py --mode self-reflection --agent qwen --dataset mmlu-pro --rounds 3 --num-samples 50

    # GPQA examples
    python run_evaluation.py --mode greedy --agent claude-haiku --dataset gpqa --gpqa-split gpqa_main --num-samples 50
    python run_evaluation.py --mode debate-majority --agents qwen claude-haiku nova-lite --dataset gpqa --rounds 3

    # ARC examples
    python run_evaluation.py --mode greedy --agent claude-haiku --dataset arc --gpqa-split ARC-Challenge --num-samples 100
    python run_evaluation.py --mode debate-majority --agents qwen claude-haiku nova-lite --dataset arc --gpqa-split ARC-Easy --rounds 3

    # MuSR examples
    python run_evaluation.py --mode greedy --agent claude-haiku --dataset musr --gpqa-split murder_mysteries --num-samples 50
    python run_evaluation.py --mode debate-majority --agents qwen claude-haiku nova-lite --dataset musr --gpqa-split object_placements --rounds 3
"""

import numpy as np
import logging
import argparse
import json
import os
from typing import List, Dict, Optional
from collections import Counter
from datasets import load_dataset

from main import (
    MultiAgentDebateManager,
    AgentResponse,
    SocialWelfareAggregator,
    ConformalCalibrator
)
from utils import load_test_data, load_gpqa_data, load_arc_data, load_musr_data, format_question_with_options
from data.negotiation.load_negotiation import load_negotiation_data
from data.contracts.load_cuad import load_cuad_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions for Data Loading
# ============================================================================

def load_evaluation_data(
    dataset: str = 'mmlu-pro',
    category: str = 'business',
    gpqa_split: str = 'gpqa_main',
    use_subset: bool = False,
    num_samples: int = 100,
    start_index: int = 0
):
    """
    Unified data loader for all supported datasets.

    Args:
        dataset: Dataset to use ('mmlu-pro' or 'gpqa')
        category: Category for MMLU-Pro (business, math, physics, etc.)
        gpqa_split: Split for GPQA (gpqa_main, gpqa_diamond, gpqa_extended)
        use_subset: Whether to use a subset
        num_samples: Number of samples when using subset
        start_index: Starting index for subset

    Returns:
        List of test examples in standardized format
    """
    if dataset == 'mmlu-pro':
        return load_test_data(
            category=category,
            use_subset=use_subset,
            num_samples=num_samples,
            start_index=start_index
        )
    elif dataset == 'gpqa':
        return load_gpqa_data(
            split=gpqa_split,
            use_subset=use_subset,
            num_samples=num_samples,
            start_index=start_index
        )
    elif dataset == 'negotiation':
        return load_negotiation_data(
            role=category if category in ('buyer', 'seller') else 'buyer',
            use_subset=use_subset,
            num_samples=num_samples,
            start_index=start_index,
            require_price_context=True,
        )
    elif dataset == 'cuad':
        return load_cuad_data(
            use_subset=use_subset,
            num_samples=num_samples,
            start_index=start_index,
        )
    elif dataset == 'arc':
        return load_arc_data(
            split=gpqa_split if gpqa_split in ('ARC-Challenge', 'ARC-Easy') else 'ARC-Challenge',
            use_subset=use_subset,
            num_samples=num_samples,
            start_index=start_index,
        )
    elif dataset == 'musr':
        valid_musr = ('murder_mysteries', 'object_placements', 'team_allocation')
        return load_musr_data(
            split=gpqa_split if gpqa_split in valid_musr else 'murder_mysteries',
            use_subset=use_subset,
            num_samples=num_samples,
            start_index=start_index,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Must be 'mmlu-pro', 'gpqa', 'negotiation', 'cuad', 'arc', or 'musr'")


# ============================================================================
# Helper Functions for Save/Load
# ============================================================================

def save_results(results: Dict, filepath: str):
    """Save evaluation results to JSON file."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"✓ Results saved to {filepath}")


def load_results(filepath: str) -> Optional[Dict]:
    """Load evaluation results from JSON file."""
    if not os.path.exists(filepath):
        logger.warning(f"File {filepath} not found. Starting fresh evaluation.")
        return None
    
    try:
        with open(filepath, 'r') as f:
            results = json.load(f)
        # logger.info(f"✓ Loaded previous results from {filepath}")
        return results
    except Exception as e:
        logger.error(f"Error loading results from {filepath}: {str(e)}")
        return None


def merge_greedy_results(old_results: Dict, new_results: Dict) -> Dict:
    """Merge old and new greedy evaluation results."""
    merged = {
        'mode': new_results['mode'],
        'agent': new_results['agent'],
        'results': old_results.get('results', []) + new_results['results']
    }
    
    # Recalculate statistics
    all_results = merged['results']
    correct = sum(1 for r in all_results if r['is_correct'])
    total = len(all_results)
    
    merged['correct'] = correct
    merged['total'] = total
    merged['accuracy'] = correct / total if total > 0 else 0.0
    
    return merged


def merge_majority_results(old_results: Dict, new_results: Dict) -> Dict:
    """Merge old and new majority voting results."""
    merged = {
        'mode': new_results['mode'],
        'agents': new_results['agents'],
        'results': old_results.get('results', []) + new_results['results']
    }
    
    # Recalculate statistics
    all_results = merged['results']
    correct = sum(1 for r in all_results if r['is_correct'])
    total = len(all_results)
    
    merged['correct'] = correct
    merged['total'] = total
    merged['accuracy'] = correct / total if total > 0 else 0.0
    
    return merged


def merge_debate_majority_results(old_results: Dict, new_results: Dict) -> Dict:
    """Merge old and new debate majority voting results."""
    num_rounds = new_results['num_rounds']
    
    # Merge results per round
    merged_results_per_round = []
    for round_idx in range(num_rounds):
        old_round = old_results.get('results_per_round', [[]])[round_idx] if round_idx < len(old_results.get('results_per_round', [])) else []
        new_round = new_results['results_per_round'][round_idx]
        merged_results_per_round.append(old_round + new_round)
    
    # Recalculate statistics per round
    round_summaries = []
    for round_idx in range(num_rounds):
        round_results = merged_results_per_round[round_idx]
        correct = sum(1 for r in round_results if r['is_correct'])
        total = len(round_results)
        accuracy = correct / total if total > 0 else 0.0
        
        round_summaries.append({
            'round': round_idx + 1,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'improvement_from_round1': accuracy - round_summaries[0]['accuracy'] if round_idx > 0 else 0.0
        })
    
    baseline_accuracy = round_summaries[0]['accuracy']
    final_accuracy = round_summaries[-1]['accuracy']
    
    return {
        'mode': new_results['mode'],
        'agents': new_results['agents'],
        'num_rounds': num_rounds,
        'round_summaries': round_summaries,
        'initial_accuracy': baseline_accuracy,
        'final_accuracy': final_accuracy,
        'total_improvement': final_accuracy - baseline_accuracy,
        'results_per_round': merged_results_per_round
    }


# ============================================================================
# MODE 1: Greedy Single Agent
# ============================================================================

def evaluate_greedy_single_agent(
    agent_name: str,
    dataset: str = 'mmlu-pro',
    data_category: str = 'business',
    gpqa_split: str = 'gpqa_main',
    use_subset: bool = False,
    num_samples: int = 100,
    max_tokens: int = 4096,
    start_index: int = 0,
    load_file: Optional[str] = None
):
    """
    Evaluate greedy single agent baseline (top-1 prediction).

    Args:
        agent_name: Name of agent to use
        dataset: Dataset to use ('mmlu-pro' or 'gpqa')
        data_category: Category for MMLU-Pro
        gpqa_split: Split for GPQA
        num_samples: Number of test samples to evaluate
        max_tokens: Maximum tokens for generation
        start_index: Starting index for continuation
        load_file: Path to previous results for continuation

    Returns:
        Dictionary with evaluation results
    """
    logger.info("\n" + "="*80)
    logger.info(f"MODE 1: GREEDY SINGLE AGENT - {agent_name}")
    logger.info("="*80)
    
    # Check for continuation
    previous_results = None
    evaluated_ids = set()
    
    if load_file:
        previous_results = load_results(load_file)
        if previous_results:
            # Validate compatibility
            if previous_results.get('mode') != 'greedy':
                logger.error(f"Cannot continue: previous mode was '{previous_results.get('mode')}', expected 'greedy'")
                return None
            if previous_results.get('agent') != agent_name:
                logger.error(f"Cannot continue: previous agent was '{previous_results.get('agent')}', expected '{agent_name}'")
                return None
            
            # Get already evaluated question IDs
            evaluated_ids = {r['question_id'] for r in previous_results.get('results', [])}
            logger.info(f"✓ Continuing from previous run with {len(evaluated_ids)} evaluated samples")
    
    # Load data
    test_data = load_evaluation_data(
        dataset=dataset,
        category=data_category,
        gpqa_split=gpqa_split,
        use_subset=use_subset,
        num_samples=num_samples,
        start_index=start_index
    )
    
    if not test_data:
        logger.warning("No test data found!")
        return None
    
    correct_predictions = 0
    results = []
    skipped_count = 0
    
    logger.info(f"\nEvaluating on {len(test_data)} samples...")
    if evaluated_ids:
        logger.info(f"Skipping {len(evaluated_ids)} already-evaluated samples")
    
    for idx, example in enumerate(test_data):
        question_id = example.get('question_id', idx)

        # Skip if already evaluated
        if question_id in evaluated_ids:
            skipped_count += 1
            continue
        
        question = example['question']
        options = example['options']
        correct_answer = example['answer']
        option_letters = [chr(65+i) for i in range(len(options))]
        
        # Initialize manager for this question
        manager = MultiAgentDebateManager(
            agents=[agent_name],
            options=option_letters,
            num_rounds=1,
            max_tokens=max_tokens
        )
        
        formatted_question = format_question_with_options(question, options)

        try:
            # Build prompt for single-agent evaluation (avoid debate context)
            single_agent_prompt = f"""{formatted_question}

Please analyze the question carefully and provide your confidence level for each answer option.

You MUST respond in the following format:
My confidence distribution:
A: [probability between 0 and 1]
B: [probability between 0 and 1]
C: [probability between 0 and 1]
...

The probabilities must sum to 1.0.

Example format:
<answer>
My confidence distribution:
A: 0.15
B: 0.60
C: 0.20
D: 0.05
</answer>

Note: If you revise your confidence during reasoning, please restate the full distribution in requested format at the end.
Now provide your response:"""

            # Get agent's probability distribution
            logits, response_text = manager.extract_logits(
                agent_id=agent_name,
                query=single_agent_prompt,
                round_num=0,
                history=[],
                use_raw_prompt=True  # Use direct prompt without debate wrapper
            )
            
            # Greedy: select option with highest probability
            prediction = max(logits.items(), key=lambda x: x[1])[0]
            confidence = logits[prediction]
            
            # logger.info(f"[{agent_name}] Raw response text:\n{response_text}")

            logger.info(f"[{agent_name}] Question {question_id}: {logits}")
            
            # Check if correct
            is_correct = prediction == correct_answer
            if is_correct:
                correct_predictions += 1
            
            results.append({
                'question_id': example.get('question_id', idx),
                'prediction': prediction,
                'correct_answer': correct_answer,
                'is_correct': is_correct,
                'confidence': confidence
            })
            
            if (idx + 1) % 10 == 0:
                current_acc = correct_predictions / (idx + 1)
                logger.info(f"Progress: {idx + 1}/{len(test_data)} | Accuracy: {current_acc:.2%}")
        
        except Exception as e:
            logger.error(f"Error on question {idx}: {str(e)}")
            results.append({
                'question_id': example.get('question_id', idx),
                'prediction': 'ERROR',
                'correct_answer': correct_answer,
                'is_correct': False,
                'confidence': 0.0
            })
    
    accuracy = correct_predictions / len(test_data)
    
    logger.info("\n" + "="*80)
    logger.info("RESULTS")
    logger.info("="*80)
    logger.info(f"Agent: {agent_name}")
    logger.info(f"Accuracy: {accuracy:.2%} ({correct_predictions}/{len(test_data)})")
    
    return {
        'mode': 'greedy',
        'agent': agent_name,
        'accuracy': accuracy,
        'correct': correct_predictions,
        'total': len(test_data),
        'results': results
    }

# ============================================================================
# MODE 2: Multi-Round Debate with Majority Voting
# ============================================================================

def evaluate_debate_majority_voting(
    agent_names: List[str],
    dataset: str = 'mmlu-pro',
    data_category: str = 'business',
    gpqa_split: str = 'gpqa_main',
    use_subset: bool = False,
    max_tokens: int = 4096,
    num_rounds: int = 3,
    num_samples: int = 50,
    start_index: int = 0,
    load_file: Optional[str] = None,
    early_stop: bool = False
):
    """
    Evaluate multi-round debate with majority voting.
    Shows accuracy improvement across debate rounds.

    Args:
        agent_names: List of agent names
        dataset: Dataset to use ('mmlu-pro' or 'gpqa')
        data_category: Category for MMLU-Pro
        gpqa_split: Split for GPQA
        max_tokens: Maximum tokens for generation
        num_rounds: Number of debate rounds
        num_samples: Number of test samples
        start_index: Starting index for sampling (default: 0)
        load_file: Path to previous results for continuation
        early_stop: Whether to stop debate when all agents reach consensus (default: False)

    Returns:
        Dictionary with evaluation results showing round-by-round improvement
    """
    logger.info("\n" + "="*80)
    logger.info(f"MODE 2: MULTI-ROUND DEBATE WITH MAJORITY VOTING")
    logger.info("="*80)
    logger.info(f"Agents: {', '.join(agent_names)}")
    logger.info(f"Debate Rounds: {num_rounds}")
    if early_stop:
        logger.info(f"Early Stopping: Enabled (stop when all agents agree)")
    
    # Check for continuation
    previous_results = None
    evaluated_ids = set()
    
    if load_file:
        previous_results = load_results(load_file)
        if previous_results:
            # Validate compatibility
            if previous_results.get('mode') != 'debate-majority':
                logger.error(f"Cannot continue: previous mode was '{previous_results.get('mode')}', expected 'debate-majority'")
                return None
            if previous_results.get('agents') != agent_names:
                logger.error(f"Cannot continue: previous agents were {previous_results.get('agents')}, expected {agent_names}")
                return None
            if previous_results.get('num_rounds') != num_rounds:
                logger.error(f"Cannot continue: previous num_rounds was {previous_results.get('num_rounds')}, expected {num_rounds}")
                return None
            
            # Get already evaluated question IDs from first round
            evaluated_ids = {r['question_id'] for r in previous_results.get('results_per_round', [[]])[0]}
            logger.info(f"✓ Continuing from previous run with {len(evaluated_ids)} evaluated samples")
    
    # Load test data
    test_data = load_evaluation_data(
        dataset=dataset,
        category=data_category,
        gpqa_split=gpqa_split,
        use_subset=use_subset,
        num_samples=num_samples,
        start_index=start_index
    )
    
    if not test_data:
        logger.warning("No test data found!")
        return None
    
    # Track results per round
    results_per_round = [[] for _ in range(num_rounds)]
    correct_per_round = [0 for _ in range(num_rounds)]
    skipped_count = 0
    early_stop_counts = [0 for _ in range(num_rounds)]  # Track stopping rounds
    sample_stopping_info = {}  # Track each sample's stopping round and results

    logger.info(f"\nEvaluating on {len(test_data)} samples...")
    if evaluated_ids:
        logger.info(f"Skipping {len(evaluated_ids)} already-evaluated samples")

    for idx, example in enumerate(test_data):
        question_id = example.get('question_id', idx)

        # Skip if already evaluated
        if question_id in evaluated_ids:
            skipped_count += 1
            continue

        question = example['question']
        options = example['options']
        correct_answer = example['answer']
        option_letters = [chr(65+i) for i in range(len(options))]

        formatted_question = format_question_with_options(question, options)

        # Initialize debate manager
        manager = MultiAgentDebateManager(
            agents=agent_names,
            options=option_letters,
            num_rounds=num_rounds,
            max_tokens=max_tokens
        )

        try:
            agent_logits = {}  # Initialize so error path can reference it safely

            # Run multi-round debate
            debate_instance = manager.run_debate(formatted_question, correct_answer)

            # Track early stopping for this sample
            stopped = False
            stopping_round = num_rounds  # Default to last round if no early stop
            sample_results = []  # Store all rounds for this sample

            # Analyze each round
            for round_idx in range(num_rounds):
                # Skip if already stopped (only applies when early_stop is True)
                if early_stop and stopped:
                    break

                round_responses = debate_instance.debate_history[round_idx]
                
                # Log individual predictions
                agent_predictions = {}
                agent_logits = {}
                for response in round_responses:
                    # Get argmax prediction from each agent
                    prediction = max(response.logits.items(), key=lambda x: x[1])[0]
                    agent_predictions[response.agent_id] = prediction
                    agent_logits[response.agent_id] = response.logits
                    logger.info(f"[{response.agent_id}] Round {round_idx + 1} prediction: {prediction} confidence: {response.logits[prediction]:.4f}")

                # Use SocialWelfareAggregator to compute social probabilities
                aggregator = SocialWelfareAggregator(
                    options=option_letters,
                    weight_strategy="uniform"
                )
                social_prob = aggregator.compute_social_scores(round_responses)

                # Select option with highest social probability
                majority_prediction = max(social_prob.items(), key=lambda x: x[1])[0]
                vote_confidence = social_prob[majority_prediction]

                # Count votes for logging
                votes = Counter(agent_predictions.values())

                # Check if consensus reached (all agents agree)
                consensus_reached = len(votes) == 1

                # Check if correct
                is_correct = majority_prediction == correct_answer
                if is_correct:
                    correct_per_round[round_idx] += 1

                # Store result for this round
                round_result = {
                    'question_id': example.get('question_id', idx),
                    'round': round_idx + 1,
                    'prediction': majority_prediction,
                    'correct_answer': correct_answer,
                    'is_correct': is_correct,
                    'votes': dict(votes),
                    'confidence': vote_confidence,
                    'agent_predictions': agent_predictions,
                    'agent_logits': agent_logits,
                    'consensus_reached': consensus_reached,
                    'stopped_here': False  # Will update if this is stopping round
                }
                sample_results.append(round_result)
                results_per_round[round_idx].append(round_result)

                # Early stopping: if all agents agree, stop here
                if early_stop and consensus_reached:
                    stopped = True
                    stopping_round = round_idx + 1
                    early_stop_counts[round_idx] += 1
                    # Mark this as the stopping round
                    round_result['stopped_here'] = True
                    logger.debug(f"Sample {question_id}: Early stop at round {stopping_round} (consensus reached)")
                    break
            
            # Track stopping info for this sample
            sample_stopping_info[question_id] = {
                'stopping_round': stopping_round - 1,  # 0-indexed
                'results': sample_results
            }

            logger.info(f"Question {idx} | Round: {round_idx + 1} | Votes: {dict(votes)} | Prediction: {majority_prediction} | "
                        f"Correct: {is_correct}")

            # Log sample result
            if (idx + 1) % 10 == 0:
                logger.info(f"\nProgress: {idx + 1}/{len(test_data)}")
                for round_idx in range(num_rounds):
                    acc = correct_per_round[round_idx] / (idx + 1)
                    logger.info(f"  Round {round_idx + 1} Accuracy: {acc:.2%}")
        
        except Exception as e:
            logger.error(f"Error on question {idx}: {str(e)}")
            for round_idx in range(num_rounds):
                results_per_round[round_idx].append({
                    'question_id': example.get('question_id', idx),
                    'round': round_idx + 1,
                    'prediction': 'ERROR',
                    'correct_answer': correct_answer,
                    'is_correct': False,
                    'votes': {},
                    'confidence': 0.0,
                    'agent_predictions': {},
                    'agent_logits': agent_logits
                })
    
    # Calculate final accuracies
    logger.info("\n" + "="*80)
    logger.info("RESULTS - ROUND-BY-ROUND ACCURACY")
    logger.info("="*80)

    # Calculate early stopping statistics
    total_samples = len(test_data)
    samples_stopped_early = sum(early_stop_counts)
    samples_reached_final_round = total_samples - samples_stopped_early

    if early_stop:
        logger.info(f"\n📊 Early Stopping Statistics:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Stopped early: {samples_stopped_early} ({samples_stopped_early/total_samples:.1%})")
        logger.info(f"  Reached final round: {samples_reached_final_round} ({samples_reached_final_round/total_samples:.1%})")

    # Header with conditional column for early stopping
    if early_stop:
        logger.info(f"\n{'Round':<10} {'Accuracy':<15} {'Correct/Total':<20} {'Improvement':<15} {'Stopped Here':<15} {'Stop Acc':<12}")
    else:
        logger.info(f"\n{'Round':<10} {'Accuracy':<15} {'Correct/Total':<20} {'Improvement':<15}")
    logger.info("-"*80)

    round_summaries = []

    for round_idx in range(num_rounds):
        if early_stop:
            # With early stopping: for each round, use results from samples' stopping round or current round
            correct_count = 0

            # Calculate accuracy for only samples that stopped at this round
            stopped_at_this_round_correct = 0
            stopped_at_this_round_total = 0

            for question_id, info in sample_stopping_info.items():
                stopping_round_idx = info['stopping_round']
                results_list = info['results']

                # Use result from stopping round if stopped before current round, else use current round
                result_idx = min(stopping_round_idx, round_idx)
                if result_idx < len(results_list):
                    result = results_list[result_idx]
                    correct_count += int(result['is_correct'])

                    # Track accuracy for samples that stopped at this specific round
                    if stopping_round_idx == round_idx and result.get('stopped_here', False):
                        stopped_at_this_round_total += 1
                        if result['is_correct']:
                            stopped_at_this_round_correct += 1

            accuracy = correct_count / total_samples if total_samples > 0 else 0

            # Calculate early stopped accuracy
            early_stopped_accuracy = stopped_at_this_round_correct / stopped_at_this_round_total if stopped_at_this_round_total > 0 else None
        else:
            # Without early stopping: use only samples that reached this round
            accuracy = correct_per_round[round_idx] / len(test_data)
            early_stopped_accuracy = None

        # Calculate improvement from round 1
        if round_idx == 0:
            baseline_accuracy = accuracy
            improvement = 0.0
        else:
            improvement = accuracy - baseline_accuracy

        stopped_count = early_stop_counts[round_idx]
        stopped_pct = stopped_count / total_samples if total_samples > 0 else 0

        # Log with conditional early stopping accuracy column
        if early_stop:
            stop_acc_str = f"{early_stopped_accuracy:.2%}" if early_stopped_accuracy is not None else "N/A"
            logger.info(f"{round_idx + 1:<10} {accuracy:<15.2%} "
                       f"{int(accuracy * total_samples)}/{total_samples:<17} "
                       f"{improvement:<15.2%} {stopped_count} ({stopped_pct:.1%})  {stop_acc_str:<12}")
        else:
            logger.info(f"{round_idx + 1:<10} {accuracy:<15.2%} "
                       f"{correct_per_round[round_idx]}/{len(test_data):<17} "
                       f"{improvement:+.2%}")

        round_summaries.append({
            'round': round_idx + 1,
            'accuracy': accuracy,
            'correct': int(accuracy * total_samples) if early_stop else correct_per_round[round_idx],
            'total': len(test_data),
            'improvement_from_round1': improvement,
            'early_stopped_accuracy': early_stopped_accuracy,
            'early_stop_count': stopped_count,
            'early_stop_rate': stopped_pct
        })
    
    # Additional statistics
    final_accuracy = round_summaries[-1]['accuracy']
    total_improvement = final_accuracy - baseline_accuracy

    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Initial Accuracy (Round 1): {baseline_accuracy:.2%}")
    logger.info(f"Final Accuracy (Round {num_rounds}): {final_accuracy:.2%}")
    logger.info(f"Total Improvement: {total_improvement:+.2%}")
    logger.info(f"Agents: {', '.join(agent_names)}")

    # Calculate overall early stopped accuracy and average stopping round
    overall_early_stopped_acc = None
    total_early_stopped_correct = 0
    total_early_stopped_samples = 0
    avg_stopping_round = num_rounds

    if early_stop:
        # Calculate average stopping round
        avg_stopping_round = sum((round_idx + 1) * count for round_idx, count in enumerate(early_stop_counts)) / samples_stopped_early if samples_stopped_early > 0 else num_rounds

        logger.info(f"\n🎯 Early Stopping Efficiency:")
        logger.info(f"  Average stopping round: {avg_stopping_round:.2f}")
        logger.info(f"  Samples stopped early: {samples_stopped_early}/{total_samples} ({samples_stopped_early/total_samples:.1%})")
        logger.info(f"  Potential rounds saved: {sum((num_rounds - (round_idx + 1)) * count for round_idx, count in enumerate(early_stop_counts))}")

        # Calculate overall early stopped accuracy
        for question_id, info in sample_stopping_info.items():
            stopping_round_idx = info['stopping_round']
            # Only count samples that stopped before the final round
            if stopping_round_idx < num_rounds - 1:
                results_list = info['results']
                if stopping_round_idx < len(results_list):
                    result = results_list[stopping_round_idx]
                    if result.get('stopped_here', False):
                        total_early_stopped_samples += 1
                        if result['is_correct']:
                            total_early_stopped_correct += 1

        if total_early_stopped_samples > 0:
            overall_early_stopped_acc = total_early_stopped_correct / total_early_stopped_samples
            logger.info(f"  Overall early stopped accuracy: {overall_early_stopped_acc:.2%} ({total_early_stopped_correct}/{total_early_stopped_samples})")

            # Compare to final round accuracy
            if final_accuracy is not None:
                early_vs_final = overall_early_stopped_acc - final_accuracy
                if early_vs_final > 0:
                    logger.info(f"  Early stopped samples are {early_vs_final:.2%} MORE accurate than final round")
                elif early_vs_final < 0:
                    logger.info(f"  Early stopped samples are {abs(early_vs_final):.2%} LESS accurate than final round")
                else:
                    logger.info(f"  Early stopped samples have SAME accuracy as final round")

    # Check if accuracy improved
    if total_improvement > 0:
        logger.info(f"\n✓ Debate improved accuracy by {total_improvement:.2%}")
    elif total_improvement < 0:
        logger.info(f"\n✗ Debate decreased accuracy by {abs(total_improvement):.2%}")
    else:
        logger.info(f"\n→ No change in accuracy across rounds")
    
    result = {
        'mode': 'debate-majority',
        'agents': agent_names,
        'num_rounds': num_rounds,
        'round_summaries': round_summaries,
        'initial_accuracy': baseline_accuracy,
        'final_accuracy': final_accuracy,
        'total_improvement': total_improvement,
        'results_per_round': results_per_round,
        'early_stop': early_stop
    }

    if early_stop:
        result['early_stopping'] = {
            'total_samples': total_samples,
            'stopped_early_count': samples_stopped_early,
            'stopped_early_rate': samples_stopped_early / total_samples if total_samples > 0 else 0,
            'avg_stopping_round': avg_stopping_round,
            'stop_counts_per_round': early_stop_counts,
            'potential_rounds_saved': sum((num_rounds - (round_idx + 1)) * count for round_idx, count in enumerate(early_stop_counts)),
            'overall_early_stopped_accuracy': overall_early_stopped_acc,
            'early_stopped_correct': total_early_stopped_correct,
            'early_stopped_total': total_early_stopped_samples
        }

    return result


# ============================================================================
# MODE 3: Self-Reflection (Single Agent)
# ============================================================================

def evaluate_self_reflection(
    agent_name: str,
    dataset: str = 'mmlu-pro',
    data_category: str = 'business',
    gpqa_split: str = 'gpqa_main',
    use_subset: bool = False,
    max_tokens: int = 4096,
    num_reflections: int = 3,
    num_samples: int = 50,
    start_index: int = 0,
    load_file: Optional[str] = None
):
    """
    Evaluate single agent with self-reflection across k rounds.
    Agent iteratively reflects on its own response before giving final answer.

    Args:
        agent_name: Name of agent to use
        dataset: Dataset to use ('mmlu-pro' or 'gpqa')
        data_category: Category for MMLU-Pro
        gpqa_split: Split for GPQA
        use_subset: Whether to use subset of data
        max_tokens: Maximum tokens for generation
        num_reflections: Number of reflection rounds (k)
        num_samples: Number of test samples
        start_index: Starting index for sampling
        load_file: Path to previous results for continuation

    Returns:
        Dictionary with evaluation results showing round-by-round improvement
    """
    logger.info("\n" + "="*80)
    logger.info(f"MODE 3: SELF-REFLECTION - {agent_name}")
    logger.info("="*80)
    logger.info(f"Reflection Rounds: {num_reflections}")

    # Check for continuation
    previous_results = None
    evaluated_ids = set()

    if load_file:
        previous_results = load_results(load_file)
        if previous_results:
            # Validate compatibility
            if previous_results.get('mode') != 'self-reflection':
                logger.error(f"Cannot continue: previous mode was '{previous_results.get('mode')}', expected 'self-reflection'")
                return None
            if previous_results.get('agent') != agent_name:
                logger.error(f"Cannot continue: previous agent was '{previous_results.get('agent')}', expected '{agent_name}'")
                return None
            if previous_results.get('num_reflections') != num_reflections:
                logger.error(f"Cannot continue: previous num_reflections was {previous_results.get('num_reflections')}, expected {num_reflections}")
                return None

            # Get already evaluated question IDs from first reflection
            evaluated_ids = {r['question_id'] for r in previous_results.get('results_per_reflection', [[]])[0]}
            logger.info(f"✓ Continuing from previous run with {len(evaluated_ids)} evaluated samples")

    # Load test data
    test_data = load_evaluation_data(
        dataset=dataset,
        category=data_category,
        gpqa_split=gpqa_split,
        use_subset=use_subset,
        num_samples=num_samples,
        start_index=start_index
    )

    if not test_data:
        logger.warning("No test data found!")
        return None

    # Track results per reflection
    results_per_reflection = [[] for _ in range(num_reflections)]
    correct_per_reflection = [0 for _ in range(num_reflections)]
    skipped_count = 0

    logger.info(f"\nEvaluating on {len(test_data)} samples...")
    if evaluated_ids:
        logger.info(f"Skipping {len(evaluated_ids)} already-evaluated samples")

    for idx, example in enumerate(test_data):
        question_id = example.get('question_id', idx)

        # Skip if already evaluated
        if question_id in evaluated_ids:
            skipped_count += 1
            continue

        question = example['question']
        options = example['options']
        correct_answer = example['answer']
        option_letters = [chr(65+i) for i in range(len(options))]

        formatted_question = format_question_with_options(question, options)

        # Initialize manager for this question
        manager = MultiAgentDebateManager(
            agents=[agent_name],
            options=option_letters,
            num_rounds=1,
            max_tokens=max_tokens
        )

        try:
            # Track reflection history
            reflection_history = []
            previous_response = None

            # Define format instructions (shared across all reflections)
            format_instructions = """
You MUST respond in the following format:
My confidence distribution:
A: [probability between 0 and 1]
B: [probability between 0 and 1]
C: [probability between 0 and 1]
...

The probabilities must sum to 1.0.

Example format:
<answer>
My confidence distribution:
A: 0.15
B: 0.60
C: 0.20
D: 0.05
</answer>

Note: If you revise your confidence during reasoning, please restate the full distribution in requested format at the end.
Now provide your response:"""

            # Run self-reflection rounds
            for reflection_idx in range(num_reflections):
                if reflection_idx == 0:
                    # Initial response - no reflection yet
                    initial_prompt = f"""{formatted_question}

Please analyze the question carefully and provide your confidence level for each answer option.
{format_instructions}"""

                    logits, response_text = manager.extract_logits(
                        agent_id=agent_name,
                        query=initial_prompt,
                        round_num=0,
                        history=[],
                        use_raw_prompt=True  # Use our custom prompt without debate wrapper
                    )
                else:
                    # Build reflection prompt
                    reflection_prompt = f"""{formatted_question}

You previously answered this question. Please review your previous response and reflect on whether your answer is correct.

Previous Response:
{previous_response}

Now, please:
1. Carefully reconsider your previous answer
2. Think about any potential errors or oversights
3. Consider alternative options if appropriate
4. Provide an updated confidence distribution
{format_instructions}"""

                    # Get reflected response
                    logits, response_text = manager.extract_logits(
                        agent_id=agent_name,
                        query=reflection_prompt,
                        round_num=reflection_idx,
                        history=[],
                        use_raw_prompt=True  # Use our custom reflection prompt without debate wrapper
                    )

                # Store response for next reflection
                previous_response = response_text

                # Get prediction (argmax)
                prediction = max(logits.items(), key=lambda x: x[1])[0]
                confidence = logits[prediction]

                logger.info(f"[{agent_name}] Question {question_id} | Reflection {reflection_idx + 1}: {logits}")

                # Check if correct
                is_correct = prediction == correct_answer
                if is_correct:
                    correct_per_reflection[reflection_idx] += 1

                # Store result for this reflection
                results_per_reflection[reflection_idx].append({
                    'question_id': question_id,
                    'reflection': reflection_idx + 1,
                    'prediction': prediction,
                    'correct_answer': correct_answer,
                    'is_correct': is_correct,
                    'confidence': confidence,
                    'logits': logits
                })

                reflection_history.append({
                    'reflection': reflection_idx + 1,
                    'prediction': prediction,
                    'confidence': confidence,
                    'is_correct': is_correct
                })

            logger.info(f"Question {idx} | Reflections: {[h['prediction'] for h in reflection_history]} | "
                       f"Correct: {correct_answer}")

            # Log progress
            if (idx + 1) % 10 == 0:
                logger.info(f"\nProgress: {idx + 1}/{len(test_data)}")
                for reflection_idx in range(num_reflections):
                    acc = correct_per_reflection[reflection_idx] / (idx + 1)
                    logger.info(f"  Reflection {reflection_idx + 1} Accuracy: {acc:.2%}")

        except Exception as e:
            logger.error(f"Error on question {idx}: {str(e)}")
            for reflection_idx in range(num_reflections):
                results_per_reflection[reflection_idx].append({
                    'question_id': question_id,
                    'reflection': reflection_idx + 1,
                    'prediction': 'ERROR',
                    'correct_answer': correct_answer,
                    'is_correct': False,
                    'confidence': 0.0,
                    'logits': {}
                })

    # Calculate final accuracies
    logger.info("\n" + "="*80)
    logger.info("RESULTS - REFLECTION-BY-REFLECTION ACCURACY")
    logger.info("="*80)

    logger.info(f"\n{'Reflection':<15} {'Accuracy':<15} {'Correct/Total':<20} {'Improvement':<15}")
    logger.info("-"*80)

    reflection_summaries = []
    baseline_accuracy = None

    for reflection_idx in range(num_reflections):
        accuracy = correct_per_reflection[reflection_idx] / len(test_data)

        # Calculate improvement from first reflection
        if reflection_idx == 0:
            baseline_accuracy = accuracy
            improvement = 0.0
        else:
            improvement = accuracy - baseline_accuracy

        logger.info(f"{reflection_idx + 1:<15} {accuracy:<15.2%} "
                   f"{correct_per_reflection[reflection_idx]}/{len(test_data):<17} "
                   f"{improvement:+.2%}")

        reflection_summaries.append({
            'reflection': reflection_idx + 1,
            'accuracy': accuracy,
            'correct': correct_per_reflection[reflection_idx],
            'total': len(test_data),
            'improvement_from_initial': improvement
        })

    # Additional statistics
    final_accuracy = reflection_summaries[-1]['accuracy']
    total_improvement = final_accuracy - baseline_accuracy

    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Initial Accuracy (Reflection 1): {baseline_accuracy:.2%}")
    logger.info(f"Final Accuracy (Reflection {num_reflections}): {final_accuracy:.2%}")
    logger.info(f"Total Improvement: {total_improvement:+.2%}")
    logger.info(f"Agent: {agent_name}")

    # Check if accuracy improved
    if total_improvement > 0:
        logger.info(f"\n✓ Self-reflection improved accuracy by {total_improvement:.2%}")
    elif total_improvement < 0:
        logger.info(f"\n✗ Self-reflection decreased accuracy by {abs(total_improvement):.2%}")
    else:
        logger.info(f"\n→ No change in accuracy across reflections")

    return {
        'mode': 'self-reflection',
        'agent': agent_name,
        'num_reflections': num_reflections,
        'reflection_summaries': reflection_summaries,
        'initial_accuracy': baseline_accuracy,
        'final_accuracy': final_accuracy,
        'total_improvement': total_improvement,
        'results_per_reflection': results_per_reflection
    }


def merge_self_reflection_results(old_results: Dict, new_results: Dict) -> Dict:
    """Merge old and new self-reflection results."""
    num_reflections = new_results['num_reflections']

    # Merge results per reflection
    merged_results_per_reflection = []
    for reflection_idx in range(num_reflections):
        old_reflection = old_results.get('results_per_reflection', [[]])[reflection_idx] if reflection_idx < len(old_results.get('results_per_reflection', [])) else []
        new_reflection = new_results['results_per_reflection'][reflection_idx]
        merged_results_per_reflection.append(old_reflection + new_reflection)

    # Recalculate statistics per reflection
    reflection_summaries = []
    for reflection_idx in range(num_reflections):
        reflection_results = merged_results_per_reflection[reflection_idx]
        correct = sum(1 for r in reflection_results if r['is_correct'])
        total = len(reflection_results)
        accuracy = correct / total if total > 0 else 0.0

        reflection_summaries.append({
            'reflection': reflection_idx + 1,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'improvement_from_initial': accuracy - reflection_summaries[0]['accuracy'] if reflection_idx > 0 else 0.0
        })

    baseline_accuracy = reflection_summaries[0]['accuracy']
    final_accuracy = reflection_summaries[-1]['accuracy']

    return {
        'mode': new_results['mode'],
        'agent': new_results['agent'],
        'num_reflections': num_reflections,
        'reflection_summaries': reflection_summaries,
        'initial_accuracy': baseline_accuracy,
        'final_accuracy': final_accuracy,
        'total_improvement': final_accuracy - baseline_accuracy,
        'results_per_reflection': merged_results_per_reflection
    }


# ============================================================================
# MODE 4: Multi-Agent Conformal Debate
# ============================================================================

def evaluate_debate_conformal(
    results_file: str,
    alpha: float = 0.1,
    weight_strategy: str = "uniform",
    score_type: str = "probability",
    calibration_indices: Optional[List[int]] = None,
    inference_indices: Optional[List[int]] = None,
    early_stop: bool = False,
    early_stop_set_size: int = 1
):
    """
    Evaluate multi-agent conformal prediction using existing debate results.
    
    Args:
        results_file: Path to existing debate results JSON file (e.g., 'results/example.json')
        alpha: Miscoverage rate (default: 0.1 for 90% coverage)
        weight_strategy: Aggregation strategy ('uniform', 'entropy', 'consistency')
        score_type: Nonconformity score type ('probability' or 'ranking', default: 'probability')
        calibration_indices: List of example indices to use for calibration (None = second half)
        inference_indices: List of example indices to use for inference (None = first half)
        early_stop: Whether to stop inference when prediction set size reaches 1 (default: False)
    
    Returns:
        Dictionary with conformal evaluation results
    """
    logger.info("\n" + "="*80)
    logger.info(f"MODE 3: MULTI-AGENT CONFORMAL WITH EXISTING DEBATE RESULTS")
    logger.info("="*80)
    logger.info(f"Results file: {results_file}")
    logger.info(f"Alpha: {alpha} (Target coverage: {(1-alpha)*100:.0f}%)")
    logger.info(f"Weight strategy: {weight_strategy}")
    logger.info(f"Score type: {score_type}")
    
    # Load existing results
    existing_results = load_results(results_file)
    if not existing_results:
        logger.error(f"Failed to load results from {results_file}")
        return None
    
    # Validate that we have debate-majority results
    if existing_results.get('mode') != 'debate-majority':
        logger.error(f"Expected 'debate-majority' mode, got '{existing_results.get('mode')}'")
        return None
    
    agent_names = existing_results['agents']
    num_rounds = existing_results['num_rounds']
    raw_results = existing_results['results_per_round']
    
    # Reorganize results by round if needed
    # Check if results are in flat format (single list) or nested format (list of lists)
    if len(raw_results) == 1 and isinstance(raw_results[0], list):
        # Flat format - reorganize by round number
        all_results = raw_results[0]
        results_per_round = [[] for _ in range(num_rounds)]
        
        # Group results by round
        for result in all_results:
            round_num = result.get('round', 1)
            results_per_round[round_num - 1].append(result)
    else:
        # Already in nested format
        results_per_round = raw_results
    
    logger.info(f"Agents: {', '.join(agent_names)}")
    logger.info(f"Debate Rounds: {num_rounds}")
    
    # Get total unique examples from first round
    unique_question_ids = set(r['question_id'] for r in results_per_round[0])
    total_examples = len(unique_question_ids)
    logger.info(f"Total examples in file: {total_examples}")
    
    # # Determine calibration and inference splits
    # calibration_indices = list(range(0, total_examples))
    # inference_indices = list(range(0, total_examples)) 
    
    # Comment only for debugging
    if calibration_indices is None and inference_indices is None:
        # Default: use 1st half for calibration, second half for inference
        split_point = total_examples // 2
        calibration_indices = list(range(0, split_point))
        inference_indices = list(range(split_point, total_examples)) 
    elif calibration_indices is None:
        # Use remaining indices for calibration
        calibration_indices = [i for i in range(total_examples) if i not in inference_indices]
    elif inference_indices is None:
        # Use remaining indices for inference
        inference_indices = [i for i in range(total_examples) if i not in calibration_indices]
    
    logger.info(f"Calibration set: {len(calibration_indices)} examples")
    logger.info(f"Inference set: {len(inference_indices)} examples")
    
    # Phase 1: Extract calibration data from existing results
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: EXTRACT CALIBRATION DATA")
    logger.info("="*80)
    
    calibration_data_per_round = [[] for _ in range(num_rounds)]
    
    for idx in calibration_indices:
        for round_idx in range(num_rounds):
            result = results_per_round[round_idx][idx]
            
            # Get agent logits and reconstruct AgentResponse objects
            agent_logits = result.get('agent_logits', {})
            correct_answer = result['correct_answer']
            
            # Get option letters from one of the agent's logits
            option_letters = list(next(iter(agent_logits.values())).keys())
            
            # Create AgentResponse objects
            agent_responses = []
            for agent_name in agent_names:
                if agent_name in agent_logits:
                    logits = agent_logits[agent_name]
                    # Create a dummy AgentResponse
                    response = AgentResponse(
                        agent_id=agent_name,
                        round_num=round_idx,
                        logits=logits,
                        response_text=""
                    )
                    agent_responses.append(response)
            
            # Compute social probabilities
            aggregator = SocialWelfareAggregator(
                options=option_letters,
                weight_strategy=weight_strategy
            )
            social_prob = aggregator.compute_social_scores(agent_responses)
            
            calibration_data_per_round[round_idx].append((social_prob, correct_answer))
            # logger.info(f'{Calibration data: len(calibration_data_per_round)}')
    
    logger.info(f"✓ Extracted calibration data for {num_rounds} rounds")
    
    # Phase 2: Conformal Calibration per Round
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: CONFORMAL CALIBRATION PER ROUND")
    logger.info("="*80)
    
    calibrators = []
    for round_idx in range(num_rounds):
        calibrator = ConformalCalibrator(alpha=alpha, score_type=score_type)
        calibrator.calibrate(calibration_data_per_round[round_idx])
        calibrators.append(calibrator)
        logger.info(f"Round {round_idx + 1}: q̂ = {calibrator.q_hat:.4f}")
    
    # Phase 3: Inference with existing results
    logger.info("\n" + "="*80)
    if early_stop:
        logger.info("PHASE 3: INFERENCE WITH EARLY STOPPING")
        logger.info("="*80)
        logger.info("Early stopping enabled: Stop when prediction set size reaches 1")
        logger.info("Accuracy calculated using predictions from stopping round for each sample")
    else:
        logger.info("PHASE 3: INFERENCE WITHOUT EARLY STOPPING")
        logger.info("="*80)
        logger.info("Early stopping disabled: Processing all rounds")
        logger.info("Accuracy calculated per round using only samples that reached that round")
    
    inference_results_per_round = [[] for _ in range(num_rounds)]
    early_stop_counts = [0 for _ in range(num_rounds)]  # Track stopping rounds
    sample_stopping_info = {}  # Track each sample's stopping round and result

    for idx in inference_indices:
        stopped = False
        stopping_round = num_rounds  # Default to last round if no early stop
        sample_results = []  # Store all rounds for this sample

        # Process rounds sequentially
        for round_idx in range(num_rounds):
            # Skip if already stopped (only applies when early_stop is True)
            if early_stop and stopped:
                break

            result = results_per_round[round_idx][idx]

            # Get agent logits
            agent_logits = result.get('agent_logits', {})
            correct_answer = result['correct_answer']
            question_id = result.get('question_id', idx)

            # Get option letters
            option_letters = list(next(iter(agent_logits.values())).keys())

            # Create AgentResponse objects
            agent_responses = []
            for agent_name in agent_names:
                if agent_name in agent_logits:
                    logits = agent_logits[agent_name]
                    response = AgentResponse(
                        agent_id=agent_name,
                        round_num=round_idx,
                        logits=logits,
                        response_text=""
                    )
                    agent_responses.append(response)

            # Compute social probabilities
            aggregator = SocialWelfareAggregator(
                options=option_letters,
                weight_strategy=weight_strategy
            )
            social_prob = aggregator.compute_social_scores(agent_responses)

            # Get conformal prediction set
            prediction_set = calibrators[round_idx].get_prediction_set(social_prob)
            covered = correct_answer in prediction_set
            set_size = len(prediction_set)

            # Greedy prediction
            greedy_pred = max(social_prob.items(), key=lambda x: x[1])[0]
            greedy_correct = greedy_pred == correct_answer

            # Calculate rank of ground truth label
            # Sort options by probability in descending order
            sorted_probs = sorted(social_prob.items(), key=lambda x: x[1], reverse=True)
            gt_rank = next((idx + 1 for idx, (opt, _) in enumerate(sorted_probs) if opt == correct_answer), len(sorted_probs))

            # Store result for this round
            round_result = {
                'question_id': question_id,
                'prediction_set': prediction_set,
                'set_size': set_size,
                'covered': covered,
                'correct_answer': correct_answer,
                'greedy_pred': greedy_pred,
                'greedy_correct': greedy_correct,
                'social_prob': social_prob,
                'gt_rank': gt_rank,
                'stopped_here': False  # Will update if this is stopping round
            }
            sample_results.append(round_result)
            inference_results_per_round[round_idx].append(round_result)

            # Early stopping: if set size is 1, stop here
            if early_stop and set_size == early_stop_set_size:
                stopped = True
                stopping_round = round_idx + 1
                early_stop_counts[round_idx] += 1
                # Mark this as the stopping round
                round_result['stopped_here'] = True
                logger.debug(f"Sample {question_id}: Early stop at round {stopping_round} (set size = 1)")
                break

        # Track stopping info for this sample
        sample_stopping_info[question_id] = {
            'stopping_round': stopping_round - 1,  # 0-indexed
            'results': sample_results
        }

        # Log progress
        if (idx + 1) % 50 == 0:
            logger.info(f"Progress: {idx + 1}/{len(inference_indices)} samples processed")
    
    # Phase 4: Results per Round with Early Stopping Analysis
    logger.info("\n" + "="*80)
    logger.info("RESULTS - CROSS-ROUND COMPARISON WITH EARLY STOPPING")
    logger.info("="*80)

    # Calculate early stopping statistics
    total_samples = len(inference_indices)
    samples_stopped_early = sum(early_stop_counts)
    samples_reached_final_round = total_samples - samples_stopped_early

    if early_stop:
        logger.info(f"\n📊 Early Stopping Statistics:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Stopped early: {samples_stopped_early} ({samples_stopped_early/total_samples:.1%})")
        logger.info(f"  Reached final round: {samples_reached_final_round} ({samples_reached_final_round/total_samples:.1%})")

    # Header with conditional column for early stopping
    if early_stop:
        logger.info(f"\n{'Round':<10} {'Coverage':<12} {'Avg Set':<12} {'Accuracy':<12} {'Stopped Here':<15} {'Stop Acc':<12}")
    else:
        logger.info(f"\n{'Round':<10} {'Coverage':<12} {'Avg Set':<12} {'Accuracy':<12} {'Stopped Here':<15}")
    logger.info("-"*80)

    round_results = []
    for round_idx in range(num_rounds):
        if early_stop:
            # With early stopping: for each round, use results from samples' stopping round or current round
            covered_count = 0
            correct_count = 0
            set_sizes = []
            gt_ranks = []

            # Calculate accuracy for only samples that stopped at this round
            stopped_at_this_round_correct = 0
            stopped_at_this_round_total = 0

            for question_id, info in sample_stopping_info.items():
                stopping_round_idx = info['stopping_round']
                results_list = info['results']

                # Use result from stopping round if stopped before current round, else use current round
                result_idx = min(stopping_round_idx, round_idx)
                if result_idx < len(results_list):
                    result = results_list[result_idx]
                    covered_count += int(result['covered'])
                    correct_count += int(result['greedy_correct'])
                    set_sizes.append(result['set_size'])
                    gt_ranks.append(result['gt_rank'])

                    # Track accuracy for samples that stopped at this specific round
                    if stopping_round_idx == round_idx and result.get('stopped_here', False):
                        stopped_at_this_round_total += 1
                        if result['greedy_correct']:
                            stopped_at_this_round_correct += 1

            coverage = covered_count / total_samples if total_samples > 0 else 0
            avg_set_size = np.mean(set_sizes) if set_sizes else 0
            accuracy = correct_count / total_samples if total_samples > 0 else 0
            avg_gt_rank = np.mean(gt_ranks) if gt_ranks else 0
            num_samples = total_samples

            # Calculate early stopped accuracy
            early_stopped_accuracy = stopped_at_this_round_correct / stopped_at_this_round_total if stopped_at_this_round_total > 0 else None
        else:
            # Without early stopping: use only samples that reached this round
            results = inference_results_per_round[round_idx]
            coverage = sum(r['covered'] for r in results) / len(results) if results else 0
            avg_set_size = np.mean([r['set_size'] for r in results]) if results else 0
            accuracy = sum(r['greedy_correct'] for r in results) / len(results) if results else 0
            avg_gt_rank = np.mean([r['gt_rank'] for r in results]) if results else 0
            num_samples = len(results)
            early_stopped_accuracy = None

        stopped_count = early_stop_counts[round_idx]
        stopped_pct = stopped_count / total_samples if total_samples > 0 else 0

        # Log with conditional early stopping accuracy column
        if early_stop:
            stop_acc_str = f"{early_stopped_accuracy:.2%}" if early_stopped_accuracy is not None else "N/A"
            logger.info(f"{round_idx + 1:<10} {coverage:<12.2%} {avg_set_size:<12.2f} {accuracy:<12.2%} "
                       f"{stopped_count} ({stopped_pct:.1%})  {stop_acc_str:<12}")
        else:
            logger.info(f"{round_idx + 1:<10} {coverage:<12.2%} {avg_set_size:<12.2f} {accuracy:<12.2%} "
                       f"{stopped_count} ({stopped_pct:.1%})")

        round_results.append({
            'round': round_idx + 1,
            'coverage': coverage,
            'avg_set_size': avg_set_size,
            'accuracy': accuracy,
            'avg_gt_rank': avg_gt_rank,
            'early_stopped_accuracy': early_stopped_accuracy,
            'num_samples': num_samples,
            'early_stop_count': stopped_count,
            'early_stop_rate': stopped_pct
        })
    
    # Calculate average stopping round (weighted by number of stops at each round)
    avg_stopping_round = sum((round_idx + 1) * count for round_idx, count in enumerate(early_stop_counts)) / samples_stopped_early if samples_stopped_early > 0 else num_rounds
    
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Nonconformity score type: {score_type}")
    logger.info(f"Target coverage: {(1-alpha)*100:.0f}%")
    logger.info(f"Round 1 coverage: {round_results[0]['coverage']:.2%}")
    logger.info(f"Round {num_rounds} coverage: {round_results[-1]['coverage']:.2%}")
    logger.info(f"Coverage improvement: {(round_results[-1]['coverage'] - round_results[0]['coverage']):.2%}")
    logger.info(f"Round 1 avg set size: {round_results[0]['avg_set_size']:.2f}")
    logger.info(f"Round {num_rounds} avg set size: {round_results[-1]['avg_set_size']:.2f}")
    logger.info(f"Round 1 avg GT rank: {round_results[0]['avg_gt_rank']:.2f}")
    logger.info(f"Round {num_rounds} avg GT rank: {round_results[-1]['avg_gt_rank']:.2f}")
    logger.info(f"GT rank improvement: {(round_results[0]['avg_gt_rank'] - round_results[-1]['avg_gt_rank']):.2f}")

    # Calculate overall early stopped accuracy (all samples that stopped early)
    overall_early_stopped_acc = None
    total_early_stopped_correct = 0
    total_early_stopped_samples = 0

    if early_stop:
        logger.info(f"\n🎯 Early Stopping Efficiency:")
        logger.info(f"  Average stopping round: {avg_stopping_round:.2f}")
        logger.info(f"  Samples stopped early: {samples_stopped_early}/{total_samples} ({samples_stopped_early/total_samples:.1%})")
        logger.info(f"  Potential rounds saved: {sum((num_rounds - (round_idx + 1)) * count for round_idx, count in enumerate(early_stop_counts))}")

        for question_id, info in sample_stopping_info.items():
            stopping_round_idx = info['stopping_round']
            # Only count samples that stopped before the final round
            if stopping_round_idx < num_rounds - 1:
                results_list = info['results']
                if stopping_round_idx < len(results_list):
                    result = results_list[stopping_round_idx]
                    if result.get('stopped_here', False):
                        total_early_stopped_samples += 1
                        if result['greedy_correct']:
                            total_early_stopped_correct += 1

        if total_early_stopped_samples > 0:
            overall_early_stopped_acc = total_early_stopped_correct / total_early_stopped_samples
            logger.info(f"  Overall early stopped accuracy: {overall_early_stopped_acc:.2%} ({total_early_stopped_correct}/{total_early_stopped_samples})")

            # Compare to final round accuracy
            if round_results[-1]['accuracy'] is not None:
                early_vs_final = overall_early_stopped_acc - round_results[-1]['accuracy']
                if early_vs_final > 0:
                    logger.info(f"  Early stopped samples are {early_vs_final:.2%} MORE accurate than final round")
                elif early_vs_final < 0:
                    logger.info(f"  Early stopped samples are {abs(early_vs_final):.2%} LESS accurate than final round")
                else:
                    logger.info(f"  Early stopped samples have SAME accuracy as final round")
    else:
        logger.info(f"\n🎯 Early Stopping Efficiency:")
        logger.info(f"  Average stopping round: {avg_stopping_round:.2f}")
        logger.info(f"  Samples stopped early: {samples_stopped_early}/{total_samples} ({samples_stopped_early/total_samples:.1%})")
        logger.info(f"  Potential rounds saved: {sum((num_rounds - (round_idx + 1)) * count for round_idx, count in enumerate(early_stop_counts))}")
    
    return {
        'mode': 'debate-conformal',
        'source_file': results_file,
        'agents': agent_names,
        'num_rounds': num_rounds,
        'alpha': alpha,
        'weight_strategy': weight_strategy,
        'score_type': score_type,
        'calibration_size': len(calibration_indices),
        'inference_size': len(inference_indices),
        'round_results': round_results,
        'inference_results_per_round': inference_results_per_round,
        'final_coverage': round_results[-1]['coverage'],
        'final_accuracy': round_results[-1]['accuracy'],
        'final_set_size': round_results[-1]['avg_set_size'],
        'final_avg_gt_rank': round_results[-1]['avg_gt_rank'],
        'early_stopping': {
            'total_samples': total_samples,
            'stopped_early_count': samples_stopped_early,
            'stopped_early_rate': samples_stopped_early / total_samples if total_samples > 0 else 0,
            'avg_stopping_round': avg_stopping_round,
            'stop_counts_per_round': early_stop_counts,
            'potential_rounds_saved': sum((num_rounds - (round_idx + 1)) * count for round_idx, count in enumerate(early_stop_counts)),
            'overall_early_stopped_accuracy': overall_early_stopped_acc,
            'early_stopped_correct': total_early_stopped_correct,
            'early_stopped_total': total_early_stopped_samples
        }
    }


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified Evaluation for Conformal Social Choice Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Greedy single agent on MMLU-Pro
  python run_evaluation.py --mode greedy --agent qwen --dataset mmlu-pro --category math --num-samples 100

  # Greedy single agent on GPQA
  python run_evaluation.py --mode greedy --agent qwen --dataset gpqa --gpqa-split gpqa_main --num-samples 50

  # Self-reflection (single agent with k reflections)
  python run_evaluation.py --mode self-reflection --agent qwen --dataset mmlu-pro --rounds 3 --num-samples 50

  # Self-reflection on GPQA
  python run_evaluation.py --mode self-reflection --agent claude-haiku --dataset gpqa --gpqa-split gpqa_diamond --rounds 3

  # Multi-round debate with majority voting
  python run_evaluation.py --mode debate-majority --agents qwen claude-haiku nova-lite --dataset mmlu-pro --rounds 3 --num-samples 50

  # ARC examples
  python run_evaluation.py --mode greedy --agent claude-haiku --dataset arc --gpqa-split ARC-Challenge --num-samples 100
  python run_evaluation.py --mode debate-majority --agents qwen claude-haiku nova-lite --dataset arc --gpqa-split ARC-Easy --rounds 3

  # MuSR examples
  python run_evaluation.py --mode greedy --agent claude-haiku --dataset musr --gpqa-split object_placements --num-samples 50
  python run_evaluation.py --mode debate-majority --agents qwen claude-haiku nova-lite --dataset musr --gpqa-split murder_mysteries --rounds 3

  # Multi-agent conformal with debate
  python run_evaluation.py --mode debate-conformal --results-file results/debate_majority_results.json
        """
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=4096,
        help='Maximum tokens for generation (default: 4096)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['greedy', 'self-reflection', 'debate-majority', 'debate-conformal'],
        help='Evaluation mode'
    )
    
    parser.add_argument(
        '--agent',
        type=str,
        default='claude-haiku',
        choices=['deepseek-r1', 'nova-lite', 'nova-pro', 'claude-haiku', 'qwen-32b', 'qwen-80b'],
        help='Single agent name (for greedy mode)'
    )
    
    parser.add_argument(
        '--agents',
        type=str,
        nargs='+',
        default=['qwen-32b', 'claude-haiku', 'nova-lite'],
        help='List of agent names (for majority/conformal/debate modes)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='mmlu-pro',
        choices=['mmlu-pro', 'gpqa', 'negotiation', 'cuad', 'arc', 'musr'],
        help='Dataset to use for evaluation (default: mmlu-pro)'
    )

    parser.add_argument(
        '--category',
        type=str,
        default='math',
        help='Data category for MMLU-Pro (default: math). For negotiation: buyer/seller.'
    )

    parser.add_argument(
        '--gpqa-split',
        type=str,
        default='gpqa_main',
        choices=['gpqa_main', 'gpqa_diamond', 'gpqa_extended', 'ARC-Challenge', 'ARC-Easy',
                 'murder_mysteries', 'object_placements', 'team_allocation'],
        help='Split/subset: GPQA (gpqa_main/gpqa_diamond/gpqa_extended), ARC (ARC-Challenge/ARC-Easy), MuSR (murder_mysteries/object_placements/team_allocation)'
    )
    
    parser.add_argument(
        '--use-subset',
        action='store_true',
        help='Use a subset of the data instead of entire dataset (default: False)'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of samples when use-subset is enabled (default: 100)'
    )
    
    parser.add_argument(
        '--calibration-start',
        type=int,
        default=50,
        help='Calibration set start index (default: 50)'
    )
    
    parser.add_argument(
        '--calibration-end',
        type=int,
        default=100,
        help='Calibration set end index (default: 100)'
    )
    
    parser.add_argument(
        '--inference-start',
        type=int,
        default=0,
        help='Inference set start index (default: 0)'
    )
    
    parser.add_argument(
        '--inference-end',
        type=int,
        default=50,
        help='Inference set end index (default: 50)'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.1,
        help='Miscoverage rate for conformal prediction (default: 0.1)'
    )
    
    parser.add_argument(
        '--early-stop-set-size',
        type=int,
        default=1,
        help='Early stopping set size (default: 1)'
    )
    
    parser.add_argument(
        '--rounds',
        type=int,
        default=3,
        help='Number of debate rounds or reflection iterations (default: 3)'
    )
    
    parser.add_argument(
        '--weight-strategy',
        type=str,
        default='uniform',
        choices=['uniform', 'entropy', 'consistency'],
        help='Aggregation weight strategy (default: uniform)'
    )
    
    parser.add_argument(
        '--score-type',
        type=str,
        default='probability',
        choices=['probability', 'ranking'],
        help='Nonconformity score type for conformal prediction (default: probability)'
    )
    
    parser.add_argument(
        '--save-file',
        type=str,
        default=None,
        help='Path to save results JSON file (default: None)'
    )
    
    parser.add_argument(
        '--load-file',
        type=str,
        default=None,
        help='Path to load previous results JSON file for continuation (default: None)'
    )
    
    parser.add_argument(
        '--start-index',
        type=int,
        default=0,
        help='Starting sample index for continuation (default: 0)'
    )
    
    parser.add_argument(
        '--results-file',
        type=str,
        default=None,
        help='Path to existing debate results JSON file (for debate-conformal mode)'
    )
    
    parser.add_argument(
        '--early-stop',
        action='store_true',
        help='Enable early stopping: for debate-conformal (stop when prediction set size=1), for debate-majority (stop when all agents agree), default: False'
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("CONFORMAL SOCIAL CHOICE - UNIFIED EVALUATION")
    logger.info("="*80)
    logger.info(f"\nMode: {args.mode.upper()}")
    
    result = None
    
    if args.mode == 'greedy':
        if not args.save_file:
            args.save_file = f'results/greedy_{args.agent}_{args.num_samples}_results.json'
        
        # Run evaluation (with continuation if load_file specified)
        new_result = evaluate_greedy_single_agent(
            args.agent,
            args.dataset,
            args.category,
            args.gpqa_split,
            args.use_subset,
            args.num_samples,
            args.max_tokens,
            args.start_index,
            args.load_file
        )
        
        # Merge with previous results if continuing
        if args.load_file and new_result:
            previous_results = load_results(args.load_file)
            if previous_results:
                result = merge_greedy_results(previous_results, new_result)
                logger.info(f"\n✓ Merged results: {result['total']} total samples, {result['accuracy']:.2%} accuracy")
            else:
                result = new_result
        else:
            result = new_result
        
        # Save results if save_file specified
        if args.save_file and result:
            save_results(result, args.save_file)

    elif args.mode == 'self-reflection':
        if not args.save_file:
            args.save_file = f'results/self_reflection_{args.agent}_{args.rounds}reflections_{args.num_samples}_results.json'

        # Run evaluation (with continuation if load_file specified)
        new_result = evaluate_self_reflection(
            args.agent,
            args.dataset,
            args.category,
            args.gpqa_split,
            args.use_subset,
            args.max_tokens,
            args.rounds,  # Reuse rounds parameter for number of reflections
            args.num_samples,
            args.start_index,
            args.load_file
        )

        # Merge with previous results if continuing
        if args.load_file and new_result:
            previous_results = load_results(args.load_file)
            if previous_results:
                result = merge_self_reflection_results(previous_results, new_result)
                logger.info(f"\n✓ Merged results: Final accuracy: {result['final_accuracy']:.2%}, Total improvement: {result['total_improvement']:+.2%}")
            else:
                result = new_result
        else:
            result = new_result

        # Save results if save_file specified
        if args.save_file and result:
            save_results(result, args.save_file)

    elif args.mode == 'debate-majority':
        if not args.save_file:
            agents_str = '_'.join(args.agents)
            early_stop_suffix = '_early_stop' if args.early_stop else ''
            args.save_file = f'results/debate_majority_{agents_str}_{args.rounds}rounds_{args.num_samples}{early_stop_suffix}_results.json'
        
        # Run evaluation (with continuation if load_file specified)
        new_result = evaluate_debate_majority_voting(
            args.agents,
            args.dataset,
            args.category,
            args.gpqa_split,
            args.use_subset,
            args.max_tokens,
            args.rounds,
            args.num_samples,
            args.start_index,
            args.load_file,
            args.early_stop
        )
        
        # Merge with previous results if continuing
        if args.load_file and new_result:
            previous_results = load_results(args.load_file)
            if previous_results:
                result = merge_debate_majority_results(previous_results, new_result)
                logger.info(f"\n✓ Merged results: Final accuracy: {result['final_accuracy']:.2%}, Total improvement: {result['total_improvement']:+.2%}")
            else:
                result = new_result
        else:
            result = new_result
        
        # Save results if save_file specified
        if args.save_file and result:
            save_results(result, args.save_file)
        
    elif args.mode == 'debate-conformal':
        if not args.results_file:
            logger.error("--results-file is required for debate-conformal mode")
            return None
        
        result = evaluate_debate_conformal(
            results_file=args.results_file,
            alpha=args.alpha,
            weight_strategy=args.weight_strategy,
            score_type=args.score_type,
            early_stop=args.early_stop,
            early_stop_set_size=args.early_stop_set_size
        )
        
        # Save results if save_file specified
        if args.save_file and result:
            save_results(result, args.save_file)
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)
    
    return result


if __name__ == "__main__":
    main()
