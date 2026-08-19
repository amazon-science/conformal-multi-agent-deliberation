"""
Conformal Social Choice: A Framework for Safe Multi-Agent Debate
Implements the 4-stage pipeline from Section 2 of the paper.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import warnings
import boto3
import json
import re
import logging
import time
import random
from config import system_prompt
from utils import get_bedrock_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Changed to DEBUG to see all debug messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class AgentResponse:
    """Stores agent's probability distribution over options."""
    agent_id: str
    round_num: int
    logits: Dict[str, float]  # {option: probability}
    response_text: str = ""  # Full text response from the agent
    
    def __post_init__(self):
        # Validate probabilities sum to 1
        total = sum(self.logits.values())
        if not np.isclose(total, 1.0, atol=1e-3):
            warnings.warn(f"Agent {self.agent_id} probabilities sum to {total:.4f}")


@dataclass
class DebateInstance:
    """Represents a single debate across multiple rounds."""
    query_id: str
    ground_truth: Optional[str]
    debate_history: List[List[AgentResponse]]  # [round][agent]
    
    def get_round(self, round_idx: int) -> List[AgentResponse]:
        """Extract responses from a specific round."""
        return self.debate_history[round_idx]


# ============================================================================
# STEP 1: Multi-Agent Debate Manager
# ============================================================================

class MultiAgentDebateManager:
    """
    Manages the debate process and extracts logits from LLM agents.
    Uses AWS Bedrock API for actual LLM inference.
    """
    
    def __init__(self, agents: List[str], options: List[str], num_rounds: int = 3, max_tokens: int = 4096,
                 region_name: str = "us-east-1"):
        """
        Initialize debate manager with Bedrock client.
        
        Args:
            agents: List of agent identifiers (will be mapped to model IDs)
            options: List of answer options (e.g., ["A", "B", "C", "D"])
            num_rounds: Number of debate rounds
            region_name: AWS region for Bedrock
        """
        self.agents = agents
        self.options = options
        self.num_rounds = num_rounds
        self.max_tokens = max_tokens
        
        # Initialize Bedrock client
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name
        )
    
    def _build_prompt(self, query: str, options: List[str], round_num: int, 
                     history: List[str]) -> str:
        """
        Build prompt for verbalized probability elicitation.
        
        Args:
            query: The question to answer
            options: List of answer options
            round_num: Current round number
            history: Conversation history from previous round
        
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are participating in a multi-agent debate to answer the following question.

Question: {query}

Options:
"""
        for i, opt in enumerate(options):
            prompt += f"{chr(65+i)}. {opt}\n"
        
        if history and round_num > 0:
            prompt += f"\n{'='*80}\nPREVIOUS DEBATE ROUND - Please review and consider these carefully:\n{'='*80}\n"
            for h in history:
                prompt += f"{h}\n"
            prompt += f"\n{'='*80}\n"
            prompt += f"""
INSTRUCTIONS FOR ROUND {round_num + 1}:
1. Review the arguments and confidence distributions from other agents in previous round
2. Consider which arguments are most convincing and supported by evidence
3. Identify any consensus emerging among agents or important disagreements
4. Update your own confidence distribution based on:
   - Your initial analysis of the question
   - Strong arguments made by other agents
   - Areas where multiple agents agree or disagree
5. Be open to changing your opinion if presented with compelling reasoning

Now, please provide your updated confidence level for each answer option.
"""
        else:
            prompt += f"""
Please analyze the question carefully and provide your confidence level for each answer option.
"""
        
        prompt += f"""

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
        
        return prompt
    
    def _parse_probabilities(self, response_text: str, options: List[str]) -> Dict[str, float]:
        r"""
        Parse verbalized probabilities from model response, prioritizing <answer> tag content.
        Handles special characters like **, ~, \, etc.
        
        Args:
            response_text: Raw text response from model
            options: List of valid options
        
        Returns:
            Dictionary mapping options to probabilities
        """
        # Step 1: Try to extract content from <answer> tags first
        answer_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
        if answer_match:
            # Use only the answer section for parsing
            text_to_parse = answer_match.group(1).strip()
            logger.debug(f"Extracted answer section: {text_to_parse[:200]}...")
        else:
            # Fallback to full response text
            text_to_parse = response_text
            logger.debug("No <answer> tag found, using full response text")
        
        probabilities = {}
        
        # Step 2: Parse probabilities from the extracted text
        # Pattern: "A: 0.15" or "A = 0.15" or "Option A: 0.15"
        # Also handles special chars like "A.: 0.15", "A: **0.95**", "A: 0~.85", "A: 0\.00"
        for i, opt in enumerate(options):
            option_letter = chr(65 + i)  # A, B, C, D, etc.
            
            # Multiple regex patterns to catch different formats
            # Pattern captures everything after the colon/equals, then we clean it
            patterns = [
                rf'{option_letter}\.?\s*:\s*([^\n\r]+)',  # Handles "A:" or "A.:"
                rf'{option_letter}\.?\s*=\s*([^\n\r]+)',  # Handles "A=" or "A.="
                rf'Option\s+{option_letter}\.?\s*:\s*([^\n\r]+)',  # Handles "Option A:" or "Option A.:"
            ]
            
            found = False
            for pattern in patterns:
                matches = re.findall(pattern, text_to_parse, re.IGNORECASE)
                if matches:
                    try:
                        # Use the LAST match to get the most recent/updated confidence
                        raw_value = matches[-1].strip()
                        
                        # Extract numeric value by removing special characters
                        # Keep only digits, decimal point, and handle scientific notation
                        cleaned_value = re.sub(r'[^\d.]', '', raw_value)
                        
                        # If we have a valid number after cleaning
                        if cleaned_value and cleaned_value != '.':
                            # Handle cases like "0." or ".0" 
                            if cleaned_value.startswith('.'):
                                cleaned_value = '0' + cleaned_value
                            if cleaned_value.endswith('.'):
                                cleaned_value = cleaned_value + '0'
                            
                            prob = float(cleaned_value)
                            probabilities[option_letter] = prob
                            found = True
                            break
                    except (ValueError, AttributeError):
                        continue
            
            if not found:
                # Default to small uniform probability if not found
                probabilities[option_letter] = 0.01
        
        # Normalize to ensure sum = 1.0
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v / total for k, v in probabilities.items()}
        else:
            # Uniform distribution if all failed
            uniform_prob = 1.0 / len(options)
            probabilities = {chr(65 + i): uniform_prob for i in range(len(options))}
        
        return probabilities
    
    def extract_logits(self, agent_id: str, query: str,
                       round_num: int, history: List[str],
                       use_raw_prompt: bool = False) -> Tuple[Dict[str, float], str]:
        """
        Extract probability distribution from agent response using Bedrock API.

        Args:
            agent_id: Agent identifier (mapped to model ID)
            query: Question with options
            round_num: Current debate round
            history: Conversation history from previous rounds
            use_raw_prompt: If True, use query as-is without building debate prompt (default: False)

        Returns:
            Tuple of (probabilities dict, response text)
        """
        # Build prompt for verbalized probabilities (unless raw prompt is requested)
        if use_raw_prompt:
            user_prompt = query
        else:
            user_prompt = self._build_prompt(query, self.options, round_num, history)
        
        # Invoke Bedrock model
        try:
            # Send request to bedrock using utility function
            max_retries = 3
            temperature = 0.7
            top_p = 1.0
            
            response_text = get_bedrock_response(
                system_prompt,
                user_prompt,
                self.max_tokens,
                temperature,
                top_p,
                max_retries,
                self.bedrock_runtime,
                agent_id,
                logger
            )
            
            # Handle both return formats: (text, in_tok, out_tok) or just text
            if isinstance(response_text, tuple):
                response_text = response_text[0]
            
            logger.info(f"[{agent_id}] Round {round_num + 1} response:\n{response_text}")
            
            # Parse probabilities from response
            probabilities = self._parse_probabilities(response_text, self.options)
            
            return probabilities, response_text
            
        except Exception as e:
            logger.error(f"Error extracting logits for {agent_id}: {str(e)}")
            # Return uniform distribution as fallback
            uniform_prob = 1.0 / len(self.options)
            return {chr(65 + i): uniform_prob for i in range(len(self.options))}, ""
    
    def run_debate(self, query: str, ground_truth: Optional[str] = None) -> DebateInstance:
        """Execute multi-round debate and collect agent distributions."""
        debate_history = []
        previous_round_summary = None
        
        #logger.info(f"Debate history for query_id={id(query)} =="*80)
        
        for round_num in range(self.num_rounds):
            round_responses = []
            round_summary = f"\n--- Round {round_num + 1} ---\n"
            
            # Pass only the previous round's history (empty list for round 0)
            history_to_pass = [previous_round_summary] if previous_round_summary is not None else []
            
            for agent_id in self.agents:
                # Get agent's distribution and response text
                logits, response_text = self.extract_logits(agent_id, 
                                                            query, 
                                                            round_num, 
                                                            history_to_pass)
                
                logger.info(f"[Debate] Round {round_num + 1} | Agent {agent_id} using previous round history: {len(history_to_pass) > 0}")
                
                response = AgentResponse(
                    agent_id=agent_id,
                    round_num=round_num,
                    logits=logits,
                    response_text=response_text
                )
                round_responses.append(response)
                
                # Add agent's response to round summary
                round_summary += f"\n[{agent_id}]:\n{response_text}\n"
            
            debate_history.append(round_responses)
            
            # Store only the current round summary for the next round
            previous_round_summary = round_summary
        
        return DebateInstance(
            query_id=f"query_{id(query)}",
            ground_truth=ground_truth,
            debate_history=debate_history
        )


# ============================================================================
# STEP 2: Social Welfare Scoring (Aggregation)
# ============================================================================

class SocialWelfareAggregator:
    """
    Aggregates individual agent beliefs using Weighted Borda Count.
    """
    
    def __init__(self, options: List[str], weight_strategy: str = "uniform",
                 entropy_lambda: float = 1.0):
        self.options = options
        self.weight_strategy = weight_strategy
        self.entropy_lambda = entropy_lambda
        self.agent_weights = {}  # {agent_id: weight}
    
    def _compute_weights(self, responses: List[AgentResponse]) -> Dict[str, float]:
        """
        Dynamically compute agent reliability weights.
        Strategies: uniform, entropy-based, consistency-based
        """
        if self.weight_strategy == "uniform":
            return {r.agent_id: 1.0 / len(responses) for r in responses}
        
        elif self.weight_strategy == "entropy":
            weights = {}
            for resp in responses:
                # Lower entropy = more confident = higher weight
                probs = np.array([resp.logits[opt] for opt in self.options])
                probs = np.clip(probs, 1e-10, 1.0)  # Avoid log(0)
                entropy = -np.sum(probs * np.log(probs))
                weights[resp.agent_id] = np.exp(-self.entropy_lambda * entropy)
            
            # Normalize
            total = sum(weights.values())
            return {k: v / total for k, v in weights.items()}
        
        elif self.weight_strategy == "consistency":
            # Weight by past performance (requires historical tracking)
            if not self.agent_weights:
                return {r.agent_id: 1.0 / len(responses) for r in responses}
            return self.agent_weights.copy()
        
        else:
            raise ValueError(f"Unknown weight strategy: {self.weight_strategy}")
    
    def compute_social_scores(self, responses: List[AgentResponse]) -> Dict[str, float]:
        """
        Aggregate agent distributions into collective social probability.
        
        Returns:
            social_prob: P_social(y|x) for each option y
        """
        weights = self._compute_weights(responses)
        social_prob = defaultdict(float)
        
        for response in responses:
            w = weights[response.agent_id]
            for option, prob in response.logits.items():
                social_prob[option] += w * prob
        
        # Ensure normalization
        total = sum(social_prob.values())
        social_prob = {k: v / total for k, v in social_prob.items()}
        
        return dict(social_prob)
    
    def update_agent_weights(self, agent_id: str, performance_score: float):
        """Update agent weight based on historical accuracy."""
        if agent_id not in self.agent_weights:
            self.agent_weights[agent_id] = 1.0
        
        # Exponential moving average
        alpha = 0.3
        self.agent_weights[agent_id] = (
            alpha * performance_score + (1 - alpha) * self.agent_weights[agent_id]
        )


# ============================================================================
# STEP 3: Conformal Calibration (Safety Layer)
# ============================================================================

class ConformalCalibrator:
    """
    Applies Split Conformal Prediction to provide coverage guarantees.
    Supports two nonconformity score types: 'probability' and 'ranking'.
    """
    
    def __init__(self, alpha: float = 0.05, score_type: str = 'probability'):
        """
        Args:
            alpha: Miscoverage rate (default 0.05 for 95% coverage)
            score_type: Type of nonconformity score ('probability' or 'ranking', default: 'probability')
        """
        self.alpha = alpha
        self.score_type = score_type
        self.q_hat = None  # Conformal threshold
        self.calibration_scores = []
        
        if score_type not in ['probability', 'ranking']:
            raise ValueError(f"score_type must be 'probability' or 'ranking', got '{score_type}'")
    
    @staticmethod
    def nonconformity_score_probability(social_prob: Dict[str, float], option: str) -> float:
        """
        Probability-based non-conformity measure: S_nc(x, y) = 1 - P_social(y|x)
        Higher score = less conforming = less likely
        """
        return 1.0 - social_prob.get(option, 0.0)
    
    @staticmethod
    def nonconformity_score_ranking(social_prob: Dict[str, float], option: str) -> float:
        """
        Ranking-based non-conformity measure: S_nc(x, y) = cumulative probability up to rank of y
        
        This is the sum of probabilities from rank 1 (highest) up to and including the rank of option y.
        - Higher-ranked options (with higher probabilities) have lower cumulative scores
        - Lower-ranked options require more cumulative mass to reach, thus higher scores
        - Higher score = less conforming = less likely
        
        Example: If sorted probs are {A: 0.6, B: 0.25, C: 0.1, D: 0.05}
        - Score(A) = 0.6
        - Score(B) = 0.6 + 0.25 = 0.85
        - Score(C) = 0.6 + 0.25 + 0.1 = 0.95
        - Score(D) = 0.6 + 0.25 + 0.1 + 0.05 = 1.0
        
        For tied probabilities, all get the same score equal to the cumulative 
        probability of all higher-ranked tiers plus the minimum mass from their tier.
        
        Example: {A: 0.4, B: 0.4, C: 0.2}
        - Score(A) = Score(B) = 0.4 (minimum to reach probability 0.4)
        - Score(C) = 1.0 (need all mass to reach probability 0.2)
        """
        
        option_prob = social_prob.get(option, 0.0)
        
        # Get unique probability values in descending order
        unique_probs = sorted(set(social_prob.values()), reverse=True)
        
        # Find the minimum cumulative mass needed to include the option's probability
        cumulative = 0.0
        for prob_tier in unique_probs:
            if prob_tier > option_prob:
                # Add mass from higher-ranked tiers
                tier_mass = sum(p for p in social_prob.values() if p == prob_tier)
                cumulative += tier_mass
            elif prob_tier == option_prob:
                # For the tier containing our option, add just the option's probability
                return cumulative + option_prob
        
        return 1.0  # Fallback
    
    def nonconformity_score(self, social_prob: Dict[str, float], option: str) -> float:
        """
        Compute non-conformity score using the selected score type.
        
        Args:
            social_prob: Dictionary of option probabilities
            option: The option to score
            
        Returns:
            Non-conformity score (higher = less conforming)
        """
        if self.score_type == 'probability':
            return self.nonconformity_score_probability(social_prob, option)
        elif self.score_type == 'ranking':
            return self.nonconformity_score_ranking(social_prob, option)
        else:
            raise ValueError(f"Unknown score_type: {self.score_type}")
    
    def calibrate(self, calibration_data: List[Tuple[Dict[str, float], str]]):
        """
        Calibrate threshold using held-out data.
        
        Args:
            calibration_data: List of (social_prob_dict, ground_truth_label)
        """
        scores = []
        
        for social_prob, true_label in calibration_data:
            # Compute non-conformity score for ground truth using selected score type
            score = self.nonconformity_score(social_prob, true_label)
            scores.append(score)
        
        self.calibration_scores = scores
        
        # Compute quantile with finite-sample correction
        n = len(scores)
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        quantile_level = min(1.0, quantile_level)
        
        self.q_hat = np.quantile(scores, quantile_level)
        
        logger.info(f"[Calibration] score_type={self.score_type}, n={n}, α={self.alpha}, q̂={self.q_hat:.4f}")
    
    def get_prediction_set(self, social_prob: Dict[str, float]) -> List[str]:
        """
        Construct conformal prediction set C(x).
        
        Returns:
            List of options with non-conformity scores ≤ q̂
        """
        if self.q_hat is None:
            raise ValueError("Calibrator not calibrated! Call calibrate() first.")
        
        prediction_set = []
        
        for option, prob in social_prob.items():
            score = self.nonconformity_score(social_prob, option)
            if score <= self.q_hat:
                prediction_set.append(option)
        
        return prediction_set
    
    def compute_coverage(self, test_data: List[Tuple[Dict[str, float], str]]) -> float:
        """Empirical coverage on test set."""
        coverage_count = 0
        
        for social_prob, true_label in test_data:
            pred_set = self.get_prediction_set(social_prob)
            if true_label in pred_set:
                coverage_count += 1
        
        return coverage_count / len(test_data)


# ============================================================================
# STEP 4: Action Policy & Main Framework
# ============================================================================

class ConformalSocialChoice:
    """
    Main framework integrating all components.
    """
    
    def __init__(self, agents: List[str], options: List[str], 
                 alpha: float = 0.05, weight_strategy: str = "uniform"):
        self.debate_manager = MultiAgentDebateManager(agents, options)
        self.aggregator = SocialWelfareAggregator(options, weight_strategy)
        self.calibrator = ConformalCalibrator(alpha)
        self.options = options
    
    def calibrate(self, calibration_queries: List[Tuple[str, str]], 
                  round_idx: int = -1):
        """
        Calibrate on labeled data.
        
        Args:
            calibration_queries: List of (query, ground_truth)
            round_idx: Which debate round to use (-1 = last round)
        """
        calibration_data = []
        
        for query, ground_truth in calibration_queries:
            # Run debate
            debate = self.debate_manager.run_debate(query, ground_truth)
            
            # Get specified round
            responses = debate.get_round(round_idx)
            
            # Aggregate
            social_prob = self.aggregator.compute_social_scores(responses)
            
            calibration_data.append((social_prob, ground_truth))
        
        # Calibrate threshold
        self.calibrator.calibrate(calibration_data)
    
    def predict(self, query: str) -> Tuple[List[str], Dict[str, float]]:
        """
        Make prediction with safety guarantee.
        
        Returns:
            prediction_set: Certified set of candidates
            social_prob: Aggregated probability distribution
        """
        # Run debate
        debate = self.debate_manager.run_debate(query)
        
        # Use final round
        responses = debate.get_round(-1)
        
        # Aggregate
        social_prob = self.aggregator.compute_social_scores(responses)
        
        # Get prediction set
        prediction_set = self.calibrator.get_prediction_set(social_prob)
        
        return prediction_set, social_prob
    
    def predict_with_policy(self, query: str) -> Dict:
        """
        Hierarchical action policy with human-in-the-loop logic.
        
        Returns:
            result: Dict with action, prediction_set, and metadata
        """
        pred_set, social_prob = self.predict(query)
        
        if len(pred_set) == 1:
            # Case 1: Full automation
            action = "automate"
            final_answer = pred_set[0]
            confidence = social_prob[final_answer]
        else:
            # Case 2: Human intervention
            action = "escalate_to_human"
            final_answer = None
            confidence = max([social_prob[opt] for opt in pred_set])
        
        return {
            "action": action,
            "prediction_set": pred_set,
            "set_size": len(pred_set),
            "final_answer": final_answer,
            "confidence": confidence,
            "social_distribution": social_prob
        }
    
    def analyze_trajectory(self, query: str, ground_truth: Optional[str] = None):
        """
        Visualize consensus evolution across rounds.
        """
        debate = self.debate_manager.run_debate(query, ground_truth)
        
        trajectory = []
        for round_idx in range(len(debate.debate_history)):
            responses = debate.get_round(round_idx)
            social_prob = self.aggregator.compute_social_scores(responses)
            pred_set = self.calibrator.get_prediction_set(social_prob)
            
            # Compute entropy
            probs = np.array([social_prob[opt] for opt in self.options])
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            
            trajectory.append({
                "round": round_idx,
                "set_size": len(pred_set),
                "prediction_set": pred_set,
                "entropy": entropy,
                "max_prob": max(social_prob.values()),
                "contains_truth": ground_truth in pred_set if ground_truth else None
            })
        
        return trajectory


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    from datasets import load_dataset
    from utils import load_test_data, format_question_with_options
    
    # Configuration
    AGENT = "claude-haiku"  # Single agent for simplified testing (no debate)
    CALIBRATION_SIZE = 100  # Samples 100-200
    INFERENCE_SIZE = 100    # Samples 0-100
    ALPHA = 0.1  # Miscoverage rate (10% for 90% coverage)
    
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Evaluate greedy single agent baseline on MMLU-Pro Business'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=4096,
        help='Maximum tokens to generate from LLM (default: 4096)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Temperature for sampling from LLM (default: 0.0)'
    )
    
    args = parser.parse_args()
    MAX_TOKENS = args.max_tokens
    
    logger.info("="*80)
    logger.info("Conformal Calibration Evaluation on MMLU-Pro Business")
    logger.info("="*80)
    
    # Load data
    logger.info("\nLoading MMLU-Pro Business test set...")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    business_data = [ex for ex in dataset if ex.get('category') == 'business']
    
    # Split data: Calibration (100-200), Inference (0-100)
    calibration_samples = business_data[100:200]
    inference_samples = business_data[0:100]
    
    logger.info(f"Calibration set: {len(calibration_samples)} samples (indices 100-200)")
    logger.info(f"Inference set: {len(inference_samples)} samples (indices 0-100)")
    logger.info(f"Using single agent: {AGENT}")
    logger.info(f"Target coverage: {(1-ALPHA)*100:.0f}% (α={ALPHA})")
    
    # ========================================================================
    # PHASE 1: Calibration - Compute social probabilities for calibration set
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: CALIBRATION")
    logger.info("="*80)
    
    calibration_data = []
    
    for idx, sample in enumerate(calibration_samples):
        question = sample['question']
        options = sample['options']
        correct_answer = sample['answer']
        option_letters = [chr(65+i) for i in range(len(options))]
        
        # Format question
        formatted_question = format_question_with_options(question, options)
        
        # Initialize single-agent manager (no debate, num_rounds=1)
        manager = MultiAgentDebateManager(
            agents=[AGENT],
            options=option_letters,
            num_rounds=1,
            max_tokens=MAX_TOKENS
        )
        
        try:
            # Get agent's probability distribution
            logits, response_text = manager.extract_logits(
                agent_id=AGENT,
                query=formatted_question,
                round_num=0,
                history=[]
            )

            # Create response and compute social score (trivial for single agent)
            response = AgentResponse(agent_id=AGENT, round_num=0, logits=logits, response_text=response_text)
            aggregator = SocialWelfareAggregator(options=option_letters, weight_strategy="uniform")
            social_prob = aggregator.compute_social_scores([response])

            # Store for calibration
            calibration_data.append((social_prob, correct_answer))
            
            if (idx + 1) % 20 == 0:
                logger.info(f"Calibration progress: {idx + 1}/{len(calibration_samples)}")
                
        except Exception as e:
            logger.error(f"Error on calibration sample {idx}: {str(e)}")
            # Use uniform distribution as fallback
            uniform_prob = 1.0 / len(option_letters)
            social_prob = {opt: uniform_prob for opt in option_letters}
            calibration_data.append((social_prob, correct_answer))
    
    logger.info(f"✓ Calibration data collected: {len(calibration_data)} samples")
    
    # ========================================================================
    # PHASE 2: Compute Conformal Threshold
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: CONFORMAL CALIBRATION")
    logger.info("="*80)
    
    calibrator = ConformalCalibrator(alpha=ALPHA)
    calibrator.calibrate(calibration_data)
    
    logger.info(f"✓ Conformal threshold q̂ = {calibrator.q_hat:.4f}")
    
    # ========================================================================
    # PHASE 3: Inference - Generate prediction sets
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: INFERENCE WITH CONFORMAL PREDICTION")
    logger.info("="*80)
    
    inference_results = []
    prediction_sets_all = []
    
    for idx, sample in enumerate(inference_samples):
        question = sample['question']
        options = sample['options']
        correct_answer = sample['answer']
        option_letters = [chr(65+i) for i in range(len(options))]
        
        # Format question
        formatted_question = format_question_with_options(question, options)
        
        # Initialize single-agent manager
        manager = MultiAgentDebateManager(
            agents=[AGENT],
            options=option_letters,
            num_rounds=1,
            max_tokens=MAX_TOKENS
        )
        
        try:
            # Get agent's probability distribution
            logits, response_text = manager.extract_logits(
                agent_id=AGENT,
                query=formatted_question,
                round_num=0,
                history=[]
            )

            # Create response and compute social score
            response = AgentResponse(agent_id=AGENT, round_num=0, logits=logits, response_text=response_text)
            aggregator = SocialWelfareAggregator(options=option_letters, weight_strategy="uniform")
            social_prob = aggregator.compute_social_scores([response])
            
            # Generate conformal prediction set
            prediction_set = calibrator.get_prediction_set(social_prob)
            
            # Check coverage
            covered = correct_answer in prediction_set
            set_size = len(prediction_set)
            
            # Greedy prediction (argmax)
            greedy_pred = max(social_prob.items(), key=lambda x: x[1])[0]
            greedy_correct = greedy_pred == correct_answer
            
            inference_results.append({
                'question_id': sample.get('question_id', idx),
                'prediction_set': prediction_set,
                'set_size': set_size,
                'covered': covered,
                'correct_answer': correct_answer,
                'social_prob': social_prob,
                'greedy_pred': greedy_pred,
                'greedy_correct': greedy_correct
            })
            
            prediction_sets_all.append(prediction_set)
            
            if (idx + 1) % 20 == 0:
                current_coverage = sum(r['covered'] for r in inference_results) / len(inference_results)
                avg_set_size = np.mean([r['set_size'] for r in inference_results])
                logger.info(f"Inference progress: {idx + 1}/{len(inference_samples)} | "
                           f"Coverage: {current_coverage:.2%} | Avg set size: {avg_set_size:.2f}")
                
        except Exception as e:
            logger.error(f"Error on inference sample {idx}: {str(e)}")
            # Record as uncovered with empty set
            inference_results.append({
                'question_id': sample.get('question_id', idx),
                'prediction_set': [],
                'set_size': 0,
                'covered': False,
                'correct_answer': correct_answer,
                'social_prob': {},
                'greedy_pred': None,
                'greedy_correct': False
            })
    
    # ========================================================================
    # PHASE 4: Results & Analysis
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("EVALUATION RESULTS")
    logger.info("="*80)
    
    # Coverage metrics
    coverage = sum(r['covered'] for r in inference_results) / len(inference_results)
    avg_set_size = np.mean([r['set_size'] for r in inference_results])
    median_set_size = np.median([r['set_size'] for r in inference_results])
    
    # Greedy baseline accuracy
    greedy_accuracy = sum(r['greedy_correct'] for r in inference_results) / len(inference_results)
    
    # Set size distribution
    set_sizes = [r['set_size'] for r in inference_results]
    singleton_rate = sum(1 for s in set_sizes if s == 1) / len(set_sizes)
    empty_rate = sum(1 for s in set_sizes if s == 0) / len(set_sizes)
    
    logger.info(f"\n📊 Conformal Prediction Metrics:")
    logger.info(f"  Target Coverage: {(1-ALPHA)*100:.0f}%")
    logger.info(f"  Empirical Coverage: {coverage:.2%} ({sum(r['covered'] for r in inference_results)}/{len(inference_results)})")
    logger.info(f"  Average Set Size: {avg_set_size:.2f}")
    logger.info(f"  Median Set Size: {median_set_size:.1f}")
    logger.info(f"  Singleton Sets: {singleton_rate:.2%}")
    logger.info(f"  Empty Sets: {empty_rate:.2%}")
    
    logger.info(f"\n🎯 Baseline Comparison:")
    logger.info(f"  Greedy (Argmax) Accuracy: {greedy_accuracy:.2%}")
    
    logger.info(f"\n📈 Set Size Distribution:")
    for size in sorted(set(set_sizes)):
        count = sum(1 for s in set_sizes if s == size)
        pct = count / len(set_sizes)
        logger.info(f"  Size {size}: {count} ({pct:.1%})")
    
    # Sample results
    logger.info(f"\n🔍 Sample Predictions (first 5):")
    logger.info("-"*80)
    for i, result in enumerate(inference_results[:5]):
        status = "✓" if result['covered'] else "✗"
        logger.info(f"{status} Q{result['question_id']}: Set={result['prediction_set']} | "
                   f"True={result['correct_answer']} | Size={result['set_size']}")
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)
