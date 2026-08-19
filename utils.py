from datasets import load_dataset
import logging
from typing import List, Dict
from collections import Counter
import time
import random

# Import from parent modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import MODEL_IDS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_bedrock_response(system_prompt, user_question, max_tokens, temperature, top_p, max_retries, bedrock_client, bedrock_model_name, logger):
    """
    Get response from Bedrock models
    
    Args:
        prompt: The prompt to send to Bedrock
        
    Returns:
        Response from Bedrock
    """
    # Try up to max_retries times if there's an error
    for retry_count in range(max_retries):
        try:
            # Add a small delay to prevent rate limiting
            time.sleep(random.uniform(0.1, 0.5))

            # Handle instance-numbered agent IDs (e.g., 'claude-haiku-1' -> 'claude-haiku')
            # This allows multiple instances of the same model in self-debate mode
            lookup_name = bedrock_model_name
            if bedrock_model_name not in MODEL_IDS:
                # Try stripping the last suffix if it's a number (e.g., '-1', '-2')
                parts = bedrock_model_name.rsplit('-', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    lookup_name = parts[0]

            bedrock_model_id = MODEL_IDS[lookup_name]
            
            # logger.info(f"Getting response using: {bedrock_model_name} ({bedrock_model_id})")
            
            response = bedrock_client.converse(
                modelId=bedrock_model_id,
                system = [
                    {"text": system_prompt}
                ],
                messages = [
                    {"role": "user", 
                     "content": [{"text": user_question}]}
                ],
                inferenceConfig={
                    "maxTokens": max_tokens,
                    # "temperature": temperature,
                    # "topP": top_p,
                }
            )

            
            # Extract response text - handle different response structures
            output_message = response['output']['message']
            # logger.info(f'Output message: {output_message}')
            input_token = response['usage']['inputTokens']
            output_token = response['usage']['outputTokens']
            
            content = output_message['content']
            
            # Combine all text parts (handles models like Qwen with reasoning + answer)
            response_parts = []
            for item in content:
                if 'text' in item:
                    response_parts.append(item['text'])
                elif 'reasoningContent' in item:
                    # Extract reasoning text from Qwen's reasoningContent structure
                    try:
                        reasoning_text = item['reasoningContent']['reasoningText']['text']
                        response_parts.append(reasoning_text)
                    except (KeyError, TypeError):
                        pass
            
            # Join all parts with newline
            response_text = '\n'.join(response_parts)

            # Return response text along with token usage
            return response_text, input_token, output_token
                    
        except Exception as e:
            if retry_count < max_retries - 1:
                logger.warning(f"Error from Bedrock (attempt {retry_count+1}/{max_retries}): {e}. Retrying...")
                time.sleep(1.0)  # Exponential backoff
            else:
                logger.error(f"All {max_retries} attempts failed. Error: {e}")
                return "Unable to provide response due to technical issues.", 0, 0

    # This should never be reached
    return "Unable to provide response due to technical issues.", 0, 0


def load_musr_data(split: str = 'murder_mysteries', use_subset: bool = False, num_samples: int = 100, start_index: int = 0):
    """
    Load samples from MuSR (Multi-Step Reasoning) benchmark.

    Args:
        split: MuSR subset to load ('murder_mysteries', 'object_placements', 'team_allocation')
        use_subset: Whether to use a subset of the data (default: False, uses entire dataset)
        num_samples: Number of samples to load when use_subset=True (default: 100)
        start_index: Starting index for sampling when use_subset=True (default: 0)

    Returns:
        List of test examples in standardized format
    """
    import ast

    valid_splits = ['murder_mysteries', 'object_placements', 'team_allocation']
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Must be one of: {', '.join(valid_splits)}")

    logger.info(f"Loading MuSR dataset, split: {split}")

    dataset = load_dataset("TAUR-Lab/MuSR", split=split)
    logger.info(f"Total {split} examples: {len(dataset)}")

    standardized_data = []
    for idx, example in enumerate(dataset):
        # Parse choices from string representation
        choices = ast.literal_eval(example['choices'])
        answer_idx = example['answer_index']

        # Combine narrative and question
        full_question = f"{example['narrative']}\n\n{example['question']}"

        standardized_data.append({
            'question_id': idx,
            'question': full_question,
            'options': choices,
            'answer': chr(65 + answer_idx),  # Convert 0-based index to A, B, C, ...
            'source': 'MuSR'
        })

    logger.info(f"Standardized {len(standardized_data)} MuSR examples")

    if use_subset:
        subset = standardized_data[start_index:start_index + num_samples]
        logger.info(f"Using {len(subset)} MuSR examples (indices {start_index} to {start_index + len(subset) - 1})")
        return subset
    else:
        logger.info(f"Using entire dataset: {len(standardized_data)} MuSR examples")
        return standardized_data


def load_arc_data(split: str = 'ARC-Challenge', use_subset: bool = False, num_samples: int = 100, start_index: int = 0):
    """
    Load test samples from AI2 ARC (AI2 Reasoning Challenge).

    Args:
        split: ARC configuration to load ('ARC-Challenge' or 'ARC-Easy')
        use_subset: Whether to use a subset of the data (default: False, uses entire dataset)
        num_samples: Number of samples to load when use_subset=True (default: 100)
        start_index: Starting index for sampling when use_subset=True (default: 0)

    Returns:
        List of test examples in standardized format
    """
    valid_splits = ['ARC-Challenge', 'ARC-Easy']
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Must be one of: {', '.join(valid_splits)}")

    logger.info(f"Loading AI2 ARC dataset, split: {split}")

    dataset = load_dataset("allenai/ai2_arc", split, split="test")
    logger.info(f"Total {split} test examples: {len(dataset)}")

    standardized_data = []
    for idx, example in enumerate(dataset):
        choices = example['choices']
        labels = choices['label']
        texts = choices['text']

        # Normalize labels: some questions use "1","2","3","4" instead of "A","B","C","D"
        if labels and labels[0].isdigit():
            option_map = {str(i + 1): chr(65 + i) for i in range(len(labels))}
            labels = [option_map.get(l, l) for l in labels]
            answer_key = option_map.get(example['answerKey'], example['answerKey'])
        else:
            answer_key = example['answerKey']

        standardized_data.append({
            'question_id': example.get('id', idx),
            'question': example['question'],
            'options': texts,
            'answer': answer_key,
            'source': 'ARC'
        })

    logger.info(f"Standardized {len(standardized_data)} ARC examples")

    if use_subset:
        subset = standardized_data[start_index:start_index + num_samples]
        logger.info(f"Using {len(subset)} ARC examples (indices {start_index} to {start_index + len(subset) - 1})")
        return subset
    else:
        logger.info(f"Using entire dataset: {len(standardized_data)} ARC examples")
        return standardized_data


def load_test_data(category: str = 'business', use_subset: bool = False, num_samples: int = 100, start_index: int = 0):
    """
    Load test samples from MMLU-Pro for a specified category.
    
    Args:
        category: Category to load (options: 'business', 'math', 'physics', 'chemistry', 'law', 'engineering')
        use_subset: Whether to use a subset of the data (default: False, uses entire dataset)
        num_samples: Number of samples to load when use_subset=True (default: 100)
        start_index: Starting index for sampling when use_subset=True (default: 0)
    
    Returns:
        List of test examples for the specified category
    """
    # Validate category
    #valid_categories = ['business', 'math', 'physics', 'chemistry', 'law', 'engineering']
    #if category not in valid_categories:
    #    raise ValueError(f"Invalid category '{category}'. Must be one of: {', '.join(valid_categories)}")
    
    logger.info(f"Loading MMLU-Pro test set for category: {category}")
    
    # Load the test split
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    
    logger.info(f"Total test examples: {len(dataset)}")
    
    # Filter for specified category
    category_data = [
        example for example in dataset 
        if example.get('category') == category
    ]
    
    logger.info(f"Total {category} test examples: {len(category_data)}")
    
    if use_subset:
        # Take num_samples starting from start_index
        category_subset = category_data[start_index:start_index + num_samples]
        logger.info(f"Using {len(category_subset)} {category} examples (indices {start_index} to {start_index + len(category_subset) - 1})")
        return category_subset
    else:
        logger.info(f"Using entire dataset: {len(category_data)} {category} examples")
        return category_data


def load_gpqa_data(split: str = 'gpqa_main', use_subset: bool = False, num_samples: int = 100, start_index: int = 0):
    """
    Load test samples from GPQA (Graduate-Level Google-Proof Q&A Benchmark).

    Args:
        split: GPQA split to load (options: 'gpqa_main', 'gpqa_diamond', 'gpqa_extended')
        use_subset: Whether to use a subset of the data (default: False, uses entire dataset)
        num_samples: Number of samples to load when use_subset=True (default: 100)
        start_index: Starting index for sampling when use_subset=True (default: 0)

    Returns:
        List of test examples in standardized format

    Note:
        GPQA is a gated dataset. You must:
        1. Accept terms at https://huggingface.co/datasets/Idavidrein/gpqa
        2. Login with: huggingface-cli login
    """
    valid_splits = ['gpqa_main', 'gpqa_diamond', 'gpqa_extended']
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Must be one of: {', '.join(valid_splits)}")

    logger.info(f"Loading GPQA dataset, split: {split}")

    try:
        # Load the dataset (requires authentication)
        # Note: 'split' parameter here is the configuration name (gpqa_main, gpqa_diamond, etc.)
        # The actual split is always 'train'
        dataset = load_dataset("Idavidrein/gpqa", split, split="train")

        logger.info(f"Total {split} examples: {len(dataset)}")

        # Standardize format to match MMLU-Pro structure
        standardized_data = []

        for idx, example in enumerate(dataset):
            # GPQA structure typically has:
            # - 'Question': the question text
            # - 'Correct Answer': the correct answer text
            # - 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3': wrong answers
            # OR it might have different field names, so we need to be flexible

            # Try common field name patterns
            question_field = None
            for field in ['Question', 'question', 'query', 'Question Text']:
                if field in example:
                    question_field = field
                    break

            if not question_field:
                logger.warning(f"Could not find question field in example {idx}")
                continue

            question = example[question_field]

            # Extract answer options
            # Try to find correct answer and incorrect answers
            correct_answer_text = None
            incorrect_answers = []

            # Pattern 1: Named fields
            if 'Correct Answer' in example:
                correct_answer_text = example['Correct Answer']
                for i in range(1, 4):
                    field = f'Incorrect Answer {i}'
                    if field in example:
                        incorrect_answers.append(example[field])

            # Pattern 2: Direct answer fields
            elif 'correct_answer' in example:
                correct_answer_text = example['correct_answer']
                for i in range(1, 4):
                    field = f'incorrect_answer_{i}'
                    if field in example:
                        incorrect_answers.append(example[field])

            # Pattern 3: Options list
            elif 'options' in example:
                options = example['options']
                answer_key = example.get('answer', example.get('correct_answer_index', 0))
                if isinstance(answer_key, int):
                    correct_idx = answer_key
                else:
                    # If answer is like 'A', 'B', 'C', 'D'
                    correct_idx = ord(answer_key) - ord('A')

                standardized_data.append({
                    'question_id': example.get('Record ID', example.get('question_id', idx)),
                    'question': question,
                    'options': options,
                    'answer': chr(65 + correct_idx),  # Convert to A, B, C, D
                    'subject': example.get('Subdomain', example.get('High-level domain', 'GPQA')),
                    'source': 'GPQA'
                })
                continue

            # If we found answer components, randomize order
            if correct_answer_text and len(incorrect_answers) >= 3:
                # Combine all answers
                all_options = [correct_answer_text] + incorrect_answers[:3]

                # Shuffle to randomize order (with fixed seed for reproducibility per question)
                import random
                rng = random.Random(idx)  # Use question index as seed
                shuffled_indices = list(range(len(all_options)))
                rng.shuffle(shuffled_indices)

                shuffled_options = [all_options[i] for i in shuffled_indices]
                correct_idx = shuffled_indices.index(0)  # Where did the correct answer end up?

                standardized_data.append({
                    'question_id': example.get('Record ID', example.get('question_id', idx)),
                    'question': question,
                    'options': shuffled_options,
                    'answer': chr(65 + correct_idx),  # A, B, C, D
                    'subject': example.get('Subdomain', example.get('High-level domain', 'GPQA')),
                    'source': 'GPQA'
                })

        logger.info(f"Standardized {len(standardized_data)} GPQA examples")

        if use_subset:
            subset = standardized_data[start_index:start_index + num_samples]
            logger.info(f"Using {len(subset)} GPQA examples (indices {start_index} to {start_index + len(subset) - 1})")
            return subset
        else:
            logger.info(f"Using entire dataset: {len(standardized_data)} GPQA examples")
            return standardized_data

    except Exception as e:
        logger.error(f"Error loading GPQA dataset: {e}")
        logger.info("\nTo access GPQA dataset:")
        logger.info("  1. Accept terms at https://huggingface.co/datasets/Idavidrein/gpqa")
        logger.info("  2. Login with: huggingface-cli login")
        raise


def format_question_with_options(question: str, options: List[str]) -> str:
    """Format question with numbered options."""
    formatted = f"{question}\n\nOptions:\n"
    for i, opt in enumerate(options):
        formatted += f"{chr(65+i)}. {opt}\n"
    return formatted
