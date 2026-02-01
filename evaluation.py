import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor
from PIL import Image
import av
import numpy as np
from tqdm import tqdm
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional


def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def load_video_frames(video_path: str, num_frames: int = 8):
    """Load video and sample frames uniformly. Pads if video is too short."""
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    if total_frames == 0:
        total_frames = sum(1 for _ in container.decode(video=0))
        container.seek(0)
    
    actual_frames = min(total_frames, num_frames)
    if actual_frames < 1:
        actual_frames = 1
    indices = np.linspace(0, max(total_frames - 1, 0), actual_frames, dtype=int)
    video_frames = read_video_pyav(container, indices)
    container.close()
    
    frames = [Image.fromarray(frame) for frame in video_frames]
    
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    
    return frames


def clean_text(text: str) -> str:
    """Clean text while preserving multi-word answers and decimal numbers."""
    text = text.lower().strip()
    # Remove special tokens
    text = re.sub(r'<\|.*?\|>', '', text)
    # Remove punctuation except dots (for decimals) and spaces
    text = re.sub(r'[^\w\s\.]', ' ', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    # Remove trailing dots (but keep decimal dots)
    text = re.sub(r'\.+$', '', text)
    return text.strip()


def extract_number(text: str) -> Optional[float]:
    """Extract a number from text, returns None if no number found."""
    text = clean_text(text)
    # Try to find a number (integer or decimal)
    match = re.search(r'^(\d+\.?\d*)', text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def is_number(text: str) -> bool:
    """Check if text represents a number."""
    text = clean_text(text)
    return bool(re.match(r'^\d+\.?\d*$', text))


def normalize_answer(answer: str) -> str:
    """Normalize ground truth answer - preserve full answer."""
    answer = clean_text(str(answer))
    
    # Remove trailing punctuation from numbers
    if re.match(r'^\d+\.?\d*$', answer):
        return answer
    
    # Check for true/false/yes/no
    if answer in ['true', 'yes']:
        return 'true'
    if answer in ['false', 'no']:
        return 'false'
    
    # For everything else, return the full cleaned answer
    return answer


def is_boolean_word(text: str) -> Optional[str]:
    """
    Check if text is a boolean word (yes/no/true/false).
    Returns normalized boolean ('true'/'false') or None.
    
    Only matches exact words, not prefixes like "not_duplicate".
    """
    # Get first word only
    first_word = text.split()[0] if text.split() else ""
    # Remove trailing punctuation from first word
    first_word = re.sub(r'[^\w]', '', first_word)
    
    if first_word in ['true', 'yes']:
        return 'true'
    if first_word in ['false', 'no']:
        return 'false'
    return None


def extract_answer(pred_text: str, true_answer: str = "") -> str:
    """
    Extract answer from model output, using true_answer as hint for format.
    
    Handles:
    - Multi-word answers (e.g., "passion flower", "wild cat")
    - Underscore-separated answers (e.g., "not_duplicate")
    - Single letter choices (a, b, c, d)
    - Numbers with trailing punctuation
    - Boolean answers (only exact yes/no/true/false)
    """
    pred_clean = clean_text(pred_text)
    true_clean = normalize_answer(true_answer) if true_answer else ""
    
    # If true answer contains underscore (like not_duplicate), check for it first
    if true_clean and '_' in true_clean:
        # Check exact match or if prediction starts with it
        if pred_clean == true_clean or pred_clean.startswith(true_clean + ' '):
            return true_clean
        # Also check if prediction contains the underscore version
        if true_clean in pred_clean:
            return true_clean
    
    # If true answer is multi-word, try to find it in prediction
    if true_clean and ' ' in true_clean:
        # Check if true answer appears in prediction
        if true_clean in pred_clean:
            return true_clean
        # Check if prediction starts with true answer
        if pred_clean.startswith(true_clean):
            return true_clean
    
    # Check for single letter answer (like "a", "b", "c")
    match = re.match(r'^[\(\[]?([a-z])[\)\]]?[\.\s,]?$', pred_clean)
    if match and (not true_clean or len(true_clean) == 1):
        return match.group(1).lower()
    
    # Also check if first token is a single letter followed by punctuation
    match = re.match(r'^[\(\[]?([a-z])[\)\]\.\s,:]', pred_clean)
    if match and (not true_clean or len(true_clean) == 1):
        return match.group(1).lower()
    
    # Check for true/false - ONLY exact word matches, not prefixes
    bool_result = is_boolean_word(pred_clean)
    if bool_result and (not true_clean or true_clean in ['true', 'false', 'yes', 'no']):
        return bool_result
    
    # Check for sentiment
    first_word = pred_clean.split()[0] if pred_clean.split() else ""
    if first_word == 'positive':
        return 'positive'
    if first_word == 'negative':
        return 'negative'
    if first_word == 'neutral':
        return 'neutral'
    
    # Check for common classification labels (duplicate, entailment, etc.)
    common_labels = ['duplicate', 'not_duplicate', 'entailment', 'contradiction', 
                     'neutral', 'equivalent', 'not_equivalent', 
                     'acceptable', 'unacceptable']
    for label in common_labels:
        if pred_clean == label or pred_clean.startswith(label + ' ') or pred_clean.startswith(label + '.'):
            return label
    
    # Check for number at start
    num_match = re.match(r'^(\d+\.?\d*)', pred_clean)
    if num_match:
        return num_match.group(1)
    
    # If true answer is known, check if it appears anywhere in prediction
    if true_clean:
        # For single words, check if it's in the prediction
        if true_clean in pred_clean.split():
            return true_clean
        # Check if prediction contains the true answer
        if true_clean in pred_clean:
            return true_clean
    
    # Return the first word or full cleaned prediction
    first_word = pred_clean.split()[0] if pred_clean.split() else pred_clean
    return first_word


def compare_answers(pred: str, true: str, tolerance: float = 0.5) -> Tuple[bool, str]:
    """
    Compare predicted and true answers with smart matching.
    
    Returns:
        (is_correct, match_type)
    
    Match types:
        - "exact": Exact string match
        - "semantic": Semantically equivalent (e.g., not_duplicate == false)
        - "numeric": Numbers within tolerance
        - "contains": One contains the other
        - "no_match": No match found
    """
    pred_norm = normalize_answer(pred)
    true_norm = normalize_answer(true)
    
    # 1. Exact match
    if pred_norm == true_norm:
        return True, "exact"
    
    # 2. Semantic equivalence mappings for classification tasks
    # These handle cases where model outputs different but equivalent labels
    semantic_groups = [
        # QQP task
        {'duplicate', 'true', 'yes', '1'},
        {'not_duplicate', 'not duplicate', 'false', 'no', '0'},
        # MRPC task
        {'equivalent', 'true', 'yes', '1'},
        {'not_equivalent', 'not equivalent', 'false', 'no', '0'},
        # Entailment tasks (MNLI, QNLI, RTE, WNLI)
        {'entailment', 'true', 'yes', '1'},
        {'not_entailment', 'not entailment', 'false', 'no', '0'},
        {'contradiction'},
        {'neutral'},
        # CoLA task - these should NOT be equivalent to each other!
        {'acceptable', 'grammatical', 'correct'},
        {'unacceptable', 'ungrammatical', 'incorrect'},
    ]
    
    for group in semantic_groups:
        if pred_norm in group and true_norm in group:
            return True, "semantic"
    
    # 3. Numeric comparison with tolerance
    pred_num = extract_number(pred)
    true_num = extract_number(true)
    
    if pred_num is not None and true_num is not None:
        if abs(pred_num - true_num) <= tolerance:
            return True, "numeric"
    
    # 4. Contains match (for multi-word or extra tokens)
    # BUT: Avoid false positives where one word is substring of opposite meaning
    # e.g., "acceptable" should NOT match "unacceptable"
    opposite_pairs = [
        ('acceptable', 'unacceptable'),
        ('grammatical', 'ungrammatical'),
        ('correct', 'incorrect'),
        ('equivalent', 'not_equivalent'),
        ('equivalent', 'not equivalent'),
        ('duplicate', 'not_duplicate'),
        ('duplicate', 'not duplicate'),
        ('entailment', 'not_entailment'),
        ('entailment', 'not entailment'),
    ]
    
    # Check if this is an opposite pair - if so, don't use contains matching
    for word1, word2 in opposite_pairs:
        if (pred_norm == word1 and true_norm == word2) or \
           (pred_norm == word2 and true_norm == word1):
            return False, "no_match"
        # Also check if one contains the other in a bad way
        if (word1 in pred_norm and word2 == true_norm) or \
           (word2 in pred_norm and word1 == true_norm):
            return False, "no_match"
        if (word1 in true_norm and word2 == pred_norm) or \
           (word2 in true_norm and word1 == pred_norm):
            return False, "no_match"
    
    # Check if true answer is contained in prediction
    if true_norm in pred_norm:
        return True, "contains"
    
    # Check if prediction is contained in true answer
    if pred_norm in true_norm:
        return True, "contains"
    
    # 5. Word-level containment for multi-word answers
    true_words = set(true_norm.split())
    pred_words = set(pred_norm.split())
    
    # If all true words are in prediction (handles "passion flower" in "passion flower is...")
    if true_words and true_words.issubset(pred_words):
        # Make sure they appear in the right order
        if true_norm in pred_norm:
            return True, "contains"
    
    # 6. First N words match (for when prediction has extra tokens)
    pred_first_words = ' '.join(pred_norm.split()[:len(true_norm.split())])
    if pred_first_words == true_norm:
        return True, "contains"
    
    return False, "no_match"


def get_task_type(split_name: str) -> str:
    """Determine evaluation type based on split name"""
    split_lower = split_name.lower()
    
    # Regression tasks (need Pearson/Spearman)
    if 'stsb' in split_lower:
        return 'regression'
    
    # Multiple choice tasks
    if any(x in split_lower for x in ['shuffle', 'mcq', 'choice', 'vqa']):
        return 'multiple_choice'
    
    # Sentiment/Classification
    if any(x in split_lower for x in ['sst', 'sentiment', 'cola', 'mnli', 'qnli', 'rte', 'wnli', 'mrpc', 'qqp']):
        return 'classification'
    
    # Default to classification
    return 'classification'


def compute_pearson(predictions: List[float], references: List[float]) -> float:
    """Compute Pearson correlation coefficient"""
    from scipy.stats import pearsonr
    if len(predictions) < 2:
        return 0.0
    try:
        corr, _ = pearsonr(predictions, references)
        return corr if not np.isnan(corr) else 0.0
    except:
        return 0.0


def compute_spearman(predictions: List[float], references: List[float]) -> float:
    """Compute Spearman correlation coefficient"""
    from scipy.stats import spearmanr
    if len(predictions) < 2:
        return 0.0
    try:
        corr, _ = spearmanr(predictions, references)
        return corr if not np.isnan(corr) else 0.0
    except:
        return 0.0


class EvalDataset(Dataset):
    def __init__(self, dataset, processor, data_root="", num_video_frames=8, max_length=2048):
        self.dataset = dataset
        self.processor = processor
        self.data_root = data_root
        self.num_video_frames = num_video_frames
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        source_type = item['source_type']
        media_path = item['media_path']
        question = item['question']
        answer = item['answer']
        
        if source_type == 'text':
            conversation = [{"role": "user", "content": [{"type": "text", "text": question}]}]
            images, videos = None, None
        elif source_type == 'image':
            conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
            images = [Image.open(os.path.join(self.data_root, media_path)).convert('RGB')]
            videos = None
        elif source_type == 'video':
            conversation = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]}]
            videos = [load_video_frames(os.path.join(self.data_root, media_path), self.num_video_frames)]
            images = None
        else:
            raise ValueError(f"Unknown source_type: {source_type}")
        
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        processed = self.processor(text=prompt, images=images, videos=videos, 
                                   truncation=True, max_length=self.max_length, return_tensors="pt")
        
        result = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in processed.items()}
        result['answer'] = answer
        result['question'] = question
        return result


@dataclass
class EvalCollator:
    processor: LlavaOnevisionProcessor
    max_length: int = 2048
    num_video_frames: int = 8
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        pad_token_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
        max_len = min(max(item['input_ids'].shape[0] for item in batch), self.max_length)
        
        batch_input_ids = []
        batch_attention_mask = []
        all_pixel_values = []
        all_image_sizes = []
        all_pixel_values_videos = []
        answers = []
        questions = []
        
        for item in batch:
            input_ids = item['input_ids']
            seq_len = input_ids.shape[0]
            
            if seq_len > max_len:
                input_ids = input_ids[-max_len:]
                seq_len = max_len
            
            padding_length = max_len - seq_len
            if padding_length > 0:
                input_ids = torch.cat([
                    torch.full((padding_length,), pad_token_id, dtype=input_ids.dtype),
                    input_ids,
                ])
                attention_mask = torch.cat([
                    torch.zeros(padding_length, dtype=torch.long),
                    torch.ones(seq_len, dtype=torch.long),
                ])
            else:
                attention_mask = torch.ones(seq_len, dtype=torch.long)
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            answers.append(item['answer'])
            questions.append(item['question'])
            
            if 'pixel_values' in item:
                pv = item['pixel_values']
                all_pixel_values.append(pv.unsqueeze(0) if pv.dim() == 3 else pv)
                if 'image_sizes' in item:
                    img_sizes = item['image_sizes']
                    all_image_sizes.append(img_sizes.unsqueeze(0) if img_sizes.dim() == 1 else img_sizes)
            
            if 'pixel_values_videos' in item:
                pv_vid = item['pixel_values_videos']
                if pv_vid.dim() == 4:
                    pv_vid = pv_vid.unsqueeze(0)
                
                current_frames = pv_vid.shape[1]
                if current_frames < self.num_video_frames:
                    pad_frames = self.num_video_frames - current_frames
                    last_frame = pv_vid[:, -1:, :, :, :]
                    padding = last_frame.repeat(1, pad_frames, 1, 1, 1)
                    pv_vid = torch.cat([pv_vid, padding], dim=1)
                elif current_frames > self.num_video_frames:
                    pv_vid = pv_vid[:, :self.num_video_frames, :, :, :]
                
                all_pixel_values_videos.append(pv_vid)
        
        result = {
            'input_ids': torch.stack(batch_input_ids),
            'attention_mask': torch.stack(batch_attention_mask),
            'answers': answers,
            'questions': questions,
        }
        
        if all_pixel_values:
            result['pixel_values'] = torch.cat(all_pixel_values, dim=0)
            if all_image_sizes:
                result['image_sizes'] = torch.cat(all_image_sizes, dim=0)
        
        if all_pixel_values_videos:
            result['pixel_values_videos'] = torch.cat(all_pixel_values_videos, dim=0)
        
        return result


def evaluate_dataset_batched(
    model, processor, dataset_split, data_root, split_name,
    num_samples=None, batch_size=4, num_video_frames=8, max_new_tokens=50, 
    device="cuda", numeric_tolerance=0.5
):
    """
    Evaluate a dataset split with appropriate metrics.
    
    - For STSB: Pearson & Spearman correlation
    - For others: Accuracy with smart matching
    
    Args:
        numeric_tolerance: Tolerance for numerical comparisons (default 0.5)
    """
    task_type = get_task_type(split_name)
    is_regression = task_type == 'regression'
    
    if num_samples is None:
        num_samples = len(dataset_split)
    else:
        num_samples = min(num_samples, len(dataset_split))
    
    eval_dataset = EvalDataset(
        dataset=dataset_split.select(range(num_samples)),
        processor=processor, data_root=data_root, num_video_frames=num_video_frames
    )
    
    # For regression, we need more tokens to capture decimal numbers
    actual_max_tokens = max_new_tokens if not is_regression else max(max_new_tokens, 10)
    
    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=EvalCollator(processor=processor, num_video_frames=num_video_frames),
        num_workers=0
    )
    
    correct, total = 0, 0
    results = []
    pred_scores = []  # For regression
    true_scores = []  # For regression
    match_type_counts = defaultdict(int)  # Track how matches were made
    
    print(f"\n{'='*70}")
    print(f"Evaluating: {split_name} ({num_samples} samples, batch_size={batch_size})")
    print(f"Task type: {task_type}, Numeric tolerance: {numeric_tolerance}")
    print(f"{'='*70}")
    
    model.eval()
    im_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    for batch in tqdm(eval_loader, desc=split_name):
        try:
            answers = batch.pop('answers')
            questions = batch.pop('questions')
            
            model_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            input_len = model_inputs['input_ids'].shape[1]
            
            with torch.no_grad():
                output_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=actual_max_tokens,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id = im_end_token_id,
                )
            
            for i in range(output_ids.shape[0]):
                pred_text = processor.tokenizer.decode(output_ids[i, input_len:], skip_special_tokens=True)
                true_answer = str(answers[i])
                
                if is_regression:
                    # For STSB: extract numeric score
                    pred_num = extract_number(pred_text)
                    true_num = extract_number(true_answer)
                    
                    pred_score = pred_num if pred_num is not None else 0.0
                    true_score = true_num if true_num is not None else 0.0
                    
                    pred_scores.append(pred_score)
                    true_scores.append(true_score)
                    
                    # Track "close enough" accuracy
                    is_close = abs(pred_score - true_score) <= numeric_tolerance
                    correct += int(is_close)
                    total += 1
                    
                    results.append({
                        'true_answer': true_answer,
                        'prediction': pred_text,
                        'pred_score': pred_score,
                        'true_score': true_score,
                        'close': is_close,
                    })
                else:
                    # For classification: use smart matching
                    pred_answer = extract_answer(pred_text, true_answer)
                    is_correct, match_type = compare_answers(pred_answer, true_answer, numeric_tolerance)
                    
                    correct += int(is_correct)
                    total += 1
                    match_type_counts[match_type] += 1
                    
                    results.append({
                        'true_answer': true_answer,
                        'prediction': pred_text,
                        'pred_extracted': pred_answer,
                        'true_normalized': normalize_answer(true_answer),
                        'correct': is_correct,
                        'match_type': match_type,
                    })
                    
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compute metrics
    if is_regression:
        pearson = compute_pearson(pred_scores, true_scores)
        spearman = compute_spearman(pred_scores, true_scores)
        accuracy = correct / total if total > 0 else 0
        
        print(f"\n📊 {split_name}:")
        print(f"   Pearson:  {pearson:.4f}")
        print(f"   Spearman: {spearman:.4f}")
        print(f"   Close (±{numeric_tolerance}): {correct}/{total} = {accuracy*100:.2f}%")
        
        for j, r in enumerate(results[:5]):
            status = "✅" if r['close'] else "❌"
            print(f"   {status} True: {r['true_score']:.2f} | Pred: {r['pred_score']:.2f} | Raw: '{r['prediction'][:30]}'")
        
        return {
            'split_name': split_name,
            'task_type': task_type,
            'pearson': pearson,
            'spearman': spearman,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'results': results
        }
    else:
        accuracy = correct / total if total > 0 else 0
        
        print(f"\n📊 {split_name}: {correct}/{total} = {accuracy*100:.2f}%")
        print(f"   Match types: {dict(match_type_counts)}")
        
        for j, r in enumerate(results[:5]):
            status = "✅" if r['correct'] else "❌"
            match_info = f"[{r['match_type']}]" if r['correct'] else ""
            print(f"   {status} True: '{r['true_normalized']}' | Pred: '{r['pred_extracted']}' | Raw: '{r['prediction'][:40]}' {match_info}")
        
        return {
            'split_name': split_name,
            'task_type': task_type,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'match_type_counts': dict(match_type_counts),
            'results': results
        }


def run_full_evaluation(model, processor, dataset, data_root, 
                        num_samples_per_split=None, batch_size=4, num_video_frames=8, 
                        max_new_tokens=50, device="cuda", save_csv="evaluation_results.csv",
                        numeric_tolerance=0.5):
    test_splits = [name for name in dataset.keys() if name != 'train']
    
    sample_info = "ALL" if num_samples_per_split is None else num_samples_per_split
    print(f"\n{'='*70}")
    print(f"EVALUATION: {len(test_splits)} splits, {sample_info} samples each, batch_size={batch_size}")
    print(f"Numeric tolerance: {numeric_tolerance}")
    print(f"{'='*70}")
    
    all_results = {}
    category_results = defaultdict(list)
    
    for split_name in test_splits:
        try:
            result = evaluate_dataset_batched(
                model, processor, dataset[split_name], data_root, split_name,
                num_samples_per_split, batch_size, num_video_frames, max_new_tokens, 
                device, numeric_tolerance
            )
            all_results[split_name] = result
            
            category = '_'.join(split_name.split('_')[:2])
            category_results[category].append(result)
        except Exception as e:
            print(f"❌ Error on {split_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    for category, cat_results in sorted(category_results.items()):
        # Separate regression vs classification
        regression_results = [r for r in cat_results if r.get('task_type') == 'regression']
        classification_results = [r for r in cat_results if r.get('task_type') != 'regression']
        
        print(f"\n📁 {category.upper()}:")
        
        # Classification metrics
        if classification_results:
            cat_correct = sum(r['correct'] for r in classification_results)
            cat_total = sum(r['total'] for r in classification_results)
            cat_acc = cat_correct / cat_total * 100 if cat_total > 0 else 0
            print(f"   Classification: {cat_correct}/{cat_total} = {cat_acc:.2f}%")
            for r in classification_results:
                print(f"     - {r['split_name']}: {r['accuracy']*100:.2f}%")
        
        # Regression metrics
        if regression_results:
            avg_pearson = np.mean([r['pearson'] for r in regression_results])
            avg_spearman = np.mean([r['spearman'] for r in regression_results])
            print(f"   Regression: Pearson={avg_pearson:.4f}, Spearman={avg_spearman:.4f}")
            for r in regression_results:
                print(f"     - {r['split_name']}: Pearson={r['pearson']:.4f}, Spearman={r['spearman']:.4f}")
    
    # Overall stats
    classification_results = [r for r in all_results.values() if r.get('task_type') != 'regression']
    regression_results = [r for r in all_results.values() if r.get('task_type') == 'regression']
    
    if classification_results:
        all_correct = sum(r['correct'] for r in classification_results)
        all_total = sum(r['total'] for r in classification_results)
        overall_acc = all_correct / all_total * 100 if all_total > 0 else 0
        print(f"\n🎯 OVERALL ACCURACY: {all_correct}/{all_total} = {overall_acc:.2f}%")
        
        # Aggregate match types
        total_match_types = defaultdict(int)
        for r in classification_results:
            for mt, count in r.get('match_type_counts', {}).items():
                total_match_types[mt] += count
        print(f"   Match breakdown: {dict(total_match_types)}")
    
    if regression_results:
        avg_pearson = np.mean([r['pearson'] for r in regression_results])
        avg_spearman = np.mean([r['spearman'] for r in regression_results])
        print(f"🎯 OVERALL CORRELATION: Pearson={avg_pearson:.4f}, Spearman={avg_spearman:.4f}")
    
    # Save to CSV
    if save_csv:
        import csv
        with open(save_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset', 'task_type', 'metric', 'value', 'correct', 'total'])
            
            for name, r in all_results.items():
                if r.get('task_type') == 'regression':
                    writer.writerow([name, 'regression', 'pearson', f"{r['pearson']:.4f}", '', ''])
                    writer.writerow([name, 'regression', 'spearman', f"{r['spearman']:.4f}", '', ''])
                else:
                    writer.writerow([name, 'classification', 'accuracy', f"{r['accuracy']*100:.2f}", r['correct'], r['total']])
            
            # Overall
            if classification_results:
                writer.writerow(['OVERALL_CLASSIFICATION', '', 'accuracy', f"{overall_acc:.2f}", all_correct, all_total])
            if regression_results:
                writer.writerow(['OVERALL_REGRESSION', '', 'pearson', f"{avg_pearson:.4f}", '', ''])
                writer.writerow(['OVERALL_REGRESSION', '', 'spearman', f"{avg_spearman:.4f}", '', ''])
        
        print(f"\n💾 Results saved to: {save_csv}")
    
    return all_results


# ============================================================================
# Test functions to verify the fixes
# ============================================================================

def test_answer_matching():
    """Test the answer matching logic with the reported bug cases."""
    print("\n" + "="*70)
    print("TESTING ANSWER MATCHING FIXES")
    print("="*70)
    
    test_cases = [
        # (prediction, true_answer, expected_correct)
        # Multi-word answers
        ("passion flower", "passion flower", True),
        ("wild cat", "wild cat", True),
        ("passion flower is a beautiful plant", "passion flower", True),
        ("wild cat in the forest", "wild cat", True),
        
        # Trailing punctuation on numbers
        ("1976.", "1976", True),
        ("1976", "1976.", True),
        ("2023.", "2023", True),
        
        # Numeric tolerance
        ("18.38932", "18.32", True),  # diff = 0.069 < 0.5
        ("18.38932.", "18.32", True),  # with trailing dot
        ("5.7", "5.5", True),  # diff = 0.2 < 0.5
        ("10.0", "10", True),  # exact after float conversion
        ("5.0", "6.0", False),  # diff = 1.0 > 0.5
        
        # Single letter answers
        ("a", "a", True),
        ("a.", "a", True),
        ("b) is correct", "b", True),
        
        # Boolean answers - exact matches only
        ("true", "true", True),
        ("yes", "true", True),
        ("false", "false", True),
        ("no", "false", True),
        
        # NOT boolean - words starting with no/yes but aren't booleans
        ("not_duplicate", "not_duplicate", True),
        ("not_duplicate", "duplicate", False),
        ("duplicate", "duplicate", True),
        ("duplicate", "not_duplicate", False),
        ("notable", "notable", True),  # starts with "no" but isn't boolean
        ("yesterday", "yesterday", True),  # starts with "yes" but isn't boolean
        
        # Classification labels with underscores
        ("not_equivalent", "not_equivalent", True),
        ("entailment", "entailment", True),
        ("contradiction", "contradiction", True),
        
        # CoLA task - acceptable vs unacceptable (should NOT match each other!)
        ("acceptable", "acceptable", True),
        ("unacceptable", "unacceptable", True),
        ("acceptable", "unacceptable", False),  # BUG FIX: was incorrectly True via contains
        ("unacceptable", "acceptable", False),  # BUG FIX: was incorrectly True via contains
        
        # Extra tokens
        ("The answer is cat", "cat", True),
        ("dog is the answer", "dog", True),
        ("not_duplicate is the answer", "not_duplicate", True),
    ]
    
    passed = 0
    failed = 0
    
    for pred, true, expected in test_cases:
        pred_extracted = extract_answer(pred, true)
        is_correct, match_type = compare_answers(pred_extracted, true, tolerance=0.5)
        
        status = "✅" if is_correct == expected else "❌"
        if is_correct == expected:
            passed += 1
        else:
            failed += 1
        
        print(f"{status} Pred: '{pred}' | True: '{true}' | Extracted: '{pred_extracted}' | "
              f"Match: {is_correct} ({match_type}) | Expected: {expected}")
    
    print(f"\nResults: {passed}/{passed+failed} passed")
    return failed == 0


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_answer_matching()