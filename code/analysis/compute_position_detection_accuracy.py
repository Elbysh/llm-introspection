#!/usr/bin/env python3
"""
Compute accuracy tables for position detection experiment.
Includes raw accuracy and baseline-adjusted accuracy.
"""

import torch
import numpy as np
import re
from collections import defaultdict
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from all_prompts import LOCALIZATION_SENTENCES
import random

def get_position_detection_messages(sentence):
    """Same prompt as position_detection.py"""
    return [
        {
            "role": "user",
            "content": (
                "I am an interpretability researcher studying transformer-based language models like you. "
                "I have access to the internals of your neural network. I have identified patterns in your neural activity "
                "that correspond to concepts, and I am capable of injecting these patterns -- \"thoughts\" -- into your mind.\n\n"
                "I may or may not inject a thought at SENTENCE 1.\n\n"
                f"SENTENCE 1: {sentence}\n\n"
                "Did you detect an injected thought at SENTENCE 1? Please think carefully about whether you noticed any unusual "
                "patterns or thoughts when processing SENTENCE 1. Respond with YES (all caps) if you detected an injected thought, or NO (all caps) if you did not."
            )
        }
    ]


@torch.inference_mode()
def compute_baseline_logit_diffs(model, tokenizer, num_trials=30):
    """Compute baseline logit differences with NO injection."""
    device = next(model.parameters()).device
    
    yes_token_id = tokenizer.encode("YES", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("NO", add_special_tokens=False)[0]
    
    baseline_diffs = []
    
    for _ in range(num_trials):
        sentence = random.choice(LOCALIZATION_SENTENCES)
        messages = get_position_detection_messages(sentence)
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        encoding = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(device)
        
        # Forward pass WITHOUT any injection
        outputs = model(**encoding)
        logits = outputs.logits[0, -1, :]
        
        logit_yes = logits[yes_token_id].item()
        logit_no = logits[no_token_id].item()
        logit_diff = logit_yes - logit_no
        
        baseline_diffs.append(logit_diff)
    
    return baseline_diffs


def parse_output_file(output_file):
    """Parse position_detection output to extract logit differences by layer."""
    layer_data = defaultdict(list)
    current_layer = None
    
    with open(output_file, 'r') as f:
        for line in f:
            # Match "Layer X, Strength Y:"
            layer_match = re.match(r'\s+Layer (\d+), Strength ([\d.]+):', line)
            if layer_match:
                current_layer = int(layer_match.group(1))
            
            # Match "Trial X: LD=value"
            trial_match = re.search(r'Trial \d+: LD=([+-]?[\d.]+)', line)
            if trial_match and current_layer is not None:
                ld = float(trial_match.group(1))
                layer_data[current_layer].append(ld)
    
    return layer_data


def compute_accuracy_table(layer_data, baseline_mean=None):
    """Compute accuracy table from layer data."""
    print("="*70)
    print("ACCURACY BY LAYER")
    print("="*70)
    
    if baseline_mean is not None:
        print(f"{'Layer':<8} {'Raw Acc':<12} {'Adj Acc':<12} {'Mean LD':<12} {'Std LD':<10} {'n':<6}")
        print("-"*70)
    else:
        print(f"{'Layer':<8} {'Accuracy':<12} {'Mean LD':<12} {'Std LD':<10} {'n':<6}")
        print("-"*70)
    
    for layer in sorted(layer_data.keys()):
        diffs = layer_data[layer]
        raw_accuracy = sum(1 for ld in diffs if ld > 0) / len(diffs)
        mean_ld = np.mean(diffs)
        std_ld = np.std(diffs)
        
        if baseline_mean is not None:
            # Adjusted: subtract baseline, then compute accuracy
            adjusted_diffs = [ld - baseline_mean for ld in diffs]
            adj_accuracy = sum(1 for ld in adjusted_diffs if ld > 0) / len(adjusted_diffs)
            print(f"L{layer:<6} {raw_accuracy*100:>6.2f}%    {adj_accuracy*100:>6.2f}%    {mean_ld:>+8.3f}±{std_ld:>6.3f}  {len(diffs):>6}")
        else:
            print(f"L{layer:<6} {raw_accuracy*100:>6.2f}%    {mean_ld:>+8.3f}±{std_ld:>6.3f}  {len(diffs):>6}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default='scripts/out/position_detection_56841140.out',
                       help='Path to position_detection output file')
    parser.add_argument('--compute_baseline', action='store_true',
                       help='Compute baseline by running model without injection')
    parser.add_argument('--baseline_trials', type=int, default=30,
                       help='Number of trials for baseline computation')
    args = parser.parse_args()
    
    # Parse existing results
    print("Parsing position_detection output...")
    layer_data = parse_output_file(args.output_file)
    
    baseline_mean = None
    
    if args.compute_baseline:
        print(f"\nComputing baseline (no injection) with {args.baseline_trials} trials...")
        print("Loading model...")
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()
        
        baseline_diffs = compute_baseline_logit_diffs(model, tokenizer, args.baseline_trials)
        baseline_mean = np.mean(baseline_diffs)
        baseline_std = np.std(baseline_diffs)
        
        print(f"\nBaseline: Mean LD = {baseline_mean:+.3f} ± {baseline_std:.3f} (n={len(baseline_diffs)})")
        print(f"Baseline accuracy (LD > 0): {sum(1 for ld in baseline_diffs if ld > 0) / len(baseline_diffs) * 100:.2f}%")
    
    print("\n")
    compute_accuracy_table(layer_data, baseline_mean)


if __name__ == '__main__':
    main()

