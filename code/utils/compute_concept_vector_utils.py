from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
import os
import argparse
import numpy as np
from pathlib import Path

def get_model_type(tokenizer):
    """Detect model type from tokenizer (llama or qwen)"""
    model_name = tokenizer.name_or_path.lower()
    if "qwen" in model_name:
        return "qwen"
    else:
        return "llama"

def format_prompt(model_type, user_message, dataset_name=None, tokenizer=None):
    """Format prompt based on model type.

    The Llama branch keeps the literal rendering the committed
    data/saved_vectors/llama vectors were built with. The Qwen branch defers to the
    tokenizer's own chat template instead: Qwen3.5 prepends a reasoning-effort system
    message and opens a <think> block unless thinking is disabled, and a hand-written
    ChatML string would silently disagree with the behavioural prompt of Experiment 1.
    """
    if model_type == "qwen":
        if tokenizer is None:
            raise ValueError("the Qwen prompt format requires the tokenizer")
        content = (
            f"Tell me about {user_message}."
            if dataset_name == "simple_data"
            else user_message
        )
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:  # llama
        if dataset_name == "simple_data":
            return f"<|start_header_id|>user<|end_header_id|>Tell me about {user_message}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        else:
            return f"<|start_header_id|>user<|end_header_id|>{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

def get_data(dataset_name): 
    """Load raw data from json files"""
    # Dataset files live at the repository root, alongside code/.
    repo_dir = Path(__file__).resolve().parents[2]
    dataset_dir = repo_dir / "data" / "dataset"
    
    if dataset_name == "simple_data":
        with open(dataset_dir / "simple_data.json", "r") as f:
            data = json.load(f)
        return data
    elif dataset_name == "complex_data":
        with open(dataset_dir / "complex_data.json", "r") as f:
            data = json.load(f)
        return data


def compute_vector_single_prompt(model, tokenizer, dataset_name, steering_prompt, layer_idx):
    """
    Compute activation vector for a single prompt/sentence
    Based on Anthropic's introspection paper methodology
    
    Returns:
        prompt_last_vector: activation at last token (e.g., <end_header_id>)
        prompt_average_vector: average activation across all prompt tokens
    """
    model_type = get_model_type(tokenizer)
    prompt = format_prompt(model_type, steering_prompt, dataset_name, tokenizer=tokenizer)
    
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True) # [batch_size, seq_len, hidden_dim]
        # like the ":" token in "Assistant:", but for llama it is <end_header_id> token
        prompt_last_vector = outputs.hidden_states[layer_idx][:, prompt_len - 1, :].detach().cpu() 

        prompt_average_vector = outputs.hidden_states[layer_idx][:, :prompt_len, :].mean(dim=1).detach().cpu()
        # print(f"prompt_last_vector: {prompt_last_vector.shape}")
        # print(f"prompt_average_vector: {prompt_average_vector.shape}")
        del outputs 
    
    return prompt_last_vector, prompt_average_vector

def compute_vectors_all_layers(model, tokenizer, dataset_name, steering_prompt, layer_indices):
    """compute_vector_single_prompt for several hidden-state indices at once.

    One forward pass already produces every hidden state, so asking for N layers one
    at a time repeats the same pass N times. That is affordable for a 32-block 8B
    model and is not for a 64-block 27B one, where it is the difference between a few
    minutes and most of an hour.

    Returns:
        dict: {layer_idx: (prompt_last_vector, prompt_average_vector)}
    """
    model_type = get_model_type(tokenizer)
    prompt = format_prompt(model_type, steering_prompt, dataset_name, tokenizer=tokenizer)

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))

    vectors = {}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        for layer_idx in layer_indices:
            hidden = outputs.hidden_states[layer_idx]
            vectors[layer_idx] = (
                hidden[:, prompt_len - 1, :].detach().cpu(),
                hidden[:, :prompt_len, :].mean(dim=1).detach().cpu(),
            )
        del outputs

    return vectors


def _mean_by_layer(collected, layer_indices):
    """Stack each layer's per-prompt vectors and average them, as the single-layer path does."""
    means = {}
    for layer_idx in layer_indices:
        rows_last, rows_avg = zip(*collected[layer_idx])
        means[layer_idx] = (
            torch.stack(rows_last, dim=0).mean(dim=0).squeeze(),
            torch.stack(rows_avg, dim=0).mean(dim=0).squeeze(),
        )
    return means


def compute_concept_vectors_all_layers(model, tokenizer, dataset_name, layer_indices):
    """compute_concept_vector for several hidden-state indices in a single sweep.

    Same arithmetic as compute_concept_vector, same stack-then-mean accumulation
    order, but each prompt is run through the model once instead of once per layer.

    Returns:
        dict: {layer_idx: {concept_name: [prompt_last_vector, prompt_average_vector]}}
    """
    layer_indices = list(layer_indices)
    data = get_data(dataset_name)
    by_layer = {layer_idx: {} for layer_idx in layer_indices}

    def collect(prompts, desc):
        collected = {layer_idx: [] for layer_idx in layer_indices}
        for prompt in tqdm(prompts, desc=desc):
            vectors = compute_vectors_all_layers(
                model, tokenizer, dataset_name, prompt, layer_indices
            )
            for layer_idx in layer_indices:
                collected[layer_idx].append(vectors[layer_idx])
        return collected

    if dataset_name == "simple_data":
        concept_words = data["concept_vector_words"]
        baseline_words = data["baseline_words"][:50]

        print(f"Computing baseline mean from {len(baseline_words)} words "
              f"over {len(layer_indices)} layers...")
        baseline_mean = _mean_by_layer(
            collect(baseline_words, "Baseline vectors"), layer_indices
        )

        concept_vectors = {
            word: compute_vectors_all_layers(
                model, tokenizer, dataset_name, word, layer_indices
            )
            for word in tqdm(concept_words, desc="Concept vectors")
        }
        for word, vectors in concept_vectors.items():
            for layer_idx in layer_indices:
                vec_last, vec_avg = (row.squeeze() for row in vectors[layer_idx])
                base_last, base_avg = baseline_mean[layer_idx]
                by_layer[layer_idx][word] = [vec_last - base_last, vec_avg - base_avg]

    elif dataset_name == "complex_data":
        for concept_name in data.keys():
            pos_sentences, neg_sentences = data[concept_name][0], data[concept_name][1]
            print(f"\nProcessing {concept_name}: {len(pos_sentences)} pos, "
                  f"{len(neg_sentences)} neg over {len(layer_indices)} layers")
            pos_mean = _mean_by_layer(
                collect(pos_sentences, f"{concept_name} (positive)"), layer_indices
            )
            neg_mean = _mean_by_layer(
                collect(neg_sentences, f"{concept_name} (negative)"), layer_indices
            )
            for layer_idx in layer_indices:
                by_layer[layer_idx][concept_name] = [
                    pos_mean[layer_idx][0] - neg_mean[layer_idx][0],
                    pos_mean[layer_idx][1] - neg_mean[layer_idx][1],
                ]
    else:
        raise ValueError(f"unknown dataset {dataset_name!r}")

    print(f"\nComputed {len(by_layer[layer_indices[0]])} steering vectors per layer "
          f"for {len(layer_indices)} layers (each with last and avg variants)")
    return by_layer


def compute_concept_vector(model, tokenizer, dataset_name, layer_idx):
    """
    Compute steering vectors for all concepts in the dataset
    
    Args:
        model: the model to use
        tokenizer: the tokenizer to use
        dataset_name: "simple_data" or "complex_data"
        layer_idx: the layer index to compute steering vectors for
        
    Returns:
        dict: {concept_name: [prompt_last_steering_vector, prompt_average_steering_vector]}
        
    Method:
        - simple_data: For each word: vector(word) - mean(vector(baseline_word) for all baselines)
        - complex_data: For each concept: mean(vectors(pos_sentences)) - mean(vectors(neg_sentences))
    """
    data = get_data(dataset_name)
    steering_vectors = {}
    
    if dataset_name == "simple_data":
        concept_words = data["concept_vector_words"]
        baseline_words = data["baseline_words"][:50]
        
        # Compute baseline means once (used for all concepts)
        print(f"Computing baseline mean from {len(baseline_words)} words...")
        baseline_vecs_last = []
        baseline_vecs_avg = []
        for word in tqdm(baseline_words, desc="Baseline vectors"):
            vec_last, vec_avg = compute_vector_single_prompt(model, tokenizer, dataset_name, word, layer_idx)
            baseline_vecs_last.append(vec_last)
            baseline_vecs_avg.append(vec_avg)
        baseline_mean_last = torch.stack(baseline_vecs_last, dim=0).mean(dim=0).squeeze() # shape [hidden_dim]
        baseline_mean_avg = torch.stack(baseline_vecs_avg, dim=0).mean(dim=0).squeeze() # shape [hidden_dim]
        
        # Compute steering vectors for each concept word
        for word in tqdm(concept_words, desc="Concept vectors"):
            vec_last, vec_avg = compute_vector_single_prompt(model, tokenizer, dataset_name, word, layer_idx)
            vec_last = vec_last.squeeze() # shape [hidden_dim]
            vec_avg = vec_avg.squeeze() # shape [hidden_dim]
            steering_vectors[word] = [vec_last - baseline_mean_last, vec_avg - baseline_mean_avg]
            
    elif dataset_name == "complex_data":
        # For each concept: mean(positive) - mean(negative)
        print(f"data keys: {data.keys()}")
        for concept_name in data.keys():
            print(f"concept_name: {concept_name}")
            pos_sentences = data[concept_name][0]  # List of positive examples
            neg_sentences = data[concept_name][1]  # List of negative examples
            
            print(f"\nProcessing {concept_name}: {len(pos_sentences)} pos, {len(neg_sentences)} neg")
            
            # Compute mean of positive sentences
            pos_vecs_last = []
            pos_vecs_avg = []
            for sentence in tqdm(pos_sentences, desc=f"{concept_name} (positive)"):
                vec_last, vec_avg = compute_vector_single_prompt(model, tokenizer, dataset_name, sentence, layer_idx)
                pos_vecs_last.append(vec_last)
                pos_vecs_avg.append(vec_avg)
            pos_mean_last = torch.stack(pos_vecs_last, dim=0).mean(dim=0).squeeze()
            pos_mean_avg = torch.stack(pos_vecs_avg, dim=0).mean(dim=0).squeeze()
            
            # Compute mean of negative sentences
            neg_vecs_last = []
            neg_vecs_avg = []
            for sentence in tqdm(neg_sentences, desc=f"{concept_name} (negative)"):
                vec_last, vec_avg = compute_vector_single_prompt(model, tokenizer, dataset_name, sentence, layer_idx)
                neg_vecs_last.append(vec_last)
                neg_vecs_avg.append(vec_avg)
            neg_mean_last = torch.stack(neg_vecs_last, dim=0).mean(dim=0).squeeze()
            neg_mean_avg = torch.stack(neg_vecs_avg, dim=0).mean(dim=0).squeeze()
            
            # Steering vectors = positive - negative (both last and avg)
            steering_vectors[concept_name] = [pos_mean_last - neg_mean_last, pos_mean_avg - neg_mean_avg]
    
    print(f"\nComputed {len(steering_vectors)} steering vectors (each with last and avg variants)")
    return steering_vectors
        
        
    
# sweep every 4 layers for now
def sweep_layers(model, tokenizer, dataset_name, layer_indices=[i for i in range(32) if i % 4 == 0], save_dir="concept_vectors"):
    """
    Sweep through all layers and compute steering vectors for each layer
    
    Args:
        model: the model to use
        tokenizer: the tokenizer to use
        dataset_name: "simple_data" or "complex_data"
        layer_indices: list of layer indices to compute steering vectors for
        save_dir: base directory to save vectors (default: "concept_vectors")
        
    Returns:
        dict: {layer_idx: {concept_name: [prompt_last_vec_numpy, prompt_avg_vec_numpy]}}
    """
    all_steering_vectors = {}   
    for layer_idx in layer_indices:
        print(f"\n{'='*60}")
        print(f"PROCESSING LAYER {layer_idx}")
        print(f"{'='*60}")
        
        # Get dictionary of concept vectors for this layer
        concept_vecs = compute_concept_vector(model, tokenizer, dataset_name, layer_idx)
        
        # Save vectors to disk
        for concept_name, (vec_last, vec_avg) in concept_vecs.items():
            concept_dir = Path(save_dir) / concept_name
            concept_dir.mkdir(parents=True, exist_ok=True)
            torch.save(vec_last, concept_dir / f"layer_{layer_idx}_prompt_last.pt")
            torch.save(vec_avg, concept_dir / f"layer_{layer_idx}_prompt_average.pt")
        
        # Convert tensors to numpy - each concept has [last_vec, avg_vec]
        all_steering_vectors[layer_idx] = {
            concept: [vec_list[0].numpy(), vec_list[1].numpy()] 
            for concept, vec_list in concept_vecs.items()
        }
    
    return all_steering_vectors