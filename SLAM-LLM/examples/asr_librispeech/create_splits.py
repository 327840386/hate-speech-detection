import os
import json
import random
from typing import List, Dict
import math

def get_wav_files(directory: str) -> List[str]:
    """
    Get all wav files from a directory
    
    Args:
        directory: Path to the directory containing wav files
    
    Returns:
        List of wav file paths
    """
    return [f for f in os.listdir(directory) if f.endswith('.wav')]

def create_json_entry(filename: str, base_path: str, is_hate: bool) -> Dict:
    """
    Create a JSON entry for a wav file
    
    Args:
        filename: Name of the wav file
        base_path: Base path to the wav file directory
        is_hate: Boolean indicating if it's hate speech
    
    Returns:
        Dictionary containing the entry information
    """
    subfolder = "hate_wav" if is_hate else "non_hate_wav"
    full_path = os.path.join(base_path, subfolder, filename)
    return {
        "key": filename,
        "source": full_path,
        "hate_speech": "Yes" if is_hate else "No"
    }

def write_jsonl(data: List[Dict], output_file: str):
    """
    Write data to a JSONL file
    
    Args:
        data: List of dictionaries to write
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')

def main():
    # Base path for the data
    base_path = "/home/liu.ten/demo/SLAM-LLM/examples/asr_librispeech/database/new_data"
    
    # Output directory
    output_dir = "/home/liu.ten/demo/SLAM-LLM/examples/asr_librispeech"
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all wav files
    hate_files = get_wav_files(os.path.join(base_path, "hate_wav"))
    non_hate_files = get_wav_files(os.path.join(base_path, "non_hate_wav"))
    
    # Shuffle files
    random.seed(42)  # For reproducibility
    random.shuffle(hate_files)
    random.shuffle(non_hate_files)
    
    # Calculate split sizes (80%, 10%, 10%)
    n_hate = len(hate_files)
    n_train = math.floor(n_hate * 0.8)
    n_val = math.floor(n_hate * 0.1)
    n_test = n_hate - n_train - n_val
    
    # Split the files
    hate_train = hate_files[:n_train]
    hate_val = hate_files[n_train:n_train+n_val]
    hate_test = hate_files[n_train+n_val:]
    
    non_hate_train = non_hate_files[:n_train]
    non_hate_val = non_hate_files[n_train:n_train+n_val]
    non_hate_test = non_hate_files[n_train+n_val:]
    
    # Create JSON entries
    train_data = []
    for f in hate_train:
        train_data.append(create_json_entry(f, base_path, True))
    for f in non_hate_train:
        train_data.append(create_json_entry(f, base_path, False))
        
    val_data = []
    for f in hate_val:
        val_data.append(create_json_entry(f, base_path, True))
    for f in non_hate_val:
        val_data.append(create_json_entry(f, base_path, False))
        
    test_data = []
    for f in hate_test:
        test_data.append(create_json_entry(f, base_path, True))
    for f in non_hate_test:
        test_data.append(create_json_entry(f, base_path, False))
    
    # Shuffle the combined data
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    # Write to files
    write_jsonl(train_data, os.path.join(output_dir, "train_data.jsonl"))
    write_jsonl(val_data, os.path.join(output_dir, "validation_data.jsonl"))
    write_jsonl(test_data, os.path.join(output_dir, "hatespeech_test_clean.jsonl"))
    
    # Print statistics
    print(f"Created dataset splits in {output_dir}:")
    print(f"Training set: {len(train_data)} files ({len(hate_train)} hate, {len(non_hate_train)} non-hate)")
    print(f"Validation set: {len(val_data)} files ({len(hate_val)} hate, {len(non_hate_val)} non-hate)")
    print(f"Test set: {len(test_data)} files ({len(hate_test)} hate, {len(non_hate_test)} non-hate)")

if __name__ == "__main__":
    main()