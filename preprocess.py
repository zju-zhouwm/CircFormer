"""
Data processing, dataset definitions, and preprocessing script for the eccDNA model.

This file serves two purposes:
1.  It defines the `sequence_process` function and `EccDNADataset` class, which are
    imported by other modules for data handling.
2.  When run as a script, it preprocesses a raw CSV file into smaller, efficient
    binary batches (`.npy` files) for faster training.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from config import Config


def sequence_process(sequence: str, config: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tokenize a DNA sequence into k-mer IDs and generate a corresponding mask.

    This function segments the sequence into k-mers, trims it to a maximum
    length symmetrically if needed, adds special [START] and [END] tokens,
    pads it to `config.max_length`, and returns the token IDs and a boolean mask.
    """
    start_token, end_token = '[START]', '[END]'
    
    if config.step >= 2:
        sequence_kmers = [sequence[i:i + config.k] for i in range(0, len(sequence) - config.k + 1, config.step)]
    else:
        sequence_kmers = [sequence[i:i + config.k] for i in range(0, max(0, len(sequence) - config.k + 1), config.step)]
    
    max_seq_kmers = config.max_length - (config.k - 1)  # Max k-mers allowed
    if len(sequence_kmers) > max_seq_kmers:
        mid = max_seq_kmers // 2
        sequence_kmers = sequence_kmers[:mid] + sequence_kmers[-mid:]
        
    tokens = [start_token] + sequence_kmers + [end_token]
    
    tokens_id = [config.DNA_VOCAB.get(tok, config.UNK_ID) for tok in tokens]
    
    padding = [config.PAD_ID] * (config.max_length - len(tokens_id))
    tokens_id += padding
    
    mask = np.array(tokens_id) != config.PAD_ID
    
    return np.array(tokens_id, dtype=np.int64), mask


class CSVDataset(Dataset):
    """
    A PyTorch Dataset for eccDNA classification that loads data directly from CSV files.
    This performs k-mer encoding on-the-fly during data loading.
    
    Args:
        config (Config): Configuration object.
        csv_path (str): Path to the CSV file containing sequences and labels.
    """
    
    def __init__(self, config: Config, csv_path: str):
        self.config = config
        self.csv_path = csv_path
        
        # Load the CSV file
        try:
            df = pd.read_csv(csv_path)
            if 'sequences' not in df.columns or 'labels' not in df.columns:
                raise ValueError('CSV file must contain "sequences" and "labels" columns')
            self.df = df
        except Exception as e:
            raise ValueError(f"Error loading CSV file {csv_path}: {e}")
        
        self.total_len = len(self.df)
    
    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.total_len
    
    def __getitem__(self, idx: int):
        """
        Retrieves a sample from the dataset by its index.
        Performs k-mer encoding on-the-fly.
        """
        row = self.df.iloc[idx]
        sequence = str(row['sequences'])
        label = int(row['labels'])
        
        # Perform k-mer encoding
        tokens, mask = sequence_process(sequence, self.config)
        
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long), torch.as_tensor(mask, dtype=torch.bool)


class EccDNADataset(Dataset):
    """
    A PyTorch Dataset for eccDNA classification that loads data from preprocessed
    .npy files. This approach is memory-efficient as it avoids loading the entire
    dataset into RAM at once.

    Args:
        config (Config): Configuration object.
        encoded_dir (str): Directory containing the encoded .npy batches.
    """
    def __init__(self, config: Config, encoded_dir: str):
        self.encoded_dir = encoded_dir
        self.config = config

        # Find and sort all batch files for tokens, labels, and masks
        self.tokens_files = sorted([os.path.join(encoded_dir, f) for f in os.listdir(encoded_dir) if f.startswith('tokens_') and f.endswith('.npy')])
        self.labels_files = sorted([os.path.join(encoded_dir, f) for f in os.listdir(encoded_dir) if f.startswith('labels_') and f.endswith('.npy')])
        self.masks_files = sorted([os.path.join(encoded_dir, f) for f in os.listdir(encoded_dir) if f.startswith('masks_') and f.endswith('.npy')])

        # Ensure that the number of files for each component is consistent
        assert len(self.tokens_files) == len(self.labels_files) == len(self.masks_files), \
            "Mismatch in the number of tokens, labels, or masks files."

        # Calculate the number of samples in each file and the cumulative count
        self.file_lens = [np.load(f, mmap_mode='r').shape[0] for f in self.tokens_files]
        self.cum_lens = np.cumsum([0] + self.file_lens)
        self.total_len = sum(self.file_lens)

    def get_all_labels(self):
        """
        Loads and returns all labels from all batch files.
        This is useful for stratified splitting but should be used with caution
        on very large datasets as it loads all labels into memory.
        """

        all_labels = []
        for label_file in self.labels_files:
            all_labels.append(np.load(label_file))
        return np.concatenate(all_labels)

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.total_len

    def __getitem__(self, idx: int):
        """
        Retrieves a sample from the dataset by its index.

        This method identifies which file the sample belongs to, loads that file
        (or the relevant part) using memory-mapping, and returns the sample.
        """
        # Find which file contains the sample at index `idx`
        file_idx = np.searchsorted(self.cum_lens, idx, side='right') - 1
        
        # Calculate the local index within that file
        local_idx = idx - self.cum_lens[file_idx]
        
        # Load the specific sample from the files using memory-mapping for efficiency
        tokens = np.load(self.tokens_files[file_idx], mmap_mode='r')[local_idx]
        labels = np.load(self.labels_files[file_idx], mmap_mode='r')[local_idx]
        masks = np.load(self.masks_files[file_idx], mmap_mode='r')[local_idx]
        
        # Convert numpy arrays to torch tensors. Using .copy() is crucial because
        # memory-mapped arrays are read-only, but PyTorch tensors must be writable.
        return torch.tensor(tokens.copy(), dtype=torch.long), torch.tensor(labels.copy(), dtype=torch.long), torch.as_tensor(masks.copy(), dtype=torch.bool)

# This block allows the script to be run directly for preprocessing data.
if __name__ == '__main__':
    # --- Argument Parsing ---
    import argparse
    from tqdm import tqdm
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser(description="Preprocess eccDNA sequence data into encoded batches.")
    parser.add_argument(
        '-i', '--input', 
        help='Path to the input CSV file. Must contain "sequences" and "labels" columns.'
    )
    parser.add_argument(
        '-o', '--output_dir', 
        required=True, 
        help='Directory where the encoded .npy batches will be saved.'
    )
    parser.add_argument(
        '-bs', '--batch_size', 
        type=int, 
        default=1000, 
        help='The number of sequences to include in each encoded file. Default: 1000'
    )
    parser.add_argument(
        '-s', '--split',
        action='store_true',
        help='Split dataset into CSV files (train/val/test) with 7:2:1 ratio'
    )
    parser.add_argument(
        '-rs', '--random_seed', 
        type=int, 
        default=61,
        help='Random seed for dataset splitting. Default: 61'
    )
    parser.add_argument(
        '-ed', '--encode_dir',
        help='Directory containing CSV files to encode (train.csv, val.csv, test.csv)'
    )
    args = parser.parse_args()

    # Parameter validation
    if args.encode_dir:
        # When using -ed parameter, -i parameter is not needed
        if args.input:
            parser.error("Cannot use both -i and -ed arguments. Use -ed for batch encoding or -i for dataset splitting.")
    else:
        # When not using -ed parameter, -i and -s parameters are required
        if not args.input:
            parser.error("The -i argument is required when not using -ed.")
        if not args.split:
            parser.error("The -s argument is required when not using -ed. Use -s for dataset splitting.")

    def save_data_in_batches(df_to_process, output_directory, config, batch_size):
        """Processes a DataFrame and saves it in batches to the specified directory."""
        os.makedirs(output_directory, exist_ok=True)
        tokens, labels, masks = [], [], []
        batch_idx = 0
        
        print(f"Processing {len(df_to_process)} sequences for directory '{os.path.basename(output_directory)}'...")
        
        for i, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc=f"Encoding {os.path.basename(output_directory)}"):
            try:
                seq, label = row['sequences'], row['labels']
                t, m = sequence_process(seq, config)
                tokens.append(t)
                labels.append(int(label))
                masks.append(m)

                # Save a batch when it reaches the specified size
                if len(tokens) >= batch_size:
                    save_batch_to_file(tokens, labels, masks, batch_idx, output_directory)
                    batch_idx += 1
                    tokens, labels, masks = [], [], []

            except Exception as e:
                print(f"\nWarning: Skipping row {i} in {os.path.basename(output_directory)} due to an error: {e}")
        
        # Save any remaining data that didn't form a full batch
        if tokens:
            save_batch_to_file(tokens, labels, masks, batch_idx, output_directory)

    def save_batch_to_file(tokens_list, labels_list, masks_list, batch_index, output_directory):
        """Helper function to save numpy arrays to disk."""
        tokens_arr = np.array(tokens_list, dtype=np.int64)
        labels_arr = np.array(labels_list, dtype=np.int64)
        masks_arr = np.array(masks_list, dtype=bool)
        
        np.save(os.path.join(output_directory, f"tokens_{batch_index}.npy"), tokens_arr)
        np.save(os.path.join(output_directory, f"labels_{batch_index}.npy"), labels_arr)
        np.save(os.path.join(output_directory, f"masks_{batch_index}.npy"), masks_arr)
    
    # --- Main Preprocessing Logic ---
    config = Config()
    
    if args.encode_dir:
        # Batch encoding mode: encode CSV files in the specified directory
        print(f"Encoding CSV files from directory: {args.encode_dir}")
        
        csv_files = {
            'train': os.path.join(args.encode_dir, 'train.csv'),
            'val': os.path.join(args.encode_dir, 'val.csv'), 
            'test': os.path.join(args.encode_dir, 'test.csv')
        }
        
        for split_name, csv_file in csv_files.items():
            if os.path.exists(csv_file):
                print(f"\nEncoding {split_name} set from {csv_file}...")
                df_split = pd.read_csv(csv_file)
                output_split_dir = os.path.join(args.output_dir, split_name)
                save_data_in_batches(df_split, output_split_dir, config, args.batch_size)
                print(f"{split_name.capitalize()} set encoding complete: {len(df_split)} samples")
            else:
                print(f"Warning: {csv_file} not found, skipping {split_name} set")
        
        print(f"\nBatch encoding complete. All encoded data saved in '{args.output_dir}'.")
        
    else:
        # Single file processing mode: read input file
        try:
            # Check for a header by reading the first line of the file.
            with open(args.input, 'r') as f:
                first_line = f.readline()
            
            # Heuristic to decide if a header is present.
            has_header = 'sequences' in first_line.lower() and 'labels' in first_line.lower()

            if has_header:
                df = pd.read_csv(args.input)
                print("Header found in input file. Reading columns as is.")
            else:
                # If no header is detected, assume two columns and assign default names.
                df = pd.read_csv(args.input, header=None, names=['sequences', 'labels'])
                print("No header detected in input file. Assigning default column names: 'sequences', 'labels'.")

            # Final verification that the DataFrame has the required columns.
            if 'sequences' not in df.columns or 'labels' not in df.columns:
                raise ValueError('Failed to load data. Input file must contain "sequences" and "labels" columns, or be a two-column file without a header.')
                
        except FileNotFoundError:
            print(f"Error: The input file was not found at: {args.input}")
            exit(1)
        except Exception as e:
            print(f"An error occurred while reading the input file: {e}")
            exit(1)

        # --- Data Processing and Batching ---
        # Dataset splitting mode: only split into CSV files, no encoding
        print("Splitting dataset into train/val/test CSV files with 7:2:1 ratio...")
        
        # 7:2:1 stratified split
        train_df, temp_df = train_test_split(
            df, test_size=0.3, random_state=args.random_seed, stratify=df['labels']
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=1/3, random_state=args.random_seed, stratify=temp_df['labels']
        )
        
        # Save CSV files to output directory
        os.makedirs(args.output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(args.output_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(args.output_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(args.output_dir, 'test.csv'), index=False)
        
        print(f"\nDataset splitting complete!")
        print(f"Total samples: {len(df)}")
        print(f"Train set: {len(train_df)} samples -> {os.path.join(args.output_dir, 'train.csv')}")
        print(f"Validation set: {len(val_df)} samples -> {os.path.join(args.output_dir, 'val.csv')}")
        print(f"Test set: {len(test_df)} samples -> {os.path.join(args.output_dir, 'test.csv')}")
