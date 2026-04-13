#!/usr/bin/env python3
"""
ROC plot data generation script

Input: Model file + CSV data file (containing DNA sequences and labels)
Output: ROC plot data CSV file (FPR, TPR, thresholds, AUC)

Usage example:
python roc_plot_data_generator.py -m model.pth -i roc.csv -o roc_plot_data.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
import logging
from typing import Dict, Any, Tuple, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("roc_plot_generator")


def load_model(model_path: str):
    """
    Load EccFormer model.
    
    Args:
        model_path: Path to model file.
        
    Returns:
        tuple: (model, config)
    """
    try:
        from config import Config
        from model import EccFormer
        
        config = Config()
        model = EccFormer(config).to(config.device)
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=config.device))
            logger.info(f"Successfully loaded pretrained model: {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model.eval()
        return model, config
        
    except ImportError as e:
        logger.error(f"Failed to import project modules: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)


def process_sequences(sequences: List[str], config):
    """
    Process DNA sequences into model input format.
    
    Args:
        sequences: List of DNA sequences.
        config: Model configuration.
        
    Returns:
        tuple: (tokens, masks)
    """
    try:
        from preprocess import sequence_process
        
        tokens_list = []
        masks_list = []
        
        for seq in sequences:
            tokens, mask = sequence_process(seq, config)
            tokens_list.append(tokens)
            masks_list.append(mask)
            
        tokens_array = np.array(tokens_list, dtype=np.int64)
        masks_array = np.array(masks_list, dtype=bool)
        
        tokens_tensor = torch.tensor(tokens_array, dtype=torch.long)
        masks_tensor = torch.tensor(masks_array, dtype=torch.bool)
        
        return tokens_tensor, masks_tensor
        
    except Exception as e:
        logger.error(f"Sequence processing failed: {e}")
        sys.exit(1)


def predict_with_model(model, tokens: torch.Tensor, masks: torch.Tensor, config):
    """
    Perform batch prediction using the model.
    
    Args:
        model: EccFormer model.
        tokens: Input tokens.
        masks: Attention masks.
        config: Model configuration.
        
    Returns:
        np.ndarray: Predicted probability array.
    """
    try:
        all_probs = []
        batch_size = config.batch_size
        
        with torch.no_grad():
            for i in range(0, len(tokens), batch_size):
                batch_tokens = tokens[i:i+batch_size].to(config.device)
                batch_masks = masks[i:i+batch_size].to(config.device)
                
                outputs = model(batch_tokens, batch_masks)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                
        return np.array(all_probs)
        
    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        sys.exit(1)


def load_test_data(input_file: str) -> Tuple[List[str], np.ndarray]:
    """
    Load test data from CSV file.
    
    Args:
        input_file: Path to input file.
        
    Returns:
        tuple: (sequences, true_labels)
    """
    try:
        df = pd.read_csv(input_file)
        logger.info(f"Successfully loaded data: {len(df)} samples")
        
        if 'labels' in df.columns:
            df['labels'] = pd.to_numeric(df['labels'], errors='coerce')
        elif 'label' in df.columns:
            df['label'] = pd.to_numeric(df['label'], errors='coerce')
            df = df.rename(columns={'label': 'labels'})
        
        sequences = df['sequences'].tolist()
        true_labels = df['labels'].values
        
        positive_count = np.sum(true_labels)
        negative_count = len(true_labels) - positive_count
        logger.info(f"Positive samples: {positive_count}, Negative samples: {negative_count}")
        
        return sequences, true_labels
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)


def generate_roc_plot_data(true_labels: np.ndarray, pred_probs: np.ndarray, output_file: str):
    """
    Generate ROC plot data and save to CSV file.
    
    Args:
        true_labels: Array of true labels.
        pred_probs: Array of predicted probabilities.
        output_file: Output file path.
    """
    try:
        fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
        roc_auc = auc(fpr, tpr)
        
        logger.info(f"ROC plot data generated")
        logger.info(f"AUC: {roc_auc:.4f}")
        logger.info(f"FPR range: [{fpr.min():.4f}, {fpr.max():.4f}]")
        logger.info(f"TPR range: [{tpr.min():.4f}, {tpr.max():.4f}]")
        logger.info(f"Threshold count: {len(thresholds)}")
        
        min_len = min(len(fpr), len(tpr), len(thresholds) + 1)
        
        roc_plot_df = pd.DataFrame({
            'fpr': fpr[:min_len],
            'tpr': tpr[:min_len],
            'thresholds': np.append(thresholds[:min_len-1], np.nan)
        })
        
        roc_plot_df.to_csv(output_file, index=False)
        logger.info(f"ROC plot data saved to: {output_file}")
        
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n# AUC: {roc_auc:.4f}\n")
            f.write(f"# Sample count: {len(true_labels)}\n")
            f.write(f"# Positive ratio: {np.mean(true_labels):.4f}\n")
            f.write(f"# ROC data points: {len(fpr)}\n")
        
        print(f"\n=== ROC Plot Data Generation Complete ===")
        print(f"Output file: {output_file}")
        print(f"AUC: {roc_auc:.4f}")
        print(f"Sample count: {len(true_labels)}")
        print(f"Positive ratio: {np.mean(true_labels):.4f}")
        print(f"ROC data points: {len(fpr)}")
        
    except Exception as e:
        logger.error(f"Failed to generate ROC plot data: {e}")
        sys.exit(1)


def main():
    """Main function for ROC plot data generation."""
    parser = argparse.ArgumentParser(
        description="ROC plot data generation script",
        epilog="""
Example usage:
  python roc_plot_data_generator.py -m model.pth -i roc.csv -o roc_plot_data.csv
        """
    )
    
    parser.add_argument(
        '-m', '--model', 
        required=True,
        help='Pretrained model file path (.pth)'
    )
    
    parser.add_argument(
        '-i', '--input', 
        required=True,
        help='Input data file path (CSV format containing DNA sequences and labels)'
    )
    
    parser.add_argument(
        '-o', '--output', 
        required=True,
        help='Output file path (ROC plot data CSV file will be saved here)'
    )
    
    parser.add_argument(
        '-b', '--batch_size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    logger.info("Starting ROC plot data generation...")
    logger.info(f"Model file: {args.model}")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Batch size: {args.batch_size}")
    
    try:
        logger.info("Loading EccFormer model...")
        model, config = load_model(args.model)
        config.batch_size = args.batch_size
        
        logger.info("Loading test data...")
        sequences, true_labels = load_test_data(args.input)
        
        logger.info("Processing DNA sequence data...")
        tokens, masks = process_sequences(sequences, config)
        
        logger.info("Running model predictions...")
        pred_probs = predict_with_model(model, tokens, masks, config)
        
        logger.info("Generating ROC plot data...")
        generate_roc_plot_data(true_labels, pred_probs, args.output)
        
        logger.info("ROC plot data generation complete!")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

