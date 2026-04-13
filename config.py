"""Configuration and vocabulary utilities for eccDNA model.

Defines the training/inference hyperparameters and builds the DNA k-mer
vocabulary with special tokens.
"""

import os
import torch
from dataclasses import dataclass
from itertools import product
from typing import Dict, Tuple,  Union



def build_DNA_vocab(k: int = 3) -> Tuple[Dict[str, int], int, int]:
	"""Build DNA k-mer vocabulary.

	Args:
		k: Length of k-mers to generate from bases A,G,C,T.

	Returns:
		A tuple of (vocab dict, PAD_ID, UNK_ID).
	"""
	special_tokens = {'[START]': 0, '[END]': 1, '[PAD]': 2, '[UNKNOW]': 3}
	bases = ['A', 'G', 'C', 'T']
	kmers = [''.join(i) for i in product(bases, repeat=k)]
	kmers.sort()
	vocab = dict(special_tokens)
	vocab.update({kmer: idx + 4 for idx, kmer in enumerate(kmers)})
	PAD_ID = vocab['[PAD]']
	UNK_ID = vocab['[UNKNOW]']
	return vocab, PAD_ID, UNK_ID


@dataclass
class Config:
	"""Hyperparameters and runtime options.

	Attributes mirror CLI options; device is auto-detected. On init the DNA
	vocabulary and special token ids are constructed from k.
	"""
	embedding_dim: int = 256
	hidden_dim: int = 768
	num_head: int = 8
	dropout: float = 0.1
	gamma: float = 0.1
	conv_channels: int = 256
	kernel_size: int = 7
	num_layers: int = 4
	output_dim: int = 2
	k: int = 6
	step: int = 1
	max_length: int = 2048
	lr: float = 2e-5
	epochs: int = 20
	batch_size: int = 32
	data_path: Union[str, None] = None  # Legacy path to raw CSV, now unused for training
	encoded_train_dir: Union[str, None] = None  # Directory for encoded training data batches
	encoded_val_dir: Union[str, None] = None  # Directory for encoded validation data batches (optional)
	encoded_test_dir: Union[str, None] = None # Directory for encoded test data batches (optional)
	test_csv: Union[str, None] = None # CSV file for testing (alternative to encoded_test_dir)
	seed: int = 61
	device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	save_path: Union[str, None] = None
	log_path: Union[str, None] = None
	save_every_epoch: bool = True  # Whether to save model for each epoch
	checkpoint_dir: Union[str, None] = None  # Checkpoint directory to save all epoch models
	early_stop_patience: int = 3  # Early stopping patience
	early_stop_min_delta: float = 0.001  # Early stopping minimum improvement threshold

	def __post_init__(self):
		"""Initialize vocabulary and special token ids based on current k."""
		self.DNA_VOCAB, self.PAD_ID, self.UNK_ID = build_DNA_vocab(k=self.k)
