
"""
Training and evaluation utilities for the eccDNA classification model.

Includes robust logging, type annotations, seed control, and flexible metrics.
"""


import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import logging
import random
import numpy as np
import shutil
from config import Config
from model import EccFormer
from preprocess import EccDNADataset, sequence_process,CSVDataset
import pandas as pd
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score



def set_seed(seed: int = 42) -> None:
	"""Set random seed for reproducibility."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

def train_model(config: Config) -> nn.Module:
	"""
	Train the EccFormer model, save the best checkpoint, and log results.
	Args:
		config (Config): Model configuration object.
	Returns:
		nn.Module: The trained model.
	"""
	logger = logging.getLogger("eccdna.trainer")
	set_seed(config.seed)
	num_workers = getattr(config, 'num_workers', 4)
	if not config.encoded_train_dir:
		logger.error("The --encoded_train_dir argument is required for training.")
		raise ValueError("The --encoded_train_dir argument is required for training.")
	if not config.save_path:
		logger.error("The --save_path argument is required for training.")
		raise ValueError("The --save_path argument is required for training.")
	if not config.log_path:
		logger.error("The --log_path argument is required for training.")
		raise ValueError("The --log_path argument is required for training.")
	train_dataset = EccDNADataset(config=config, encoded_dir=config.encoded_train_dir)
	if config.encoded_val_dir:
		val_dataset = EccDNADataset(config=config, encoded_dir=config.encoded_val_dir)
		train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
		val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
	else:
		logger.warning("No validation set provided. Splitting 10% of the training data for validation.")
		indices = list(range(len(train_dataset)))
		try:
			labels_for_stratify = train_dataset.get_all_labels()
		except AttributeError:
			logger.warning("Dataset does not support fetching all labels for stratification. Splitting randomly.")
			labels_for_stratify = None
		train_indices, val_indices = train_test_split(indices, test_size=0.1, random_state=config.seed, stratify=labels_for_stratify)
		train_sampler = SubsetRandomSampler(train_indices)
		val_sampler = SubsetRandomSampler(val_indices)
		train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
		val_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=True)
	model = EccFormer(config).to(config.device)
	if torch.cuda.device_count() > 1:
		logger.info(f"Using {torch.cuda.device_count()} GPUs for training.")
		model = nn.DataParallel(model)
	else:
		logger.info(f"Using single GPU: {config.device}")
	optimizer = optim.AdamW(model.parameters(), lr=config.lr)
	criterion = nn.CrossEntropyLoss()
	val_accuracy_metric = BinaryAccuracy().to(config.device)
	val_precision_metric = BinaryPrecision().to(config.device)
	val_recall_metric = BinaryRecall().to(config.device)
	val_f1_metric = BinaryF1Score().to(config.device)
	best_val_accuracy = 0.0
	log_file = config.log_path
	with open(log_file, 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1'])
	# Initialize checkpoint directory
	if config.save_every_epoch:
		if config.checkpoint_dir is None and config.save_path is not None:
			# Automatically derive checkpoint directory from save_path
			save_dir = os.path.dirname(config.save_path)
			if save_dir == "":
				save_dir = "."
			config.checkpoint_dir = os.path.join(save_dir, "checkpoints")
		if config.checkpoint_dir:
			os.makedirs(config.checkpoint_dir, exist_ok=True)
			logger.info(f"Checkpoint directory: {config.checkpoint_dir}")
	
	logger.info(f"Starting training for {config.epochs} epochs on device {config.device}.")
	logger.info(f"Log file will be saved to: {log_file}")
	
	# Early stopping related variables
	no_improve_count = 0
	best_val_accuracy = 0.0
	
	for epoch in tqdm(range(config.epochs), desc="Epochs"):
		model.train()
		total_train_loss = 0
		train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Training]", leave=False)
		for tokens, labels, masks in train_iterator:
			tokens, labels, masks = tokens.to(config.device), labels.to(config.device), masks.to(config.device)
			optimizer.zero_grad()
			outputs = model(tokens, masks)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			total_train_loss += loss.item()
			train_iterator.set_postfix(loss=loss.item())
		avg_train_loss = total_train_loss / len(train_loader)
		model.eval()
		total_val_loss = 0
		val_accuracy_metric.reset()
		val_precision_metric.reset()
		val_recall_metric.reset()
		val_f1_metric.reset()
		val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Validation]", leave=False)
		with torch.no_grad():
			for tokens, labels, masks in val_iterator:
				tokens, labels, masks = tokens.to(config.device), labels.to(config.device), masks.to(config.device)
				outputs = model(tokens, masks)
				loss = criterion(outputs, labels)
				total_val_loss += loss.item()
				preds = torch.argmax(outputs, dim=1)
				val_accuracy_metric.update(preds, labels)
				val_precision_metric.update(preds, labels)
				val_recall_metric.update(preds, labels)
				val_f1_metric.update(preds, labels)
		avg_val_loss = total_val_loss / len(val_loader)
		val_accuracy = val_accuracy_metric.compute()
		val_precision = val_precision_metric.compute()
		val_recall = val_recall_metric.compute()
		val_f1 = val_f1_metric.compute()
		logger.info(
			f"Epoch {epoch+1}/{config.epochs} | "
			f"Train Loss: {avg_train_loss:.4f} | "
			f"Val Loss: {avg_val_loss:.4f} | "
			f"Val Acc: {val_accuracy:.4f} | "
			f"Val Precision: {val_precision:.4f} | "
			f"Val Recall: {val_recall:.4f} | "
			f"Val F1: {val_f1:.4f}"
		)
		
		# Get model state
		model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
		
		# Save current epoch model
		if config.save_every_epoch and config.checkpoint_dir:
			epoch_model_path = os.path.join(config.checkpoint_dir, f"model_epoch_{epoch+1:03d}.pth")
			torch.save(model_state, epoch_model_path)
			logger.info(f"Epoch model saved to {epoch_model_path}")
		
		# Check if it's the best model (considering min_delta)
		improvement = val_accuracy - best_val_accuracy
		if improvement > config.early_stop_min_delta:
			best_val_accuracy = val_accuracy
			no_improve_count = 0
			
			# Save best model to config.save_path
			torch.save(model_state, config.save_path)
			logger.info(f"Best model saved to {config.save_path} with accuracy: {best_val_accuracy:.4f}")
			
			# Also save best model copy in checkpoint directory
			if config.save_every_epoch and config.checkpoint_dir:
				best_model_copy = os.path.join(config.checkpoint_dir, "model_best.pth")
				shutil.copy2(config.save_path, best_model_copy)
				logger.info(f"Best model copy saved to {best_model_copy}")
		else:
			no_improve_count += 1
			logger.info(f"No improvement for {no_improve_count} epoch(s) (improvement: {improvement:.6f}, min_delta: {config.early_stop_min_delta}")
		
		# Record log
		with open(log_file, 'a', newline='') as f:
			writer = csv.writer(f)
			writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, val_accuracy.item(), val_precision.item(), val_recall.item(), val_f1.item()])
		
		# Early stopping check
		if no_improve_count >= config.early_stop_patience:
			logger.info(f"Early stopping triggered after {epoch+1} epochs (patience: {config.early_stop_patience})")
			break
	
	logger.info(f"Training complete. Best validation accuracy: {best_val_accuracy:.4f}")
	if config.save_every_epoch and config.checkpoint_dir:
		logger.info(f"All epoch models saved in: {config.checkpoint_dir}")
	return model



def test_model(config: Config) -> Dict[str, Any]:
	"""
	Evaluate a trained model on a separate test set.
	Supports both pre-encoded data directories and raw CSV files.
	Args:
		config (Config): Model configuration object.
	Returns:
		Dict[str, Any]: Dictionary of test metrics.
	"""
	logger = logging.getLogger("eccdna.trainer")
	logger.info("--- Starting Evaluation on Test Set ---")
	num_workers = getattr(config, 'num_workers', 4)
	
	# Check if either encoded_test_dir or test_csv is provided
	if not config.encoded_test_dir and not config.test_csv:
		logger.error("Either --encoded_test_dir or --test_csv argument is required for testing.")
		raise ValueError("Either --encoded_test_dir or --test_csv argument is required for testing.")
	
	try:
		if config.encoded_test_dir:
			# Use pre-encoded data
			test_dataset = EccDNADataset(config=config, encoded_dir=config.encoded_test_dir)
			logger.info(f"Loaded {len(test_dataset)} samples from encoded directory: {config.encoded_test_dir}")
		else:
			# Use raw CSV file
			test_dataset = CSVDataset(config=config, csv_path=config.test_csv)
			logger.info(f"Loaded {len(test_dataset)} samples from CSV file: {config.test_csv}")
		
		test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
	except (FileNotFoundError, AssertionError, ValueError) as e:
		logger.error(f"Error loading test data: {e}")
		return {}
	model = EccFormer(config).to(config.device)
	if not os.path.exists(config.save_path):
		logger.error(f"Model checkpoint not found at {config.save_path}")
		return {}
	try:
		model.load_state_dict(torch.load(config.save_path, map_location=config.device))
		if torch.cuda.device_count() > 1:
			model = nn.DataParallel(model)
		logger.info(f"Successfully loaded model from {config.save_path}")
	except Exception as e:
		logger.error(f"Error loading model state: {e}")
		return {}
	test_accuracy_metric = BinaryAccuracy().to(config.device)
	test_precision_metric = BinaryPrecision().to(config.device)
	test_recall_metric = BinaryRecall().to(config.device)
	test_f1_metric = BinaryF1Score().to(config.device)
	model.eval()
	test_iterator = tqdm(test_loader, desc="[Testing]")
	with torch.no_grad():
		for tokens, labels, masks in test_iterator:
			tokens, labels, masks = tokens.to(config.device), labels.to(config.device), masks.to(config.device)
			outputs = model(tokens, masks)
			preds = torch.argmax(outputs, dim=1)
			test_accuracy_metric.update(preds, labels)
			test_precision_metric.update(preds, labels)
			test_recall_metric.update(preds, labels)
			test_f1_metric.update(preds, labels)
	test_accuracy = test_accuracy_metric.compute()
	test_precision = test_precision_metric.compute()
	test_recall = test_recall_metric.compute()
	test_f1 = test_f1_metric.compute()
	logger.info("\n--- Test Set Performance ---")
	logger.info(f"Accuracy:  {test_accuracy:.4f}")
	logger.info(f"Precision: {test_precision:.4f}")
	logger.info(f"Recall:    {test_recall:.4f}")
	logger.info(f"F1-Score:  {test_f1:.4f}")
	logger.info("--------------------------\n")
	return {
		"accuracy": float(test_accuracy),
		"precision": float(test_precision),
		"recall": float(test_recall),
		"f1": float(test_f1),
	}
