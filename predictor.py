"""Prediction utilities: scoring FASTA candidates and writing positives."""

from typing import Dict, List, Optional
import numpy as np
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from config import Config
from model import EccFormer
from preprocess import sequence_process



from tqdm import tqdm

def predict_fasta(
	model_path: str,
	fasta_path: str,
	output_fasta: str,
	config: Optional[Config] = None,
	batch_size: int = 512,
	output_prob_tsv: Optional[str] = None
) -> None:
	"""
	Predicts which sequences in a FASTA file are positive for eccDNA using a trained model checkpoint.
	Writes the positive sequences to a new FASTA file. Optionally outputs a TSV with all predictions and probabilities.
	Args:
		model_path: Path to trained model checkpoint (.pth)
		fasta_path: Input FASTA file
		output_fasta: Output FASTA file for positive predictions
		config: Optional Config object
		batch_size: Batch size for prediction (default: 512)
		output_prob_tsv: Optional path to write all predictions and probabilities as TSV
	"""
	import os
	if not os.path.exists(model_path):
		raise FileNotFoundError(f"Error: Model checkpoint not found at '{model_path}'")
	if not os.path.exists(fasta_path):
		raise FileNotFoundError(f"Error: Input FASTA file not found at '{fasta_path}'")

	cfg = config or Config()
	model = EccFormer(cfg)
	try:
		state = torch.load(model_path, map_location=cfg.device)
		model.load_state_dict(state)
		model.to(cfg.device)
		model.eval()
	except Exception as e:
		raise RuntimeError(f"Error loading the model state from '{model_path}'. The file might be corrupted or incompatible. Original error: {e}")

	# --- 2. Data Preparation (streaming, batch) ---
	print(f"Processing sequences from '{fasta_path}'...")
	records = list(SeqIO.parse(fasta_path, "fasta"))
	if not records:
		print("Warning: No valid sequences were found in the input FASTA file. Output will be empty.")
		SeqIO.write([], output_fasta, "fasta")
		return

	# Preprocess all sequences (with progress bar)
	seq_ids = []
	seq_strs = []
	tokens = []
	masks = []
	for record in tqdm(records, desc="Encoding sequences"):
		sequence = str(record.seq).upper()
		if not sequence:
			print(f"Warning: Skipping empty sequence with ID: {record.id}")
			continue
		try:
			ids, mask = sequence_process(sequence, cfg)
			seq_ids.append(str(record.id))
			seq_strs.append(sequence)
			tokens.append(ids)
			masks.append(mask)
		except Exception as e:
			print(f"Warning: Skipping sequence {record.id} due to error: {e}")

	if not tokens:
		print("Warning: No valid sequences after preprocessing. Output will be empty.")
		SeqIO.write([], output_fasta, "fasta")
		return

	# --- 3. Model Inference (batched) ---
	print("Running model predictions in batches...")
	all_preds = []
	all_probs = []
	positive_records: List[SeqRecord] = []
	num_samples = len(tokens)
	for start in tqdm(range(0, num_samples, batch_size), desc="Predicting"):
		end = min(start + batch_size, num_samples)
		batch_tokens = torch.tensor(np.array(tokens[start:end]), dtype=torch.long).to(cfg.device)
		batch_masks = torch.as_tensor(np.array(masks[start:end]), dtype=torch.bool).to(cfg.device)
		with torch.no_grad():
			outputs = model(batch_tokens, batch_masks)
			probs = torch.softmax(outputs, dim=1)
			preds = torch.argmax(probs, dim=1)
		all_preds.extend(preds.cpu().tolist())
		all_probs.extend(probs.cpu().tolist())
		# Collect positive records for FASTA output
		for i, pred in enumerate(preds):
			if int(pred.item()) == 1:
				record_id = seq_ids[start + i]
				seq_str = seq_strs[start + i]
				positive_records.append(SeqRecord(Seq(seq_str), id=record_id, description=""))

	# --- 4. Write Results ---
	try:
		SeqIO.write(positive_records, output_fasta, "fasta")
		print(f"Found {len(positive_records)} positive sequences. Results saved to '{output_fasta}'.")
	except Exception as e:
		raise IOError(f"An error occurred while writing the output FASTA file to '{output_fasta}'. Check permissions and disk space. Original error: {e}")

	# Optionally write all predictions and probabilities to TSV
	if output_prob_tsv:
		import csv
		try:
			with open(output_prob_tsv, 'w', newline='') as f:
				writer = csv.writer(f, delimiter='\t')
				writer.writerow(["id", "prediction", "prob_0", "prob_1"])
				for i, (pid, pred, prob) in enumerate(zip(seq_ids, all_preds, all_probs)):
					writer.writerow([pid, pred, prob[0], prob[1]])
			print(f"Prediction probabilities saved to '{output_prob_tsv}'.")
		except Exception as e:
			print(f"Warning: Could not write prediction probabilities to TSV: {e}")


