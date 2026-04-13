
"""
Unified CLI entrypoint for eccDNA model: training, prediction, pipeline, and testing.

Flags are unified across tasks. See --help for details and usage examples.
"""


import os
import sys
import argparse
import logging
from typing import Optional
from config import Config
from trainer import train_model, test_model
from predictor import predict_fasta
from genomic_pipeline import (
	sort_bam, bam_to_bed, call_peaks, process_split_reads, process_discordant_reads,
	intersect_and_count, merge_counts, extract_sequence
)



def run_full_pipeline(
	input_bam: str,
	reference: str,
	model_path: str,
	output_dir: str,
	min_read: int = 0,
	logger: Optional[logging.Logger] = None
) -> None:
	"""
	Execute genomic preprocessing and model prediction end-to-end.
	Args:
		input_bam (str): Input BAM file path.
		reference (str): Reference genome FASTA file.
		model_path (str): Trained model checkpoint path.
		output_dir (str): Output directory for results.
		min_read (int): Minimum supporting reads threshold.
		logger (logging.Logger, optional): Logger for progress and errors.
	"""
	os.makedirs(output_dir, exist_ok=True)
	tmp_dir = os.path.join(output_dir, "tmp")
	os.makedirs(tmp_dir, exist_ok=True)
	sorted_bam = os.path.join(tmp_dir, "sorted.bam")
	aln_bed = os.path.join(tmp_dir, "aln.bed")
	peak_site = os.path.join(tmp_dir, "peaks.site")
	peak_bed = os.path.join(tmp_dir, "peaks.bed")
	split_bed = os.path.join(tmp_dir, "split.bed")
	disc_bed = os.path.join(tmp_dir, "disc.bed")
	split_count_bed = os.path.join(tmp_dir, "split_count.bed")
	disc_count_bed = os.path.join(tmp_dir, "disc_count.bed")
	candidates_bed = os.path.join(tmp_dir, "candidates.bed")
	candidates_fa = os.path.join(tmp_dir, "candidates.fa")
	steps = [
		("Sorting BAM", lambda: sort_bam(input_bam, sorted_bam)),
		("Converting BAM to BED", lambda: bam_to_bed(sorted_bam, aln_bed)),
		("Calling peaks", lambda: call_peaks(sorted_bam, peak_site, peak_bed)),
		("Processing split reads", lambda: process_split_reads(aln_bed, split_bed)),
		("Processing discordant reads", lambda: process_discordant_reads(aln_bed, disc_bed)),
		("Intersecting peaks with split reads", lambda: intersect_and_count(peak_bed, split_bed, split_count_bed)),
		("Intersecting peaks with discordant reads", lambda: intersect_and_count(peak_bed, disc_bed, disc_count_bed)),
		("Merging counts", lambda: merge_counts(split_count_bed, disc_count_bed, candidates_bed, min_read)),
		("Extracting sequences", lambda: extract_sequence(reference, candidates_bed, candidates_fa)),
	]
	for step_name, step_fn in steps:
		if logger:
			logger.info(f"[Pipeline] {step_name}...")
		else:
			print(f"[Pipeline] {step_name}...")
		try:
			step_fn()
		except Exception as e:
			msg = f"Pipeline step failed: {step_name}\nError: {e}"
			if logger:
				logger.error(msg)
			else:
				print(msg)
			sys.exit(1)
	final_result = os.path.join(output_dir, "cicle.res.bed")
	circle_res_fa = os.path.join(output_dir, "circle.res.fa")
	import shutil
	shutil.copy2(candidates_bed, final_result)
	if logger:
		logger.info("[Pipeline] Predicting positives with trained model...")
	else:
		print("[Pipeline] Predicting positives with trained model...")
	predict_fasta(model_path, candidates_fa, circle_res_fa)
	if logger:
		logger.info(f"Done: final result saved to {final_result}\nAll files saved in {output_dir}")
	else:
		print(f"Done: final result saved to {final_result}")
		print(f"All files saved in {output_dir}")



def main() -> None:
    """
    Parse CLI arguments and dispatch to train, predict, pipeline, or test tasks.
    Provides robust error handling, logging, and user feedback.
    """
    parser = argparse.ArgumentParser(
        description="eccDNA unified CLI: train, predict, pipeline, test, kmer_attention",
        epilog="Example: python eccdna.py -t kmer_attention -m model.pth -i input.fa -o kmer_attention.csv"
    )
    parser.add_argument('-t', '--task', required=True, choices=['train', 'predict', 'pipe', 'test', 'kmer_attention'], help='Task type: train|predict|pipe|test|kmer_attention')
    # Argument groups for clarity
    train_group = parser.add_argument_group('Training Arguments')
    train_group.add_argument('-etr', '--encoded_train_dir', help='Directory of encoded training data batches')
    train_group.add_argument('-eva', '--encoded_val_dir', help='(Optional) Directory of encoded validation data batches')
    train_group.add_argument('-e', '--epochs', type=int, help='Number of epochs')
    train_group.add_argument('-b', '--batch_size', type=int, help='Batch size for training and testing')
    train_group.add_argument('-l', '--lr', type=float, help='Learning rate')
    train_group.add_argument('-sp', '--save_path', help='Output model path for train (.pth)')
    train_group.add_argument('-g', '--log_path', help='Output log CSV for train')

    predict_group = parser.add_argument_group('Prediction Arguments')
    predict_group.add_argument('-i', '--input', help='Input file for prediction (FASTA) or pipeline (BAM)')
    predict_group.add_argument('-o', '--output', help='Output file for prediction (FASTA) or output folder for pipeline')
    predict_group.add_argument('-m', '--model_path', help='Input model (.pth) for prediction, testing, or pipeline')

    pipe_group = parser.add_argument_group('Pipeline Arguments')
    pipe_group.add_argument('-r', '--reference', help='Reference genome for pipeline')
    pipe_group.add_argument('-n', '--min_read', type=int, default=3, help='Min supporting reads (pipeline), default 3')

    test_group = parser.add_argument_group('Testing Arguments')
    test_group.add_argument('-ete', '--encoded_test_dir', help='Directory of encoded test data batches')
    test_group.add_argument('-csv', '--test_csv', help='CSV file for testing (alternative to encoded_test_dir)')

    args = parser.parse_args()
    cfg = Config()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    logger = logging.getLogger("eccdna")

    # Common arguments that can be overridden from CLI
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.model_path:
        cfg.save_path = args.model_path # Use -m for loading model in test/predict tasks


    try:
        if args.task == 'train':
            if not args.encoded_train_dir:
                parser.error('Train task requires --encoded_train_dir.')
            cfg.encoded_train_dir = args.encoded_train_dir
            cfg.encoded_val_dir = args.encoded_val_dir
            if args.epochs: cfg.epochs = args.epochs
            if args.lr: cfg.lr = args.lr
            if args.save_path: cfg.save_path = args.save_path
            if args.log_path: cfg.log_path = args.log_path
            logger.info("Starting training...")
            train_model(cfg)

        elif args.task == 'predict':
            if not (args.model_path and args.input and args.output):
                parser.error('Predict task requires -m (model_path), -i (input), and -o (output).')
            logger.info("Starting prediction...")
            predict_fasta(args.model_path, args.input, args.output)

        elif args.task == 'pipe':
            if not (args.input and args.reference and args.model_path and args.output):
                parser.error('Pipeline task requires -i, -r, -m, and -o.')
            logger.info("Starting end-to-end pipeline...")
            run_full_pipeline(args.input, args.reference, args.model_path, args.output, args.min_read, logger=logger)

        elif args.task == 'test':
            if not (args.model_path and (args.encoded_test_dir or args.test_csv)):
                parser.error('Test task requires -m (model_path) and either --encoded_test_dir or --test_csv.')
            cfg.encoded_test_dir = args.encoded_test_dir
            cfg.test_csv = args.test_csv
            logger.info("Starting model evaluation on test set...")
            test_model(cfg)

        elif args.task == 'kmer_attention':
            if not (args.model_path and args.input):
                parser.error('kmer_attention task requires -m (model_path) and -i (input).')
            logger.info("Starting k-mer attention analysis...")
            from kmer_attention import kmer_attention_main
            kmer_attention_main(args.model_path, args.input, args.output)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)



if __name__ == '__main__':
	main()
