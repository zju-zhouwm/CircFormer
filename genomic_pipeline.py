
"""
Genomic preprocessing pipeline for eccDNA candidate discovery.

This module provides robust, modular utilities for BAM sorting, BAM->BED conversion, peak calling,
split/discordant read processing, intersections, merging counts, and sequence extraction.
It includes type annotations, logging, error handling, and external tool checks for reliability.

Requirements:
	- samtools
	- bedtools
	- Genrich
	- seqtk
	- pybedtools
	- pandas
"""


import os
import sys
import subprocess
import shutil
import logging
from typing import Optional
import pandas as pd
import pybedtools



def check_external_tool(tool: str) -> None:
	"""
	Check if an external tool is available in PATH. Raise RuntimeError if not found.
	"""
	if shutil.which(tool) is None:
		raise RuntimeError(f"Required external tool '{tool}' not found in PATH. Please install it.")

def run_command(cmd: str, logger: Optional[logging.Logger] = None) -> None:
	"""
	Run a shell command and exit on failure with message.
	Args:
		cmd (str): Command to execute.
		logger (logging.Logger, optional): Logger for error messages.
	"""
	try:
		subprocess.check_call(cmd, shell=True)
	except subprocess.CalledProcessError as e:
		msg = f"Command failed: {cmd}\nError: {e}"
		if logger:
			logger.error(msg)
		else:
			print(msg)
		sys.exit(1)



def sort_bam(input_bam: str, output_bam: str) -> None:
	"""
	Name-sort BAM to facilitate downstream pairing operations.
	Args:
		input_bam (str): Input BAM file path.
		output_bam (str): Output sorted BAM file path.
	"""
	check_external_tool('samtools')
	run_command(f"samtools sort -n -o {output_bam} {input_bam}")



def bam_to_bed(input_bam: str, output_bed: str) -> None:
	"""
	Convert BAM to BED and normalize fields expected by later steps.
	Args:
		input_bam (str): Input BAM file path.
		output_bed (str): Output BED file path.
	"""
	check_external_tool('bedtools')
	run_command(f"bedtools bamtobed -i {input_bam} | sed 's|/|\t|g' | cut -f1-5,7 > {output_bed}")



def call_peaks(input_bam: str, output_site: str, output_bed: str) -> None:
	"""
	Call peaks using Genrich and export BED of peak intervals.
	Args:
		input_bam (str): Input BAM file path.
		output_site (str): Output Genrich site file.
		output_bed (str): Output BED file for peaks.
	"""
	check_external_tool('Genrich')
	run_command(f"Genrich -v -l 100 -g 200 -p 0.05 -t {input_bam} -o {output_site} && cut -f1-3 {output_site} > {output_bed}")



def process_split_reads(input_bed: str, output_bed: str) -> None:
	"""
	Aggregate split-read signals into candidate intervals with attributes.
	Args:
		input_bed (str): Input BED file path.
		output_bed (str): Output BED file path for split reads.
	"""
	df = pybedtools.BedTool(input_bed).to_dataframe()
	df['chrom'] = df[['chrom', 'name']].apply(lambda x: f'{x[0]}__{x[1]}', axis=1)
	x = pybedtools.BedTool.sort(pybedtools.BedTool.from_dataframe(df))
	d = pybedtools.BedTool.merge(x, c='5,6', o='collapse,collapse')
	dM = d.groupby(g=[1], c=[2,3,4,5], o=['min','max','collapse','collapse'])
	dM = pybedtools.BedTool.to_dataframe(dM, names=['chrom','start','end','pair','strand'])
	pd.options.mode.chained_assignment = None  # Suppress chained assignment warning
	splitM = dM[(dM.pair == "1,2,1") & (dM.strand == "+,-,+") |
				(dM.pair == "2,1,2") & (dM.strand == "-,+,-") |
				(dM.pair == "1,2,1") & (dM.strand == "-,+,-") |
				(dM.pair == "2,1,2") & (dM.strand == "+,-,+")]
	if splitM.empty:
		splitM = pd.DataFrame(columns=['chrom','start','end','pair','strand','read','length'])
	else:
		splitM[['chrom','read']] = splitM['chrom'].str.split('__', expand=True)
		splitM['length'] = splitM['end'] - splitM['start']
		splitM = splitM.sort_values(by=['chrom', 'start'])
	splitM.to_csv(output_bed, header=False, index=False, sep='\t')



def process_discordant_reads(input_bed: str, output_bed: str) -> None:
	"""
	Aggregate discordant pair signals into intervals with attributes.
	Args:
		input_bed (str): Input BED file path.
		output_bed (str): Output BED file path for discordant reads.
	"""
	df = pybedtools.BedTool(input_bed).to_dataframe()
	df['chrom'] = df[['chrom', 'name']].apply(lambda x: f'{x[0]}__{x[1]}', axis=1)
	x = pybedtools.BedTool.sort(pybedtools.BedTool.from_dataframe(df))
	y = pybedtools.BedTool.merge(x, c='5,6', o='collapse,collapse')
	d = pybedtools.BedTool.to_dataframe(y, names=['chrom','start','end','pair','strand'])
	dM = d[(d.pair != "1,2") & (d.pair != "2,1")]
	dM = pybedtools.BedTool.from_dataframe(dM).groupby(g=[1], c=[2,3,4,5], o=['min','max','collapse','collapse'])
	dM = pybedtools.BedTool.to_dataframe(dM, names=['chrom','start','end','pair','strand'])
	pd.options.mode.chained_assignment = None
	disc = dM[((dM.pair == "1,2") & (dM.strand == "-,+")) | ((dM.pair == "2,1") & (dM.strand == "-,+"))]
	if disc.empty:
		# Create an empty DataFrame with the expected columns to avoid errors
		disc = pd.DataFrame(columns=['chrom','start','end','pair','strand','read','length'])
	else:
		disc[['chrom','read']] = disc['chrom'].str.split('__', expand=True)
		disc['length'] = disc.end - disc.start
		disc = disc.sort_values(by=['chrom', 'start'])
	disc.to_csv(output_bed, header=False, index=False, sep='\t')



def intersect_and_count(peak_bed: str, read_bed: str, output_bed: str) -> None:
	"""
	Intersect peaks with reads and count unique supporting reads per peak.
	Args:
		peak_bed (str): BED file with peaks.
		read_bed (str): BED file with reads.
		output_bed (str): Output BED file for intersection counts.
	"""
	a = pybedtools.BedTool(peak_bed)
	b = pybedtools.BedTool(read_bed)
	see = a.intersect(b, wb=True, wa=True, nonamecheck=True)
	s = pybedtools.BedTool(see).to_dataframe()
	if not s.empty:
		s['cov'] = (s.strand - s.score) / (s.end - s.start) * 100
		s = s[(s['cov'] <= 100)]
		s = pybedtools.BedTool.from_dataframe(s).groupby(g=[1,2,3], c=[9], o=['count_distinct'])
		s.saveas(output_bed)
	else:
		with open(output_bed, 'w') as f:
			pass



def merge_counts(
	split_count_bed: str,
	disc_count_bed: str,
	output_bed: str,
	min_read: int = 0
) -> pd.DataFrame:
	"""
	Merge split/discordant counts, compute lengths, and filter by min_read.
	Args:
		split_count_bed (str): BED file with split read counts.
		disc_count_bed (str): BED file with discordant read counts.
		output_bed (str): Output BED file for merged counts.
		min_read (int): Minimum supporting reads threshold.
	Returns:
		pd.DataFrame: The merged DataFrame.
	"""
	def read_files(filelist):
		dfs = []
		for fn in filelist:
			df = pd.read_csv(fn, sep='\t', header=None, names=['Chr', 'start', 'end', 'read'])
			dfs.append(df)
		from functools import reduce
		return reduce(lambda left, right: pd.merge(left, right, on=['Chr', 'start', 'end'], how='inner'), dfs).fillna(0)
	files = [split_count_bed, disc_count_bed]
	df = read_files(files)
	df['length'] = df.end - df.start
	df = df[df['read_x'] - min_read >= 0]
	df.to_csv(output_bed, header=False, index=False, sep='\t')
	return df



def extract_sequence(reference: str, bed_file: str, output_fasta: str) -> None:
	"""
	Extract FASTA sequences for intervals using seqtk subseq.
	Args:
		reference (str): Reference genome FASTA file.
		bed_file (str): BED file with intervals.
		output_fasta (str): Output FASTA file path.
	"""
	check_external_tool('seqtk')
	run_command(f"seqtk subseq {reference} {bed_file} > {output_fasta}")


