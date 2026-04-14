# CircFormer: A circular attention-based framework for accurate eccDNA identification in plants

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

CircFormer is a deep learning framework for circular DNA (eccDNA) classification using k-mer tokenization with convolution-enhanced self-attention and symmetric positional encoding designed for circular DNA features.

## 📦 Installation

### Requirements
- Python 3.8+
- PyTorch 2.1.0+
- CUDA 11.8+ (optional, for GPU)

### Install Dependencies
```bash
git clone https://github.com/your-username/circformer.git
cd circformer
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### External Tools (for genomic pipeline)
```bash
# Ubuntu/Debian
sudo apt-get install samtools bedtools seqtk

# macOS
brew install samtools bedtools seqtk

# Genrich (download from https://github.com/jsh58/Genrich)
```

## 🏃 Quick Start

### 1. Data Preparation
```bash
# Split raw CSV into train/val/test (7:2:1 ratio)
python preprocess.py -i raw_data.csv -o dataset/ -s

# Encode the split datasets
python preprocess.py -ed dataset/ -o encoded_data/
```

### 2. Model Training
```bash
python eccdna.py -t train -etr encoded_data/train -eva encoded_data/val -sp model.pth -g train_log.csv -e 20 -b 32 -l 2e-5
```

### 3. Prediction
```bash
python eccdna.py -t predict -m model.pth -i candidates.fa -o positives.fa
```

### 4. Genomic Pipeline (from BAM to predictions)
```bash
python eccdna.py -t pipe -i sample.bam -r reference.fa -m model.pth -o results/ -n 3  # Minimum supporting reads
```

## 📖 Usage

All functions are accessed through the unified CLI interface `eccdna.py`:

```bash
python eccdna.py -t <task> [task-specific arguments]
```

### Available Tasks

| Task | Description | Required Arguments |
|------|-------------|-------------------|
| `train` | Train model | `-etr`, `-sp`, `-g` |
| `predict` | Predict FASTA file | `-m`, `-i`, `-o` |
| `pipe` | Genomic pipeline | `-i`, `-r`, `-m`, `-o` |
| `test` | Evaluate model | `-m`, `-ete` or `-csv` |
| `kmer_attention` | k-mer attention analysis | `-m`, `-i`, `-o` |

For detailed command examples, refer to the **Quick Start** section above.

## 📁 Project Structure

```
circformer/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.py                    # Configuration and vocabulary
├── model.py                     # CircFormer model architecture
├── preprocess.py                # Data preprocessing and datasets
├── trainer.py                   # Training and evaluation utilities
├── predictor.py                 # Prediction functions
├── genomic_pipeline.py          # Genomic processing pipeline
├── eccdna.py                    # Unified command-line interface
├── kmer_attention.py            # k-mer attention analysis
└── generate_roc.py              # ROC curve data generation
```
