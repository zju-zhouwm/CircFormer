# EccFormer: Deep Learning-based Circular DNA (eccDNA) Classifier

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

EccFormer is a deep learning model for classifying circular DNA (eccDNA) sequences. It uses k-mer tokenization with convolution-enhanced self-attention and symmetric positional encoding designed for circular DNA features.

## 🚀 Key Features

- **k-mer tokenization**: Convert DNA sequences to processable tokens (default k=6)
- **Symmetric positional encoding**: Specialized for circular DNA
- **Convolution-enhanced attention**: Combines local feature extraction with global attention
- **Complete genomic pipeline**: End-to-end workflow from BAM files to predictions
- **Efficient training**: Memory-mapped data loading, multi-GPU support, batch preprocessing
- **Comprehensive evaluation**: ROC curve generation, k-mer attention analysis
- **Unified CLI**: All functions accessible through `eccdna.py`

## 📦 Installation

### Requirements
- Python 3.8+
- PyTorch 2.1.0+
- CUDA 11.8+ (optional, for GPU)

### Install Dependencies
```bash
git clone https://github.com/your-username/eccformer.git
cd eccformer
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
python eccdna.py -t train \
  -etr encoded_data/train \
  -eva encoded_data/val \
  -sp model.pth \
  -g train_log.csv \
  -e 20 -b 32 -l 2e-5
```

### 3. Prediction
```bash
python eccdna.py -t predict \
  -m model.pth \
  -i candidates.fa \
  -o positives.fa
```

### 4. Genomic Pipeline (from BAM to predictions)
```bash
python eccdna.py -t pipe \
  -i sample.bam \
  -r reference.fa \
  -m model.pth \
  -o results/ \
  -n 3  # Minimum supporting reads
```

## 📖 Usage

### Unified Command-line Interface
All functions are accessed through `eccdna.py`:

```bash
python eccdna.py -t <task> [task-specific arguments]
```

#### Available Tasks:
| Task | Description | Required Arguments |
|------|-------------|-------------------|
| `train` | Train model | `-etr`, `-sp`, `-g` |
| `predict` | Predict FASTA file | `-m`, `-i`, `-o` |
| `pipe` | Genomic pipeline | `-i`, `-r`, `-m`, `-o` |
| `test` | Evaluate model | `-m`, `-ete` or `-csv` |
| `kmer_attention` | k-mer attention analysis | `-m`, `-i`, `--output_csv` |

### Basic Training
```bash
python eccdna.py -t train \
  -etr encoded_train/ \
  -eva encoded_val/ \
  -sp model.pth \
  -g train_log.csv
```

### Model Evaluation
```bash
# Using encoded data
python eccdna.py -t test -m model.pth -ete encoded_test/

# Using raw CSV
python eccdna.py -t test -m model.pth -csv test_data.csv
```

### k-mer Attention Analysis
```bash
python eccdna.py -t kmer_attention \
  -m model.pth \
  -i sequences.csv \
  --output_csv kmer_attention.csv
```

### ROC Curve Generation
```bash
python generate_roc.py -m model.pth -i test_data.csv -o roc_plot_data.csv
```

## 📁 Project Structure

```
eccformer/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.py                    # Configuration and vocabulary
├── model.py                     # EccFormer model architecture
├── preprocess.py                # Data preprocessing and datasets
├── trainer.py                   # Training and evaluation utilities
├── predictor.py                 # Prediction functions
├── genomic_pipeline.py          # Genomic processing pipeline
├── eccdna.py                    # Unified command-line interface
├── kmer_attention.py            # k-mer attention analysis
└── generate_roc.py              # ROC curve data generation
```

## 🧬 Model Architecture

### Core Components
1. **k-mer tokenizer**: Segments DNA into overlapping k-mers (default k=6)
2. **Symmetric positional encoding**: Specifically designed for circular DNA
3. **Convolution-enhanced self-attention**: Combines 1D convolution with multi-head attention
4. **Gaussian penalty mechanism**: Enhances positional awareness
5. **Stacked encoder layers**: Multiple attention layers with residual connections

### Default Hyperparameters (config.py)
```python
embedding_dim = 256      # Embedding dimension
hidden_dim = 768         # Hidden dimension
num_head = 8             # Attention heads
dropout = 0.1            # Dropout rate
gamma = 0.1              # Gaussian penalty strength
conv_channels = 256      # Convolution channels
kernel_size = 7          # Convolution kernel size
num_layers = 4           # Encoder layers
k = 6                    # k-mer length
step = 1                 # k-mer step
max_length = 2048        # Maximum sequence length
```

## 📊 Data Formats

### Training Data (CSV)
```csv
sequences,labels
ATGCTAGCTAGCTAGCTAGCTAGC,1
GCTAGCTAGCTAGCTAGCTAGCTA,0
CTAGCTAGCTAGCTAGCTAGCTAG,1
...
```

- `sequences`: DNA sequence string (A,G,C,T only)
- `labels`: Binary labels (1=eccDNA, 0=non-eccDNA)

### FASTA Format
```fasta
>chr1:1000-2000
ATGCTAGCTAGCTAGCTAGCTAGC
>chr2:3000-4000
GCTAGCTAGCTAGCTAGCTAGCTA
```

### Encoded Data Format (generated by preprocess.py)
- `tokens_X.npy`: k-mer token sequences
- `labels_X.npy`: Labels
- `masks_X.npy`: Padding masks

## 📈 Performance Evaluation

### Evaluation Metrics
```bash
python eccdna.py -t test -m model.pth -ete encoded_test/
```
Output includes: Accuracy, Precision, Recall, F1-Score

### ROC Curve Generation
```bash
python generate_roc.py -m model.pth -i test_data.csv -o roc_plot_data.csv
```

## 🧪 Example Workflow

```bash
# 1. Create demo data
echo -e "sequences,labels\nATGCTAGCTAGCTAGCTAGCTAGC,1\nGCTAGCTAGCTAGCTAGCTAGCTA,0" > demo.csv

# 2. Preprocess
python preprocess.py -i demo.csv -o demo_dataset/ -s
python preprocess.py -ed demo_dataset/ -o demo_encoded/

# 3. Train
python eccdna.py -t train -etr demo_encoded/train -eva demo_encoded/val \
  -sp demo_model.pth -g demo_log.csv -e 5 -b 16

# 4. Evaluate
python eccdna.py -t test -m demo_model.pth -csv demo_dataset/test.csv

# 5. Predict
echo -e ">test1\nATGCTAGCTAGCTAGCTAGCTAGC\n>test2\nGCTAGCTAGCTAGCTAGCTAGCTA" > test.fa
python eccdna.py -t predict -m demo_model.pth -i test.fa -o predictions.fa
```

## 🔍 FAQ

**Q: How to handle long DNA sequences?**  
A: Sequences longer than 2048 k-mers are symmetrically truncated from both ends.

**Q: How much training data is needed?**  
A: At least 1000 positive and 1000 negative examples for reasonable performance.

**Q: How to adjust k-mer length?**  
A: Modify `k` parameter in `config.py` or via command line.

**Q: Pipeline steps fail?**  
A: Ensure external tools (samtools, bedtools, seqtk, Genrich) are installed correctly.

**Q: Run on CPU?**  
A: PyTorch automatically detects CUDA. Force CPU with `export CUDA_VISIBLE_DEVICES=""`.

## 📚 Citation

If you use EccFormer in your research, please cite:

```bibtex
@software{eccformer2024,
  title = {EccFormer: Deep Learning-based eccDNA Classifier},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-username/eccformer},
  note = {Deep learning framework for circular DNA detection}
}
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the project
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**EccFormer** - Making circular DNA detection smarter and more accurate!