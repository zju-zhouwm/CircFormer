"""
k-mer attention analysis script: Input CSV or FASTA, output k-mer attention ranking CSV
"""
import argparse
import pandas as pd
import torch
import numpy as np
from collections import defaultdict
from model import EccFormer
from preprocess import sequence_process, Config
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os

class KmerDataset(Dataset):
    def __init__(self, file_path, config, filetype):
        self.samples = []
        self.kmer_seqs = []
        self.kmer_coordinates = []
        self.kmer_position_ratios = []
        self.sequence_types = []  # Store type for each sequence
        if filetype == 'csv':
            df = pd.read_csv(file_path)
            # Check if 'sequences' column exists
            if 'sequences' not in df.columns:
                raise ValueError('CSV file must contain "sequences" column')
            
            # Check for 'type' column (optional)
            has_type = 'type' in df.columns
            type_list = df['type'].tolist() if has_type else ['' for _ in range(len(df))]
            
            for idx, (seq, seq_type) in enumerate(zip(df['sequences'], type_list)):
                seq = str(seq).upper()
                tokens, mask = sequence_process(seq, config)
                k = config.k
                step = config.step
                L = len(seq)  # Sequence length
                kmers = [seq[i:i+k] for i in range(0, max(0, len(seq)-k+1), step)]
                # Generate bioinformatics format coordinates (starting from 1)
                coordinates = [f"{i+1}-{i+k}" for i in range(0, max(0, len(seq)-k+1), step)]
                # Calculate relative position: a/L, where a is kmer start position (starting from 0)
                relative_positions = [i / L for i in range(0, max(0, len(seq)-k+1), step)]
                self.samples.append(torch.tensor(tokens, dtype=torch.long))
                self.kmer_seqs.append(kmers)
                self.kmer_coordinates.append(coordinates)
                self.kmer_position_ratios.append(relative_positions)
                self.sequence_types.append(seq_type)
        elif filetype == 'fa':
            from Bio import SeqIO
            self.sequence_types = []  # Initialize for FASTA files
            for record in SeqIO.parse(file_path, 'fasta'):
                seq = str(record.seq).upper()
                tokens, mask = sequence_process(seq, config)
                k = config.k
                step = config.step
                L = len(seq)  # Sequence length
                kmers = [seq[i:i+k] for i in range(0, max(0, len(seq)-k+1), step)]
                # Generate bioinformatics format coordinates (starting from 1)
                coordinates = [f"{i+1}-{i+k}" for i in range(0, max(0, len(seq)-k+1), step)]
                # Calculate relative position: a/L, where a is kmer start position (starting from 0)
                relative_positions = [i / L for i in range(0, max(0, len(seq)-k+1), step)]
                self.samples.append(torch.tensor(tokens, dtype=torch.long))
                self.kmer_seqs.append(kmers)
                self.kmer_coordinates.append(coordinates)
                self.kmer_position_ratios.append(relative_positions)
                self.sequence_types.append('')  # FASTA files don't have type information
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        if hasattr(self, 'sequence_types') and self.sequence_types:
            return self.samples[idx], self.kmer_seqs[idx], self.kmer_coordinates[idx], self.kmer_position_ratios[idx], self.sequence_types[idx]
        else:
            return self.samples[idx], self.kmer_seqs[idx], self.kmer_coordinates[idx], self.kmer_position_ratios[idx], ''

def kmer_attention_main(model_path, input_path, output_csv, batch_size=32, device=None):
    if input_path.endswith('.csv'):
        filetype = 'csv'
    elif input_path.endswith('.fa') or input_path.endswith('.fasta'):
        filetype = 'fa'
    else:
        raise ValueError('Input file must be .csv or .fa/.fasta')
    config = Config()
    dataset = KmerDataset(input_path, config, filetype)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: (
        torch.nn.utils.rnn.pad_sequence([item[0] for item in x], batch_first=True),
        [item[1] for item in x],
        [item[2] for item in x],
        [item[3] for item in x],
        [item[4] for item in x]
    ))
    model = EccFormer(config)
    state = torch.load(model_path, map_location=device or config.device)
    model.load_state_dict(state)
    model.to(device or config.device)
    model.eval()
    # Store kmer attention scores for each sequence
    sequence_attention_data = []
    sequence_counter = 0
    
    with torch.no_grad():
        for batch_tokens, batch_kmers, batch_coordinates, batch_position_ratios, batch_types in tqdm(dataloader, desc="Extracting attention"):
            batch_tokens = batch_tokens.to(device or config.device)
            mask = (batch_tokens != config.PAD_ID)
            x = model.embedding(batch_tokens)
            x = model.posencoding(x, mask)
            # Use the last encoder layer
            last_layer = model.encoder_layers[-1]
            attn_mask = ~mask
            
            # Forward propagation to the last layer
            for layer in model.encoder_layers:
                x_conv = x.permute(0, 2, 1)
                x_conv = layer.conv1(x_conv)
                x_conv = layer.conv_activation(x_conv)
                x_conv = x_conv.permute(0, 2, 1)
                Q = layer.query(x_conv).view(x_conv.size(0), x_conv.size(1), layer.num_head, layer.attention_head_dim).transpose(1, 2)
                K = layer.key(x_conv).view(x_conv.size(0), x_conv.size(1), layer.num_head, layer.attention_head_dim).transpose(1, 2)
                scores = torch.einsum('bnqd,bnkd->bnqk', Q, K) / np.sqrt(layer.attention_head_dim)
                penalty_matrix = layer.get_penalty_matrix_from_mask(mask, gamma=layer.gamma, std_param=layer.std_param, device=x_conv.device)
                scores = scores - penalty_matrix[:, None, :, :]
                mask_expand = mask[:, None, None, :].expand_as(scores)
                scores = scores.masked_fill(~mask_expand, -1e10)
                attn_probs = torch.nn.functional.softmax(scores, dim=-1)
                
                # If it's the last layer, save attention weights
                if layer == last_layer:
                    # Only use attention weights from the last layer, not averaging across heads
                    attn_last_layer = attn_probs.mean(1).cpu().numpy()  # (batch, seq_len, seq_len)
                    break
                
                # Continue forward propagation
                V = layer.value(x_conv).view(x_conv.size(0), x_conv.size(1), layer.num_head, layer.attention_head_dim).transpose(1, 2)
                context = torch.einsum('bnqk,bnkd->bnqd', attn_probs, V)
                context = context.transpose(1, 2).contiguous().view(x_conv.size(0), x_conv.size(1), layer.attention_all_dim)
                out = layer.dropout(layer.fc(context))
                x = layer.LayerNorm(x_conv + out)
            
            # Process kmer attention scores for each sequence
            for i, (kmer_seq, coordinate_seq, ratio_seq, seq_type) in enumerate(zip(batch_kmers, batch_coordinates, batch_position_ratios, batch_types)):
                seq_len = len(kmer_seq)
                attn_diag = np.diag(attn_last_layer[i][:seq_len, :seq_len])
                
                # Collect all kmers, corresponding attention scores, coordinates, and position ratios for this sequence
                kmer_scores = []
                for kmer, score, coordinate, ratio in zip(kmer_seq, attn_diag, coordinate_seq, ratio_seq):
                    kmer_scores.append((kmer, score, coordinate, ratio))
                
                # Sort by attention score in descending order
                kmer_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Take top 10 kmers
                top_10_kmers = kmer_scores[:10]
                
                # Save results
                for rank, (kmer, score, coordinate, ratio) in enumerate(top_10_kmers, 1):
                    sequence_attention_data.append({
                        'sequence_id': sequence_counter,
                        'kmer': kmer,
                        'type': seq_type,
                        'coordinate': coordinate,
                        'attention_score': score,
                        'relative_position': ratio,
                        'rank': rank
                    })
                
                sequence_counter += 1
    
    # Output results to CSV
    df = pd.DataFrame(sequence_attention_data)
    df.to_csv(output_csv, index=False)
    print(f"Top-10 kmer attention per sequence saved to {output_csv}")
    print(f"Total sequences processed: {sequence_counter}")

def main():
    parser = argparse.ArgumentParser(description="Analyze k-mer attention weights from a trained model.")
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to trained model .pth file')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input csv or fasta file')
    parser.add_argument('-o', '--output', type=str, default='kmer_attention.csv', help='Output CSV file')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    kmer_attention_main(args.model_path, args.input, args.output, args.batch_size, args.device)

if __name__ == '__main__':
    main()
