"""Model architecture for eccDNA classification (EccFormer)."""

import math
import torch
import torch.nn as nn
from config import Config



class LearnableSymmetricPositionalEncoding(nn.Module):
	"""
	Learnable symmetric positional encoding over sequence length.
	Only applies symmetric encoding to valid positions (non-PAD), PAD positions get encoding 0.
	Args:
		max_seq_len (int): Maximum sequence length supported.
		d_model (int): Embedding dimension.
	"""
	def __init__(self, max_seq_len: int, d_model: int) -> None:
		super().__init__()
		self.max_seq_len: int = max_seq_len
		self.max_s: int = (max_seq_len - 1) // 2
		self.position_embedding: nn.Parameter = nn.Parameter(torch.empty(self.max_s + 2, d_model))  # +2 to include index 0 and max_s+1
		nn.init.xavier_normal_(self.position_embedding)

	def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model)
			mask (torch.Tensor): Boolean mask of shape (batch, seq_len) indicating valid positions
		Returns:
			torch.Tensor: Output tensor with positional encoding added.
		"""
		batch_size, seq_len, d_model = x.shape
		if seq_len > self.max_seq_len:
			raise ValueError(f"Input sequence length ({seq_len}) exceeds configured max_seq_len ({self.max_seq_len})")
		
		# Initialize positional encoding as zeros (for PAD positions)
		pos_embed = torch.zeros_like(x)
		
		for batch_idx in range(batch_size):
			valid_len = mask[batch_idx].sum().item()  # Number of valid positions
			
			# Apply symmetric positional encoding to valid positions
			for i in range(valid_len):
				s = min(i, valid_len - 1 - i) + 1  # Symmetric index starting from 1
				pos_embed[batch_idx, i] = self.position_embedding[s]
			
			# PAD positions remain as zeros (already initialized)
		
		return x + pos_embed



class SelfAttention(nn.Module):
	"""
	Convolution-augmented multi-head self-attention with Gaussian penalty.
	Args:
		config (Config): Model configuration object.
	"""
	def __init__(self, config: Config) -> None:
		super().__init__()
		self.num_head: int = config.num_head
		self.attention_all_dim: int = config.hidden_dim
		self.attention_head_dim: int = int(config.hidden_dim / config.num_head)
		assert self.attention_head_dim * config.num_head == config.hidden_dim, "hidden_dim must be divisible by num_head."
		self.conv1 = nn.Conv1d(in_channels=config.embedding_dim, out_channels=config.conv_channels, kernel_size=config.kernel_size, padding=config.kernel_size // 2)
		self.conv_activation = nn.ReLU()
		self.query = nn.Linear(config.conv_channels, self.attention_all_dim)
		self.key = nn.Linear(config.conv_channels, self.attention_all_dim)
		self.value = nn.Linear(config.conv_channels, self.attention_all_dim)
		self.dropout = nn.Dropout(config.dropout)
		self.fc = nn.Linear(config.hidden_dim, config.conv_channels)
		self.LayerNorm = nn.LayerNorm(config.conv_channels)
		# Learnable parameters for Gaussian penalty
		self.gamma = nn.Parameter(torch.tensor(config.gamma))  # Penalty strength
		self.std_param = nn.Parameter(torch.tensor(0.2))       # Standard deviation in normalized space

	@staticmethod
	def get_penalty_matrix_from_mask(mask: torch.Tensor, gamma: torch.Tensor, std_param: torch.Tensor, device=None) -> torch.Tensor:
		"""
		Build a penalty matrix using normalized Gaussian distribution.
		Args:
			mask (torch.Tensor): Boolean mask of shape (batch, seq_len)
			gamma (torch.Tensor): Learnable penalty scaling factor
			std_param (torch.Tensor): Learnable standard deviation in normalized space
			device: torch device
		Returns:
			torch.Tensor: Penalty matrix of shape (batch, seq_len, seq_len)
		"""
		batch, seq_len = mask.shape
		penalty_matrix = torch.zeros((batch, seq_len, seq_len), device=device or mask.device)
		for i in range(batch):
			valid_len = mask[i].sum().item()
			if valid_len == 0:
				continue
			positions = torch.arange(1, valid_len + 1, device=mask.device)  # Start from 1
			
			# Normalize positions to [0, 1] range
			normalized_positions = positions / (valid_len + 1)
			mean = 0.5  # Fixed center in normalized space
			
			# Learnable standard deviation with constraints
			std = torch.clamp(std_param, min=0.01, max=0.5)
			
			# Gaussian distribution
			exponent = -((normalized_positions - mean) ** 2) / (2 * std ** 2)
			base_penalty = torch.exp(exponent)
			exponential_factor = torch.exp(gamma) - 1
			penalty_vec = base_penalty * exponential_factor
			
			# Build symmetric penalty matrix
			# The penalty for attending from position i to j depends on both positions
			penalty_i = penalty_vec.unsqueeze(1).expand(-1, valid_len)
			penalty_j = penalty_vec.unsqueeze(0).expand(valid_len, -1)
			penalty = (penalty_i + penalty_j) / 2 
			
			penalty_matrix[i, :valid_len, :valid_len] = penalty
		return penalty_matrix

	def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
		"""
		Args:
			x (torch.Tensor): Input tensor of shape (batch, seq_len, embedding_dim)
			mask (torch.Tensor): Boolean mask of shape (batch, seq_len)
		Returns:
			torch.Tensor: Output tensor after self-attention and residual connection.
		"""
		# x: (batch, seq_len, embedding_dim)
		x = x.permute(0, 2, 1)  # (batch, embedding_dim, seq_len)
		x = self.conv1(x)
		x = self.conv_activation(x)
		x = x.permute(0, 2, 1)  # (batch, seq_len, conv_channels)
		batch_size, seq_len, _ = x.size()
		Q = self.query(x).view(batch_size, seq_len, self.num_head, self.attention_head_dim).transpose(1, 2)
		K = self.key(x).view(batch_size, seq_len, self.num_head, self.attention_head_dim).transpose(1, 2)
		V = self.value(x).view(batch_size, seq_len, self.num_head, self.attention_head_dim).transpose(1, 2)
		scores = torch.einsum('bnqd,bnkd->bnqk', Q, K) / math.sqrt(self.attention_head_dim)
		if mask is None:
			raise ValueError(f"mask must be provided (got None). Input shape: {x.shape}")
		penalty_matrix = SelfAttention.get_penalty_matrix_from_mask(mask, gamma=self.gamma, std_param=self.std_param, device=x.device)
		scores = scores - penalty_matrix[:, None, :, :]
		mask_expand = mask[:, None, None, :].expand_as(scores)
		scores = scores.masked_fill(~mask_expand, -1e10)
		attention_probs = torch.nn.functional.softmax(scores, dim=-1)
		attention_probs = self.dropout(attention_probs)
		context = torch.einsum('bnqk,bnkd->bnqd', attention_probs, V)
		context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.attention_all_dim)
		out = self.dropout(self.fc(context))
		return self.LayerNorm(x + out)



class EccFormer(nn.Module):
	"""
	Encoder network producing binary logits for eccDNA classification.
	Args:
		config (Config): Model configuration object.
	"""
	def __init__(self, config: Config) -> None:
		super().__init__()
		self.embedding: nn.Embedding = nn.Embedding(
			num_embeddings=len(config.DNA_VOCAB),
			embedding_dim=config.embedding_dim,
			padding_idx=config.PAD_ID
		)
		self.posencoding: LearnableSymmetricPositionalEncoding = LearnableSymmetricPositionalEncoding(
			config.max_length, config.embedding_dim
		)
		self.encoder_layers: nn.ModuleList = nn.ModuleList([
			SelfAttention(config) for _ in range(config.num_layers)
		])
		self.flatten: nn.Linear = nn.Linear(config.embedding_dim, config.hidden_dim)
		self.fc1: nn.Linear = nn.Linear(config.hidden_dim, config.output_dim)

	def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
		"""
		Args:
			x (torch.Tensor): Input token ids of shape (batch, seq_len)
			mask (torch.Tensor): Boolean mask of shape (batch, seq_len)
		Returns:
			torch.Tensor: Output logits of shape (batch, output_dim)
		"""
		x = self.embedding(x)
		x = self.posencoding(x, mask)
		if mask is None:
			raise ValueError(f"mask must be provided (got None). Input shape: {x.shape}")
		mask_inverted = ~mask.bool()
		for layer in self.encoder_layers:
			x = layer(x, mask_inverted)
		x = x.mean(dim=1)  # Global average pooling over sequence length
		x = self.flatten(x)
		return self.fc1(x)
