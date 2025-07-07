"""
ARC Trajectory Transformer Model
GPT-style transformer for sequence modeling of ARC trajectories
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    """Multi-head causal attention"""
    
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']
        self.head_dim = self.n_embd // self.n_head
        
        # Key, query, value projections
        self.key = nn.Linear(self.n_embd, self.n_embd)
        self.query = nn.Linear(self.n_embd, self.n_embd)
        self.value = nn.Linear(self.n_embd, self.n_embd)
        
        # Output projection
        self.proj = nn.Linear(self.n_embd, self.n_embd)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.get('attn_pdrop', 0.1))
        self.resid_dropout = nn.Dropout(config.get('resid_pdrop', 0.1))
        
        # Causal mask
        max_seq_len = config.get('max_sequence_length', 64)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )
    
    def forward(self, x, attention_mask=None):
        B, T, C = x.size()  # Batch, Time, Channels
        
        # Calculate Q, K, V
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)
        
        # Apply causal mask
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: (B, T) -> (B, 1, 1, T)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        
        # Output projection
        out = self.proj(out)
        out = self.resid_dropout(out)
        
        return out

class MLP(nn.Module):
    """Feedforward network"""
    
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config['n_embd'], 4 * config['n_embd'])
        self.fc2 = nn.Linear(4 * config['n_embd'], config['n_embd'])
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.get('resid_pdrop', 0.1))
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP"""
    
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config['n_embd'])
        self.attention = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config['n_embd'])
        self.mlp = MLP(config)
    
    def forward(self, x, attention_mask=None):
        # Pre-norm architecture
        x = x + self.attention(self.ln1(x), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class ARCTrajectoryTransformer(nn.Module):
    """
    Trajectory Transformer for ARC tasks
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config['vocab_size']
        self.n_embd = config['n_embd']
        self.max_seq_len = config.get('max_sequence_length', 64)
        
        # Token embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.n_embd)
        self.position_embedding = nn.Embedding(self.max_seq_len, self.n_embd)
        
        # Dropout
        self.embd_dropout = nn.Dropout(config.get('embd_pdrop', 0.1))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config['n_layer'])
        ])
        
        # Final layer norm and output head
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        B, T = input_ids.size()
        
        # Position indices
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)  # (1, T)
        
        # Embeddings
        token_embd = self.token_embedding(input_ids)  # (B, T, n_embd)
        pos_embd = self.position_embedding(pos)       # (1, T, n_embd)
        
        # Combine embeddings
        x = token_embd + pos_embd
        x = self.embd_dropout(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        loss = None
        if labels is not None:
            # Compute cross-entropy loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {
            'logits': logits,
            'loss': loss
        }
    
    def generate(self, input_ids, max_new_tokens=32, temperature=1.0, top_k=None, top_p=None, 
                 pad_token_id=10, eos_token_id=21):
        """
        Generate sequences using the model
        """
        self.eval()
        B, T = input_ids.size()
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Get current sequence length
                current_length = input_ids.size(1)
                
                # Stop if we exceed max sequence length
                if current_length >= self.max_seq_len:
                    print(f"Stopping: exceeded max_seq_len ({self.max_seq_len})")
                    break
                
                # Forward pass
                outputs = self.forward(input_ids)
                logits = outputs['logits']
                
                # Get logits for the last token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Set logits to -inf for removed tokens
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Clamp token to valid vocabulary range
                next_token = torch.clamp(next_token, 0, self.vocab_size - 1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if EOS token is generated
                if next_token.item() == eos_token_id:
                    print(f"Stopping: EOS token generated at step {step}")
                    break
                    
                print(f"Step {step}: generated token {next_token.item()}")
        
        return input_ids

def create_model(config):
    """Create and return the model"""
    return ARCTrajectoryTransformer(config)