"""
30x30 ARC Trajectory Transformer Model
GFlowNet의 30x30 패딩 방식을 사용하는 Trajectory Transformer
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
        
        # Causal mask for longer sequences
        max_seq_len = config.get('max_sequence_length', 1024)
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
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply causal mask
        causal_mask = self.causal_mask[:, :, :T, :T]
        att = att.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.view(B, 1, 1, T)
            att = att.masked_fill(attention_mask == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble
        
        # Output projection
        y = self.resid_dropout(self.proj(y))
        return y

class TransformerBlock(nn.Module):
    """Transformer block"""
    
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ln1 = nn.LayerNorm(config['n_embd'])
        self.ln2 = nn.LayerNorm(config['n_embd'])
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(config['n_embd'], 4 * config['n_embd']),
            nn.GELU(),
            nn.Linear(4 * config['n_embd'], config['n_embd']),
            nn.Dropout(config.get('resid_pdrop', 0.1))
        )
    
    def forward(self, x, attention_mask=None):
        # Attention with residual connection
        x = x + self.attention(self.ln1(x), attention_mask)
        # MLP with residual connection
        x = x + self.mlp(self.ln2(x))
        return x

class ARC30x30TrajectoryTransformer(nn.Module):
    """Trajectory Transformer for variable-sized ARC grids (padded to 30x30)"""
    
    def __init__(self, config):
        super().__init__()
        
        # Grid and sequence dimensions
        self.obs_dim = config.get('observation_dim', 900)  # 30x30 = 900
        self.action_dim = config.get('action_dim', 1)
        self.reward_dim = config.get('reward_dim', 1)
        self.value_dim = config.get('value_dim', 1)
        
        # Model configuration
        self.n_embd = config['n_embd']
        self.vocab_size = config['vocab_size']
        self.max_seq_len = config['max_sequence_length']
        
        # Sequence structure: [obs(900), action(1), reward(1), value(1)] per step
        self.step_size = self.obs_dim + self.action_dim + self.reward_dim + self.value_dim  # 903 tokens
        
        # Token embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.n_embd)
        
        # Position embeddings
        self.position_embedding = nn.Embedding(self.max_seq_len, self.n_embd)
        
        # Component type embeddings (obs, action, reward, value)
        self.component_embedding = nn.Embedding(4, self.n_embd)
        
        # 2D spatial embeddings for grid positions
        self.spatial_row_embedding = nn.Embedding(30, self.n_embd // 2)
        self.spatial_col_embedding = nn.Embedding(30, self.n_embd // 2)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config['n_layer'])])
        
        # Layer norm
        self.ln_f = nn.LayerNorm(self.n_embd)
        
        # Output heads for different components
        self.obs_head = nn.Linear(self.n_embd, 11, bias=False)    # 0-10 (colors + pad)
        self.action_head = nn.Linear(self.n_embd, 5, bias=False)  # 0-4 (actions)
        self.reward_head = nn.Linear(self.n_embd, 5, bias=False)  # 0-4 (reward levels)
        self.value_head = nn.Linear(self.n_embd, 5, bias=False)   # 0-4 (value levels)
        
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
    
    def get_component_type(self, position: int) -> int:
        """
        Get component type for a position in the sequence
        0: observation, 1: action, 2: reward, 3: value
        """
        pos_in_step = position % self.step_size
        if pos_in_step < self.obs_dim:
            return 0  # observation
        elif pos_in_step < self.obs_dim + self.action_dim:
            return 1  # action
        elif pos_in_step < self.obs_dim + self.action_dim + self.reward_dim:
            return 2  # reward
        else:
            return 3  # value
    
    def get_spatial_position(self, position: int) -> Tuple[int, int]:
        """
        Get 2D spatial position for observation tokens
        Returns (row, col) for 30x30 grid
        """
        pos_in_step = position % self.step_size
        if pos_in_step < self.obs_dim:  # Only for observation tokens
            grid_pos = pos_in_step
            row = grid_pos // 30
            col = grid_pos % 30
            return row, col
        else:
            return 0, 0  # Default for non-observation tokens
    
    def forward(self, input_ids, attention_mask=None, targets=None):
        B, T = input_ids.size()
        
        # Clamp input_ids to valid vocabulary range
        input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
        
        # Token embeddings
        token_emb = self.token_embedding(input_ids)  # (B, T, n_embd)
        
        # Position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)  # (1, T)
        pos_emb = self.position_embedding(pos)  # (1, T, n_embd)
        
        # Component type embeddings
        component_types = torch.tensor([self.get_component_type(i) for i in range(T)], 
                                     dtype=torch.long, device=input_ids.device).unsqueeze(0)  # (1, T)
        comp_emb = self.component_embedding(component_types)  # (1, T, n_embd)
        
        # Spatial embeddings for observation tokens
        spatial_emb = torch.zeros(B, T, self.n_embd, device=input_ids.device)
        for i in range(T):
            row, col = self.get_spatial_position(i)
            if self.get_component_type(i) == 0:  # observation token
                row_emb = self.spatial_row_embedding(torch.tensor(row, device=input_ids.device))
                col_emb = self.spatial_col_embedding(torch.tensor(col, device=input_ids.device))
                spatial_emb[:, i, :self.n_embd//2] = row_emb
                spatial_emb[:, i, self.n_embd//2:] = col_emb
        
        # Combine all embeddings
        x = token_emb + pos_emb + comp_emb + spatial_emb
        
        # Apply dropout
        x = F.dropout(x, p=0.1, training=self.training)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Apply component-specific output heads
        logits = torch.zeros(B, T, self.vocab_size, device=x.device)
        
        for i in range(T):
            comp_type = self.get_component_type(i)
            if comp_type == 0:  # observation
                obs_logits = self.obs_head(x[:, i])  # (B, 11)
                logits[:, i, :11] = obs_logits
            elif comp_type == 1:  # action
                action_logits = self.action_head(x[:, i])  # (B, 5)
                logits[:, i, 11:16] = action_logits
            elif comp_type == 2:  # reward
                reward_logits = self.reward_head(x[:, i])  # (B, 5)
                logits[:, i, 16:21] = reward_logits
            elif comp_type == 3:  # value
                value_logits = self.value_head(x[:, i])  # (B, 5)
                logits[:, i, 21:26] = value_logits
        
        outputs = {'logits': logits}
        
        # Calculate loss if targets provided
        if targets is not None:
            # Get loss weights from config
            obs_weight = getattr(self, 'observation_weight', 1.0)
            action_weight = getattr(self, 'action_weight', 2.0)
            reward_weight = getattr(self, 'reward_weight', 1.0)
            value_weight = getattr(self, 'value_weight', 1.5)
            
            loss = 0.0
            total_tokens = 0
            
            for i in range(T):
                comp_type = self.get_component_type(i)
                
                if comp_type == 0:  # observation
                    obs_loss = F.cross_entropy(logits[:, i, :11], 
                                             torch.clamp(targets[:, i], 0, 10))
                    loss += obs_weight * obs_loss
                elif comp_type == 1:  # action
                    action_targets = torch.clamp(targets[:, i] - 11, 0, 4)
                    action_loss = F.cross_entropy(logits[:, i, 11:16], action_targets)
                    loss += action_weight * action_loss
                elif comp_type == 2:  # reward
                    reward_targets = torch.clamp(targets[:, i] - 16, 0, 4)
                    reward_loss = F.cross_entropy(logits[:, i, 16:21], reward_targets)
                    loss += reward_weight * reward_loss
                elif comp_type == 3:  # value
                    value_targets = torch.clamp(targets[:, i] - 21, 0, 4)
                    value_loss = F.cross_entropy(logits[:, i, 21:26], value_targets)
                    loss += value_weight * value_loss
                
                total_tokens += 1
            
            loss = loss / total_tokens
            outputs['loss'] = loss
        
        return outputs
    
    def generate(self, input_ids, max_new_tokens=64, temperature=1.0, top_k=None, top_p=0.9, pad_token_id=10, eos_token_id=15):
        """Generate sequence continuation"""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get predictions for next token
                outputs = self.forward(input_ids)
                logits = outputs['logits']
                
                # Focus on the last token
                logits = logits[:, -1, :] / temperature
                
                # Apply top_k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # Apply top_p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if EOS token generated
                if next_token.item() == eos_token_id:
                    break
                
                # Prevent infinite generation
                if input_ids.size(1) >= self.max_seq_len:
                    break
        
        return input_ids

def create_30x30_model(config):
    """Create 30x30 ARC Trajectory Transformer model"""
    model = ARC30x30TrajectoryTransformer(config)
    
    # Set loss weights as model attributes
    model.observation_weight = config.get('observation_weight', 1.0)
    model.action_weight = config.get('action_weight', 2.0)
    model.reward_weight = config.get('reward_weight', 1.0)
    model.value_weight = config.get('value_weight', 1.5)
    
    return model