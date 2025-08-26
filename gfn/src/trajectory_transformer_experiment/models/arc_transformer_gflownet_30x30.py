#!/usr/bin/env python3
"""
GFlowNet 방식 30x30 ARC Trajectory Transformer Model
GFlowNet의 실제 30x30 패딩 방식을 사용하는 Trajectory Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
import math
from typing import Dict, Optional, Tuple

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, attn_pdrop: float = 0.1, resid_pdrop: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()
        
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        
        return out

class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, mlp_ratio: int = 4, 
                 attn_pdrop: float = 0.1, resid_pdrop: float = 0.1):
        super().__init__()
        
        self.ln1 = LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.ln2 = LayerNorm(n_embd)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_ratio * n_embd),
            nn.GELU(),
            nn.Linear(mlp_ratio * n_embd, n_embd),
            nn.Dropout(resid_pdrop)
        )
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class SpatialPositionEmbedding(nn.Module):
    """30x30 그리드의 2D 위치 임베딩"""
    
    def __init__(self, n_embd: int):
        super().__init__()
        self.n_embd = n_embd
        
        # 30x30 그리드의 각 위치에 대한 임베딩
        self.row_embedding = nn.Embedding(30, n_embd // 2)
        self.col_embedding = nn.Embedding(30, n_embd // 2)
        
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (batch_size, seq_len) - 각 토큰의 그리드 내 위치 인덱스 (0-899)
        """
        batch_size, seq_len = positions.size()
        
        # 1D 인덱스를 2D 좌표로 변환
        row_indices = positions // 30  # 행 인덱스
        col_indices = positions % 30   # 열 인덱스
        
        # 클램핑하여 유효 범위 보장
        row_indices = torch.clamp(row_indices, 0, 29)
        col_indices = torch.clamp(col_indices, 0, 29)
        
        # 행과 열 임베딩 결합
        row_emb = self.row_embedding(row_indices)  # (batch_size, seq_len, n_embd//2)
        col_emb = self.col_embedding(col_indices)  # (batch_size, seq_len, n_embd//2)
        
        spatial_emb = torch.cat([row_emb, col_emb], dim=-1)  # (batch_size, seq_len, n_embd)
        return spatial_emb

class ARCGFlowNet30x30Transformer(nn.Module):
    """GFlowNet 방식 30x30 ARC Trajectory Transformer"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # 설정 값들
        self.vocab_size = config.get('vocab_size', 26)
        self.n_embd = config.get('n_embd', 128)
        self.n_layer = config.get('n_layer', 3)
        self.n_head = config.get('n_head', 8)
        self.obs_dim = config.get('observation_dim', 900)  # 30x30 = 900
        self.action_dim = config.get('action_dim', 1)
        self.reward_dim = config.get('reward_dim', 1) 
        self.value_dim = config.get('value_dim', 1)
        self.max_seq_length = config.get('max_sequence_length', 920)
        
        # 시퀀스 구조: [obs(900), action(1), reward(1), value(1)] = 903 토큰/스텝
        self.step_size = self.obs_dim + self.action_dim + self.reward_dim + self.value_dim
        
        # 임베딩 레이어들
        self.token_embedding = nn.Embedding(self.vocab_size, self.n_embd)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.n_embd)
        
        # 컴포넌트 타입 임베딩 (observation, action, reward, value)
        self.component_embedding = nn.Embedding(4, self.n_embd)
        
        # 30x30 그리드용 공간 위치 임베딩
        self.spatial_embedding = SpatialPositionEmbedding(self.n_embd)
        
        # 드롭아웃
        self.embd_dropout = nn.Dropout(config.get('embd_pdrop', 0.1))
        
        # Transformer 블록들
        self.blocks = nn.ModuleList([
            TransformerBlock(
                self.n_embd, 
                self.n_head,
                attn_pdrop=config.get('attn_pdrop', 0.1),
                resid_pdrop=config.get('resid_pdrop', 0.1)
            ) for _ in range(self.n_layer)
        ])
        
        # 최종 레이어 정규화
        self.ln_f = LayerNorm(self.n_embd)
        
        # 컴포넌트별 출력 헤드
        self.obs_head = nn.Linear(self.n_embd, 11, bias=False)  # 0-9 색상 + 패딩(10)
        self.action_head = nn.Linear(self.n_embd, 5, bias=False)  # 0-4 액션
        self.reward_head = nn.Linear(self.n_embd, 5, bias=False)  # 0-4 리워드
        self.value_head = nn.Linear(self.n_embd, 5, bias=False)   # 0-4 값
        
        # 가중치 초기화
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def get_component_type_and_position(self, sequence_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """시퀀스 인덱스로부터 컴포넌트 타입과 공간 위치 계산"""
        batch_size, seq_len = sequence_indices.size()
        device = sequence_indices.device
        
        component_types = torch.zeros_like(sequence_indices)
        spatial_positions = torch.zeros_like(sequence_indices)
        
        for i in range(seq_len):
            step_idx = i // self.step_size
            within_step_idx = i % self.step_size
            
            if within_step_idx < self.obs_dim:  # observation (0-899)
                component_types[:, i] = 0  # observation type
                spatial_positions[:, i] = within_step_idx  # 그리드 내 위치
            elif within_step_idx < self.obs_dim + self.action_dim:  # action
                component_types[:, i] = 1  # action type
                spatial_positions[:, i] = 0  # 액션은 공간 위치 없음
            elif within_step_idx < self.obs_dim + self.action_dim + self.reward_dim:  # reward
                component_types[:, i] = 2  # reward type
                spatial_positions[:, i] = 0
            else:  # value
                component_types[:, i] = 3  # value type
                spatial_positions[:, i] = 0
        
        return component_types, spatial_positions
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # 토큰 임베딩
        token_emb = self.token_embedding(input_ids)
        
        # 위치 임베딩
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_emb = self.position_embedding(position_ids)
        
        # 컴포넌트 타입과 공간 위치 계산
        component_types, spatial_positions = self.get_component_type_and_position(position_ids)
        
        # 컴포넌트 타입 임베딩
        component_emb = self.component_embedding(component_types)
        
        # 공간 위치 임베딩 (observation 토큰에만 적용)
        spatial_emb = self.spatial_embedding(spatial_positions)
        obs_mask = (component_types == 0).float().unsqueeze(-1)  # observation 마스크
        spatial_emb = spatial_emb * obs_mask  # observation이 아니면 0
        
        # 모든 임베딩 결합
        x = token_emb + position_emb + component_emb + spatial_emb
        x = self.embd_dropout(x)
        
        # Transformer 블록들 통과
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # 최종 정규화
        x = self.ln_f(x)
        
        # 컴포넌트별 로짓 계산
        obs_logits = self.obs_head(x)
        action_logits = self.action_head(x)
        reward_logits = self.reward_head(x)
        value_logits = self.value_head(x)
        
        # 컴포넌트 타입에 따라 적절한 로짓 선택
        logits = torch.zeros(batch_size, seq_len, self.vocab_size, device=device)
        
        # Observation positions (0-10: colors + padding)
        obs_mask = (component_types == 0)
        if obs_mask.any():
            logits[obs_mask, :11] = obs_logits[obs_mask]
        
        # Action positions (11-15: actions)
        action_mask = (component_types == 1)
        if action_mask.any():
            logits[action_mask, 11:16] = action_logits[action_mask]
        
        # Reward positions (16-20: rewards)
        reward_mask = (component_types == 2)
        if reward_mask.any():
            logits[reward_mask, 16:21] = reward_logits[reward_mask]
        
        # Value positions (21-25: values)
        value_mask = (component_types == 3)
        if value_mask.any():
            logits[value_mask, 21:26] = value_logits[value_mask]
        
        outputs = {'logits': logits}
        
        # 손실 계산
        if labels is not None:
            # 컴포넌트별 가중치 적용
            obs_weight = 1.0
            action_weight = 2.0
            reward_weight = 1.0
            value_weight = 1.5
            
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_component_types = component_types[..., 1:].contiguous()
            
            # 각 토큰별 손실 계산
            token_losses = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
            token_losses = token_losses.view(batch_size, seq_len - 1)
            
            # 컴포넌트별 가중치 적용
            weights = torch.ones_like(shift_component_types, dtype=torch.float)
            weights[shift_component_types == 0] = obs_weight   # observation
            weights[shift_component_types == 1] = action_weight # action
            weights[shift_component_types == 2] = reward_weight # reward
            weights[shift_component_types == 3] = value_weight  # value
            
            weighted_losses = token_losses * weights
            
            # 어텐션 마스크 적용
            if attention_mask is not None:
                mask = attention_mask[..., 1:].contiguous()
                weighted_losses = weighted_losses * mask
                loss = weighted_losses.sum() / mask.sum()
            else:
                loss = weighted_losses.mean()
            
            outputs['loss'] = loss
        
        return outputs
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50,
                temperature: float = 1.0, top_k: Optional[int] = None, 
                top_p: Optional[float] = None, pad_token_id: int = 10,
                eos_token_id: int = 15) -> torch.Tensor:
        """자동회귀 생성"""
        
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        generated_ids = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # 현재까지의 시퀀스로 다음 토큰 예측
            with torch.no_grad():
                outputs = self.forward(generated_ids)
                logits = outputs['logits']
                
                # 마지막 토큰의 로짓
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k 필터링
                if top_k is not None:
                    values, indices = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < values[:, -1:]] = -float('inf')
                
                # Top-p 필터링
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    for i in range(batch_size):
                        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                        next_token_logits[i][indices_to_remove] = -float('inf')
                
                # 샘플링
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 시퀀스에 추가
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # EOS 토큰으로 종료
                if next_token.item() == eos_token_id:
                    break
        
        return generated_ids

def create_gflownet_30x30_model(config: Dict) -> ARCGFlowNet30x30Transformer:
    """GFlowNet 방식 30x30 모델 생성"""
    return ARCGFlowNet30x30Transformer(config)