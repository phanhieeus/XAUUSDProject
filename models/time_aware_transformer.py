import torch
import torch.nn as nn
import math

class Time2Vec(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Chia embedding_dim thành 2 phần bằng nhau cho linear và periodic
        self.linear_dim = embedding_dim // 2
        self.periodic_dim = embedding_dim // 2
        
        # Linear embedding
        self.linear = nn.Linear(1, self.linear_dim)
        
        # Periodic embedding
        self.periodic = nn.Linear(1, self.periodic_dim)
        
    def forward(self, x):
        """
        Args:
            x: tensor shape [batch_size, seq_len, 1] - normalized timestamps
        Returns:
            tensor shape [batch_size, seq_len, embedding_dim] - time embeddings
        """
        # Linear embedding
        linear_emb = self.linear(x)  # [batch_size, seq_len, linear_dim]
        
        # Periodic embedding
        periodic_emb = torch.sin(self.periodic(x))  # [batch_size, seq_len, periodic_dim]
        
        # Kết hợp linear và periodic embeddings
        time_emb = torch.cat([linear_emb, periodic_emb], dim=-1)  # [batch_size, seq_len, embedding_dim]
        return time_emb

class FeatureEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim)
        
    def forward(self, x):
        """
        Args:
            x: tensor shape [batch_size, seq_len, input_dim] - Close, Volume features
        Returns:
            tensor shape [batch_size, seq_len, embedding_dim] - feature embeddings
        """
        return self.embedding(x)

class TimeAwareAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        # Linear projections
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Time decay parameter
        self.time_decay = nn.Parameter(torch.ones(1))
        
    def forward(self, x, timestamps):
        """
        Args:
            x: tensor shape [batch_size, seq_len, embedding_dim] - input embeddings
            timestamps: tensor shape [batch_size, seq_len, 1] - normalized timestamps
        Returns:
            tensor shape [batch_size, seq_len, embedding_dim] - attended embeddings
        """
        batch_size, seq_len, _ = x.shape
        
        # Tính time differences và decay weights
        time_diffs = timestamps.unsqueeze(2) - timestamps.unsqueeze(1)  # [batch_size, seq_len, seq_len, 1]
        time_diffs = time_diffs.squeeze(-1)  # [batch_size, seq_len, seq_len]
        decay_weights = torch.exp(-self.time_decay * torch.abs(time_diffs))  # [batch_size, seq_len, seq_len]
        
        # Linear projections
        q = self.q_proj(x)  # [batch_size, seq_len, embedding_dim]
        k = self.k_proj(x)  # [batch_size, seq_len, embedding_dim]
        v = self.v_proj(x)  # [batch_size, seq_len, embedding_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch_size, num_heads, seq_len, seq_len]
        
        # Apply time decay to attention scores
        decay_weights = decay_weights.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
        scores = scores * decay_weights  # [batch_size, num_heads, seq_len, seq_len]
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Reshape and combine heads
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.embedding_dim)  # [batch_size, seq_len, embedding_dim]
        
        # Final projection
        attn_output = self.out_proj(attn_output)  # [batch_size, seq_len, embedding_dim]
        
        return attn_output

class TimeAwareTransformer(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_heads, num_classes, dropout=0.1):
        super().__init__()
        
        # Embedding layers
        self.time_embedding = Time2Vec(embedding_dim)
        self.feature_embedding = FeatureEmbedding(input_dim, embedding_dim)
        
        # Time-aware attention
        self.time_aware_attention = TimeAwareAttention(embedding_dim, num_heads)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes)
        )
        
    def forward(self, features, timestamps):
        """
        Args:
            features: tensor shape [batch_size, seq_len, input_dim] - Close, Volume features
            timestamps: tensor shape [batch_size, seq_len, 1] - normalized timestamps
        Returns:
            tensor shape [batch_size, num_classes] - classification logits
        """
        # Embed features và timestamps
        feature_emb = self.feature_embedding(features)  # [batch_size, seq_len, embedding_dim]
        time_emb = self.time_embedding(timestamps)  # [batch_size, seq_len, embedding_dim]
        
        # Kết hợp embeddings
        x = feature_emb + time_emb  # [batch_size, seq_len, embedding_dim]
        
        # Áp dụng time-aware attention
        x = self.time_aware_attention(x, timestamps)  # [batch_size, seq_len, embedding_dim]
        
        # Lấy embedding của điểm cuối chuỗi
        x = x[:, -1, :]  # [batch_size, embedding_dim]
        
        # Classification
        logits = self.classifier(x)  # [batch_size, num_classes]
        
        return logits 