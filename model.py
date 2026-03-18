import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class GaitResNet(nn.Module):
    def __init__(self, input_size=69, num_classes=4, dropout=0.5):
        """
        ResNet-1D architecture optimized for few-shot gait analysis.
        - Residual connections prevent vanishing gradients.
        - Feature Normalization (LayerNorm) for stable embeddings.
        - L2-Normalized output head (Cosine Similarity).
        """
        super(GaitResNet, self).__init__()
        
        # 1. Initial Projection
        self.input_proj = nn.Linear(input_size, 64)
        self.input_bn = nn.BatchNorm1d(64)
        
        # 2. Residual Blocks
        self.layer1 = ResBlock1D(64)
        self.downsample = nn.Conv1d(64, 128, kernel_size=1)
        self.layer2 = ResBlock1D(128)
        
        # 3. Global Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 4. Normalized Head
        # Concatenate Avg + Max pooling for richer representation
        self.fc = nn.Linear(128 * 2, 128)
        self.output = nn.Linear(128, num_classes)
        self.gender_output = nn.Linear(128, 2) # 0=Male, 1=Female
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq, feat)
        
        # BN expects (batch, feat, seq)
        x = x.transpose(1, 2)
        
        # Initial projection
        # Note: input_proj expects (batch, ..., feat)
        # So we transpose back for the linear layer then return to CNN form
        x = x.transpose(1, 2)
        x = F.relu(self.input_bn(self.input_proj(x).transpose(1, 2)))
        
        # Layer 1
        x = self.layer1(x)
        
        # Layer 2 with skip-connection downsampling
        res = self.downsample(x)
        x = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1) # dummy pool to change shape if needed, but kernel_size=1 conv handles channels
        x = self.layer2(res) # simple version
        
        # Global Pooling
        avg_x = self.pool(x)
        max_x = self.max_pool(x)
        x = torch.cat([avg_x, max_x], dim=1) # (batch, 256, 1)
        
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc(x)))
        
        # Dual-Head Result
        id_logits = self.output(x)
        gender_logits = self.gender_output(x)
        
        return id_logits, gender_logits

# Mapping back for train.py compatibility
GaitLSTM = GaitResNet

if __name__ == "__main__":
    m = GaitResNet(input_size=69, num_classes=4)
    d = torch.randn(8, 60, 69)
    id_o, g_o = m(d)
    print(f"ResNet OK: {d.shape} -> ID: {id_o.shape}, Gender: {g_o.shape}")
    print(f"Params: {sum(p.numel() for p in m.parameters() if p.requires_grad):,}")
