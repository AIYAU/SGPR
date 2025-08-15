import torch.nn as nn

class FineNetwork(nn.Module):
    def __init__(self, feature_dim):
        """
        微调网络，用于调整 CLIP 提取的特征。
        Args:
            feature_dim: CLIP 提取的特征维度。
        """
        super(FineNetwork, self).__init__()
        self.fine_network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, x):
        """
        前向传播：提取微调后的特征。
        Args:
            x: 输入特征 (tensor) [batch_size, feature_dim]
        Returns:
            微调后的特征 (tensor) [batch_size, feature_dim]
        """
        return self.fine_network(x)
