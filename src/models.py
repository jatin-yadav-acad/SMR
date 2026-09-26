"""
models.py
=========
Production PyTorch neural architectures for Sparse Mechanistic Routing (SMR).
Includes ResNet-18, ResNet-50, and the Pre-Allocated Sparse Head (PASH)
for task-incremental and class-incremental continual learning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PreAllocatedSparseHead(nn.Module):
    """
    Pre-Allocated Sparse Head (PASH).
    Pre-allocates the maximum classification capacity at initialization
    and applies a hard -inf mask on unseen classes to prevent illegal logits,
    eliminating GPU memory reallocation and fragmentation across tasks.
    """
    def __init__(self, in_features: int, max_classes: int = 100, initial_classes: int = 0):
        super().__init__()
        self.in_features = in_features
        self.max_classes = max_classes
        self.current_active_classes = initial_classes
        self.classifier = nn.Linear(in_features, max_classes)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.classifier.weight, a=math.sqrt(5))
            if self.classifier.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.classifier.weight)
                bound = 1.0 / math.sqrt(fan_in if fan_in > 0 else 1)
                nn.init.uniform_(self.classifier.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(x)
        if self.current_active_classes < self.max_classes:
            mask = torch.full_like(logits, float('-inf'))
            mask[:, :self.current_active_classes] = 0.0
            logits = logits + mask
        return logits


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inp: int, out: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out, out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        self.shortcut = nn.Sequential()
        if stride != 1 or inp != out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inp: int, out: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, out, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out, out, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        self.conv3 = nn.Conv2d(out, out * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or inp != out * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, out * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out * self.expansion)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    """
    ResNet-18 architecture with configurable input conv (CIFAR 3x3 vs ImageNet 7x7),
    configurable layer-4 width, and Pre-Allocated Sparse Head (PASH).
    """
    def __init__(self, num_classes: int = 10, cifar_style: bool = True, layer4_width: int = 512):
        super().__init__()
        self.cifar_style = cifar_style
        self.inp = 64
        self.num_classes = num_classes
        self.layer4_width = layer4_width
        self.active_classes = num_classes

        if cifar_style:
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.Identity()
            self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        else:
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)

        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, layer4_width, 2, stride=2)
        self.head = PreAllocatedSparseHead(layer4_width, max_classes=num_classes, initial_classes=num_classes)

    def _make_layer(self, block, out: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.inp, out, s))
            self.inp = out * block.expansion
        return nn.Sequential(*layers)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)

    def extract_layer_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extracts spatial average-pooled activation representations from all layer stages:
        'conv1' (64), 'layer1' (64), 'layer2' (128), 'layer3' (256), 'layer4' (512).
        Used for CKA, covariance eigenspectra decay, and effective rank analysis.
        """
        out0 = F.relu(self.bn1(self.conv1(x)))
        pool0 = F.adaptive_avg_pool2d(out0, 1).view(out0.size(0), -1)
        out1 = self.maxpool(out0)
        out1 = self.layer1(out1)
        pool1 = F.adaptive_avg_pool2d(out1, 1).view(out1.size(0), -1)
        out2 = self.layer2(out1)
        pool2 = F.adaptive_avg_pool2d(out2, 1).view(out2.size(0), -1)
        out3 = self.layer3(out2)
        pool3 = F.adaptive_avg_pool2d(out3, 1).view(out3.size(0), -1)
        out4 = self.layer4(out3)
        pool4 = F.adaptive_avg_pool2d(out4, 1).view(out4.size(0), -1)
        return {
            'conv1': pool0,
            'layer1': pool1,
            'layer2': pool2,
            'layer3': pool3,
            'layer4': pool4
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.head(feat)


class ResNet50(nn.Module):
    """
    ResNet-50 architecture with Pre-Allocated Sparse Head (PASH)
    for larger-scale continua like Split ImageNet-100.
    """
    def __init__(self, num_classes: int = 100, cifar_style: bool = False):
        super().__init__()
        self.cifar_style = cifar_style
        self.inp = 64
        self.num_classes = num_classes

        if cifar_style:
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        self.head = PreAllocatedSparseHead(2048, max_classes=num_classes, initial_classes=num_classes)

    def _make_layer(self, block, out: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.inp, out, s))
            self.inp = out * block.expansion
        return nn.Sequential(*layers)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.head(feat)



class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    """Multi-head Self-Attention with per-head projection."""
    def __init__(self, dim: int, num_heads: int = 4, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class Mlp(nn.Module):
    """Two-layer MLP with GELU non-linearity."""
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    """Pre-LN Transformer Encoder Block."""
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Compact Vision Transformer (ViT) for continual representation learning.
    Uses LayerNorm rather than BatchNorm to investigate normalization dynamics.
    """
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_chans: int = 3,
                 num_classes: int = 10, embed_dim: int = 192, depth: int = 6,
                 num_heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.num_classes = num_classes
        self.active_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = PreAllocatedSparseHead(embed_dim, max_classes=num_classes, initial_classes=num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.head(feat)


class MobileNetV3Small(nn.Module):
    """
    MobileNetV3-Small architecture for modern edge continual learning.
    Uses depthwise-separable convolutions, inverted residual bottleneck blocks,
    Squeeze-and-Excitation (SE), Hard-Swish activations, and Pre-Allocated Sparse Head (PASH).
    """
    def __init__(self, num_classes: int = 10, cifar_style: bool = True):
        super().__init__()
        import torchvision.models as tv_models
        base = tv_models.mobilenet_v3_small(weights=None)
        if cifar_style:
            # CIFAR images are 32x32: adjust initial conv stride from 2 to 1 for spatial resolution
            base.features[0][0].stride = (1, 1)
        self.features_block = base.features
        self.avgpool = base.avgpool
        self.num_classes = num_classes
        self.active_classes = num_classes
        self.head = PreAllocatedSparseHead(576, max_classes=num_classes, initial_classes=num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features_block(x)
        out = self.avgpool(out)
        return torch.flatten(out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.head(feat)


def create_model(arch: str = "resnet18", initial_classes: int = 10, max_classes: int = 10, cifar_style: bool = True, layer4_width: int = 512) -> nn.Module:
    """Factory function to instantiate models for continual learning experiments."""
    arch = arch.lower()
    if arch == "resnet18":
        model = ResNet18(num_classes=max_classes, cifar_style=cifar_style, layer4_width=layer4_width)
        model.head.current_active_classes = initial_classes
        model.active_classes = initial_classes
        return model
    elif arch == "resnet50":
        model = ResNet50(num_classes=max_classes, cifar_style=cifar_style)
        model.head.current_active_classes = initial_classes
        model.active_classes = initial_classes
        return model
    elif arch in ("vit", "transformer"):
        model = VisionTransformer(num_classes=max_classes, embed_dim=192, depth=6, num_heads=4)
        model.head.current_active_classes = initial_classes
        model.active_classes = initial_classes
        return model
    elif arch in ("mobilenetv3", "mobilenetv3_small", "mobilenet"):
        model = MobileNetV3Small(num_classes=max_classes, cifar_style=cifar_style)
        model.head.current_active_classes = initial_classes
        model.active_classes = initial_classes
        return model
    else:
        raise ValueError(f"Unsupported architecture: {arch}. Choose 'resnet18', 'resnet50', 'vit', or 'mobilenetv3_small'.")

