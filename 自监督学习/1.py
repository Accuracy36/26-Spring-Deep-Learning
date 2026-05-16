import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score
import numpy as np
import os

# ==========================================
# 0. 全局设置与设备选择
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的计算设备: {DEVICE}")

DATA_DIR = "./CIFAKE" 
# ==========================================
# 1. 数据增强与加载 (Data Pipeline)
# ==========================================
class SimCLRTransform:
    def __init__(self, size=32):
        s = 1.0
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
        ])

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)


normal_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

def get_dataloaders(batch_size=256, label_ratio=0.1):
    train_dataset_simclr = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=SimCLRTransform())
    train_dataset_linear = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=normal_transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=normal_transform)

    # 提取 10% 的带标签数据用于下游任务
    num_train = len(train_dataset_linear)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    subset_indices = indices[:int(num_train * label_ratio)]
    
    subset_dataset = Subset(train_dataset_linear, subset_indices)

    simclr_loader = DataLoader(train_dataset_simclr, batch_size=batch_size, shuffle=True, drop_last=True)
    linear_loader = DataLoader(subset_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return simclr_loader, linear_loader, test_loader

# ==========================================
# 2. 模型定义 (Encoder + Projection Head)
# ==========================================
class SimCLRModel(nn.Module):
    def __init__(self, feature_dim=128, projection_dim=64, use_relu=True, encoder_type='resnet18'):
        super().__init__()
        if encoder_type == 'resnet18':
            resnet = models.resnet18(weights=None)
            resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            resnet.maxpool = nn.Identity()
            resnet.fc = nn.Identity() 
            self.encoder = resnet
            encoder_out_dim = 512 
            
        elif encoder_type == 'mobilenet_v2':
            mobilenet = models.mobilenet_v2(weights=None)
            mobilenet.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
            mobilenet.classifier = nn.Identity()
            self.encoder = mobilenet
            encoder_out_dim = 1280 
        else:
            raise ValueError("Unsupported encoder type")
        # 附加实验：消融 ReLU
        if use_relu:
            self.projection_head = nn.Sequential(
                nn.Linear(encoder_out_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, projection_dim)
            )
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(encoder_out_dim, feature_dim),
                nn.Linear(feature_dim, projection_dim)
            )

    def forward(self, x, return_features=False):
        features = self.encoder(x)
        if return_features:
            return features
        return self.projection_head(features)

# ==========================================
# 3. 损失函数 (NT-Xent)
# ==========================================
def info_nce_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, p=2, dim=1) 
    sim_matrix = torch.matmul(z, z.T) / temperature
    
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim_matrix.masked_fill_(mask, -9e15) 
    
    pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
    for i in range(batch_size):
        pos_mask[i, i + batch_size] = True
        pos_mask[i + batch_size, i] = True
        
    pos_sim = sim_matrix[pos_mask].view(2 * batch_size, 1)
    log_prob = pos_sim - torch.logsumexp(sim_matrix, dim=1, keepdim=True)
    loss = -log_prob.mean()
    return loss

# ==========================================
# 4. 训练流程封装
# ==========================================
def pretrain(model, dataloader, epochs=5, temperature=0.5):
    """阶段一：无监督预训练"""
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    print(f"\n--- 开始 SimCLR 预训练 (Temperature={temperature}) ---")
    for epoch in range(epochs):
        total_loss = 0
        for (view_1, view_2), _ in dataloader:
            view_1, view_2 = view_1.to(DEVICE), view_2.to(DEVICE)
            
            optimizer.zero_grad()
            z_i = model(view_1)
            z_j = model(view_2)
            loss = info_nce_loss(z_i, z_j, temperature)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

def linear_probe(model, train_loader, test_loader, epochs=5):
    """阶段二：冻结特征，训练线性分类器"""
    model.eval()
    for param in model.encoder.parameters():
        param.requires_grad = False
        
    classifier = nn.Linear(512, 2).to(DEVICE)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print("\n--- 开始 Linear Probing (仅使用少量标签) ---")
    for epoch in range(epochs):
        classifier.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.no_grad():
                features = model(images, return_features=True)
            
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # 评估
    return evaluate(model, classifier, test_loader)

def train_baseline(train_loader, test_loader, epochs=5):
    """Baseline：不预训练，直接全监督训练（但只给10%数据）"""
    print("\n--- 开始 Baseline 训练 (无预训练直接分类) ---")
    model = SimCLRModel().to(DEVICE)
    classifier = nn.Linear(512, 2).to(DEVICE)
    
    params = list(model.encoder.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params, lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        classifier.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            features = model(images, return_features=True)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    return evaluate(model, classifier, test_loader, is_baseline=True)

def evaluate(model, classifier, test_loader, is_baseline=False):
    if not is_baseline: model.eval()
    classifier.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            features = model(images, return_features=True)
            outputs = classifier(features)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"评估结果 -> Accuracy: {acc*100:.2f}%, F1 Score: {f1:.4f}")
    return acc, f1

# ==========================================
# 5. 主程序与实验执行
# ==========================================
if __name__ == '__main__':
    # 准备数据
    simclr_loader, linear_loader, test_loader = get_dataloaders(batch_size=128, label_ratio=0.1)
    
    # 【实验 A】: Baseline (不进行预训练)
    acc_base, f1_base = train_baseline(linear_loader, test_loader, epochs=10)
    
    # 【实验 B】: 标准 SimCLR 流程
    model_simclr = SimCLRModel(use_relu=True).to(DEVICE)
    pretrain(model_simclr, simclr_loader, epochs=10, temperature=0.5)
    acc_simclr, f1_simclr = linear_probe(model_simclr, linear_loader, test_loader, epochs=10)
    
    # 【附加实验 1】: 超参数分析 (降低温度系数到 0.1)
    model_temp01 = SimCLRModel(use_relu=True).to(DEVICE)
    pretrain(model_temp01, simclr_loader, epochs=10, temperature=0.1)
    acc_temp01, _ = linear_probe(model_temp01, linear_loader, test_loader, epochs=10)
    
    # 【附加实验 2】: 移除 Projection Head 的 ReLU
    model_no_relu = SimCLRModel(use_relu=False).to(DEVICE)
    pretrain(model_no_relu, simclr_loader, epochs=10, temperature=0.5)
    acc_no_relu, _ = linear_probe(model_no_relu, linear_loader, test_loader, epochs=10)
    
    # 汇总输出
    print("\n================ 最终实验结果汇总 ================")
    print(f"1. Baseline (随机初始化): Acc = {acc_base*100:.2f}%")
    print(f"2. SimCLR (标准, Tau=0.5, 带ReLU): Acc = {acc_simclr*100:.2f}%")
    print(f"3. 附加实验1 (修改温度Tau=0.1): Acc = {acc_temp01*100:.2f}%")
    print(f"4. 附加实验2 (移除头部的ReLU): Acc = {acc_no_relu*100:.2f}%")
    print("==================================================")
