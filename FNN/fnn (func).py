import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
def prepare_data():
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42
    )
    

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).view(-1, 1)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val).view(-1, 1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).view(-1, 1)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

class FNN(nn.Module):
    def __init__(self, hidden_dims, activation_fn=F.relu):
        super(FNN, self).__init__()
        self.activation_fn = activation_fn
        layers = []
        
        # 输入层维度固定为 10
        input_dim = 10
        
        # 动态添加隐藏层
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            input_dim = h_dim
            
        self.hidden_layers = nn.ModuleList(layers)
        # 输出层：1个神经元用于回归
        self.output = nn.Linear(input_dim, 1)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation_fn(layer(x))
        return self.output(x)

# 训练模型
def train_model(model, X_train, y_train, X_val, y_val, lr=0.001, epochs=200):
    # 损失函数: 均方误差 (MSE)
    criterion = nn.MSELoss()
    # 优化器: Adam
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        # 训练过程
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # 验证过程
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            v_loss = criterion(val_outputs, y_val)
            
        train_losses.append(loss.item())
        val_losses.append(v_loss.item())
        
        # 保存基于验证集损失的最佳模型参数
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_model_state = model.state_dict()

    return train_losses, val_losses, best_model_state, best_val_loss


def run_experiment():
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()

    lr=0.01

    fixed_dims=[64,32]

    activation_configs = {
        "ReLU": F.relu,
        "Sigmoid": torch.sigmoid,
        "Tanh": torch.tanh,
        "LeakyReLU": F.leaky_relu,
        "Swish": F.silu
    }

    plt.figure(figsize=(10, 6))

    for name, activation_fn in activation_configs.items():
        print(f"正在测试激活函数: {name}...")
        model = FNN(hidden_dims=fixed_dims, activation_fn=activation_fn)
        t_losses, v_losses, best_state, _ = train_model(model, X_train, y_train, X_val, y_val,lr,200)
        
        # 在测试集上评估最佳模型
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = nn.MSELoss()(test_outputs, y_test)
            print(f"激活函数: {name} - 测试集 MSE: {test_loss.item():.2f}")
        
        # 画出验证集损失曲线
        plt.plot(v_losses, label=f"activation function: {name} (Val Loss)")

    plt.title("loss of different learning rate")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale('log')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_experiment()
