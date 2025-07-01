# 导入必要库
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# 设置全局随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FCNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        return self.lin(x)

def prepare_data():
    # 1. 加载数据
    network_df = pd.read_csv(r'E:\data-main\key_data\5千万 人口迁徙数据-363个城市\人口迁徙数据-城市间（2020-2023.10）\第三版数据处理\network_df.csv')
    city_panel_df = pd.read_csv(r'E:\data-main\key_data\5千万 人口迁徙数据-363个城市\人口迁徙数据-城市间（2020-2023.10）\第三版数据处理\regression_data_fill.csv')
    
    # 2. 数据预处理
    cols_to_log = ['population', 'pergdp', 'budget', 'investment', 'invention']
    city_panel_df[cols_to_log] = np.log(city_panel_df[cols_to_log])
    
    features = ['weighted_degree', 'closeness_centrality', 'population', 'diversity', 
               'budget', 'investment','industry', 'invention', 'eigenvector_centrality','mobility_pro']
    target = 'pergdp'
    invalid_city = '三沙市'

    city_panel_df = city_panel_df[city_panel_df['city'] != invalid_city].reset_index(drop=True)
    city_panel_df[features] = city_panel_df[features].fillna(0)

    # 3. 数据集划分（按城市划分）
    cities = city_panel_df['city'].unique()
    indices = np.arange(len(cities))
    train_idx, test_idx = train_test_split(indices, test_size=0.1, shuffle=True,random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1/0.9, shuffle=True,random_state=42)

    # 4. 特征标准化（核心修改：仅用训练数据计算统计量）
    scaler_node = StandardScaler()
    scaler_label = StandardScaler()
    
    # 获取训练数据（关键！）
    train_cities = cities[train_idx]
    train_mask = city_panel_df['city'].isin(train_cities)
    train_data = city_panel_df[train_mask]
    
    # 仅用训练数据拟合标准化器
    scaler_node.fit(train_data[features])
    scaler_label.fit(train_data[[target]])
    
    # 对全体数据进行转换
    city_panel_df[features] = scaler_node.transform(city_panel_df[features])
    city_panel_df[target] = scaler_label.transform(city_panel_df[[target]])

    # 5. 构建节点数据（无图结构）
    node_features = torch.tensor(city_panel_df[features].values, dtype=torch.float)
    labels = torch.tensor(city_panel_df[target].values, dtype=torch.float).view(-1, 1)
    
    data = Data(
        x=node_features,
        y=labels
    ).to(device)
    
    # 6. 设置掩码（基于城市划分）
    train_mask = torch.zeros(len(cities), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros(len(cities), dtype=torch.bool)
    val_mask[val_idx] = True
    test_mask = torch.zeros(len(cities), dtype=torch.bool)
    test_mask[test_idx] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    return data, scaler_label

def train_model(data):
    model = FCNet(
        in_channels=data.x.shape[1],
        hidden_channels=128,
        out_channels=1
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    criterion = torch.nn.MSELoss()
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    train_losses = []
    val_losses = []
    
    # 创建普通DataLoader
    train_dataset = torch.utils.data.TensorDataset(
        data.x[data.train_mask],
        data.y[data.train_mask]
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )
    
    best_val_loss = float('inf')
    patience = 20
    counter = 0
    
    for epoch in range(1, 1000):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            out = model(batch_x)  # 仅输入节点特征
            loss = criterion(out, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证步骤
        model.eval()
        with torch.no_grad():
            val_x = data.x[data.val_mask].to(device)
            val_y = data.y[data.val_mask].to(device)
            val_pred = model(val_x)
            val_loss = criterion(val_pred, val_y)
            val_losses.append(val_loss.item())
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), 'best_model.pt')
            else:
                counter += 1
            
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
        ax.clear()
        ax.plot(train_losses, label='Train Loss', color='r', marker='o', markersize=3)
        ax.plot(val_losses, label='Val Loss', color='b', marker='s', markersize=3)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend()
        ax.set_title(f'Epoch {epoch} Loss (Current Val: {val_loss.item():.4f})')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.pause(0.01)
        
        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss.item():.4f}")
    plt.ioff()
    plt.show()
    return model

def evaluate_model(model, data, scaler):
    model.eval()
    with torch.no_grad():
        # 分别预测各数据集
        metrics = {}
        for mask_name in ['train', 'val', 'test']:
            mask = getattr(data, f'{mask_name}_mask')
            x = data.x[mask].to(device)
            y_true = data.y[mask].cpu().numpy()
            
            preds = model(x).cpu().numpy()
            
            preds_denorm = scaler.inverse_transform(preds)
            true_denorm = scaler.inverse_transform(y_true)
            
            metrics[mask_name] = {
                'mae': mean_absolute_error(true_denorm, preds_denorm),
                'mse': mean_squared_error(true_denorm, preds_denorm),
                'r2': r2_score(true_denorm, preds_denorm),
                'mape': np.mean(np.abs((true_denorm - preds_denorm) / true_denorm)) * 100
            }
            
            print(f"FCNet | {mask_name} Metrics:")
            print(f"  R²: {metrics[mask_name]['r2']:.3f}")
            print(f"  MAE: {metrics[mask_name]['mae']:.4f}")
            print(f"  MSE: {metrics[mask_name]['mse']:.4f}")
            print(f"  MAPE: {metrics[mask_name]['mape']:.3f}%")
            print("-"*40)
            
        return metrics

if __name__ == "__main__":
    data, scaler = prepare_data()
    model = train_model(data)
    
    try:
        model.load_state_dict(torch.load('best_model.pt'))
    except Exception as e:
        print(f"Model loading warning: {e}")
    
    metrics = evaluate_model(model, data, scaler)
    
    result_df = pd.DataFrame([metrics])
    os.makedirs("results", exist_ok=True)
    result_df.to_csv("results/fcnet_results.csv", index=False)
    print("\n训练完成，结果已保存至 results/fcnet_results.csv")