# 导入必要库
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,RobustScaler
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

class EGATLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim, attn_hidden_layers=3):
        super().__init__()
        self.out_channels = out_channels
        
        self.lin_source = torch.nn.Linear(in_channels, out_channels) 
        self.lin_target = torch.nn.Linear(in_channels, out_channels)   
        self.lin_edge = torch.nn.Linear(edge_dim, out_channels)

        input_dim = 3 * out_channels
        self.attn_mlp = torch.nn.Sequential()

        hidden_dims = [
            #max(out_channels*2, 1),
            max(out_channels*1, 1),
            max(out_channels//2, 1),
            #max(out_channels//4, 1),
        ][:attn_hidden_layers]
    
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            self.attn_mlp.append(torch.nn.Linear(prev_dim, h_dim))
            self.attn_mlp.append(torch.nn.LeakyReLU())
            prev_dim = h_dim
        
        self.attn_mlp.append(torch.nn.Linear(prev_dim, 1))
        #self.attn_mlp.append(torch.nn.Softplus())
        #self.attn_mlp.append(torch.nn.ReLU())
        
        '''for layer in self.attn_mlp:
            if isinstance(layer, torch.nn.Linear):
                if layer.out_features == 1:
                    torch.nn.init.xavier_uniform_(layer.weight, gain=0.1)
                else:
                    torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')'''
        
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.raw_attention = None

        self.phy_loss = torch.tensor(0.0).to(device) 

    def forward(self, x, edge_index, edge_attr):
        source, target = edge_index
        if self.training:
        # 重新 detach 并附加梯度标记，保证后续操作被记录
            edge_attr = edge_attr.clone().detach().requires_grad_(True)
            x         = x.clone().detach().requires_grad_(True)
        x_source = self.lin_source(x)
        x_target = self.lin_target(x)
        edge_trans = self.lin_edge(edge_attr)

        combined = torch.cat([x_source[source], x_target[target], edge_trans], dim=1)  #128*3
        e = self.attn_mlp(combined).squeeze()
        self.raw_attention = e
        e = torch.relu(e)
        
        out = torch.zeros_like(x_source)
        out.index_add_(
            0, target,
            e.unsqueeze(-1) * x_source[source]
        )

        # 新增物理约束计算
        if self.training:
            with torch.enable_grad():
                grad_edge = torch.autograd.grad(
                    e, edge_attr, 
                    grad_outputs=torch.ones_like(e),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0]
                
                grad_pop = torch.autograd.grad(
                    e, x_target,
                    grad_outputs=torch.ones_like(e),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0][target, 2]

            # 计算平均梯度
            distance_grad_mean = grad_edge[:, 1].mean()  # distance是第二个维度
            weight_grad_mean = grad_edge[:, 0].mean()    # weight是第一个维度
            pop_grad_mean = grad_pop.mean()

            # 宽松约束（允许个别样本违反，约束整体趋势）
            distance_loss = torch.relu(distance_grad_mean)  # 整体distance梯度应<=0
            weight_loss = torch.relu(-weight_grad_mean)     # 整体weight梯度应>=0
            pop_loss = torch.relu(-pop_grad_mean)      # 整体population梯度应>=0

            # 调整损失系数（根据需要调整）
            self.phy_loss =  distance_loss+ weight_loss + pop_loss   

        return out + x_target

class EGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim):
        super().__init__()
        self.conv1 = EGATLayer(in_channels, hidden_channels, edge_dim, attn_hidden_layers=3)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = EGATLayer(hidden_channels, hidden_channels, edge_dim, attn_hidden_layers=3)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels,out_channels)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin(x)
        return x
def prepare_data():
    # 1. 加载数据（请根据实际路径修改）
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

    # 3. 数据集划分
    cities = city_panel_df['city'].unique()
    indices = np.arange(len(cities))
    train_idx, test_idx = train_test_split(indices, test_size=0.1, shuffle= True,random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.3/0.9,shuffle=True
                                          , random_state=42)
    # 4. 隔离标签标准化（关键修改！）
    # 特征仍然全局标准化（因测试节点特征在训练时可见）
    scaler_node = StandardScaler()
    city_panel_df[features] = scaler_node.fit_transform(city_panel_df[features])  # 保持原特征标准化
    
    # 标签仅用训练集标准化
    scaler_label = StandardScaler()
    # 提取训练集标签进行拟合
    train_labels = city_panel_df.loc[city_panel_df.index.isin(train_idx), [target]]
    scaler_label.fit(train_labels)
    # 应用至全体数据
    city_panel_df[target] = scaler_label.transform(city_panel_df[[target]])
    # 5. 边处理
    valid_edges_mask = (
        network_df['o'].isin(city_panel_df['city']) & 
        network_df['d'].isin(city_panel_df['city'])
    )
    network_df = network_df[valid_edges_mask].reset_index(drop=True)
    
    scaler_edge = StandardScaler()
    network_df[['weight', 'distance']] = scaler_edge.fit_transform(network_df[['weight', 'distance']])

    # 6. 构建图数据
    city_to_idx = {city: idx for idx, city in enumerate(city_panel_df['city'])}
    
    edge_source = network_df['o'].map(city_to_idx).values.astype(int)
    edge_target = network_df['d'].map(city_to_idx).values.astype(int)
    edge_index = torch.tensor([edge_source, edge_target], dtype=torch.long)
    
    edge_attr = torch.tensor(network_df[['weight', 'distance']].values, dtype=torch.float)
    node_features = torch.tensor(city_panel_df[features].values, dtype=torch.float)
    labels = torch.tensor(city_panel_df[target].values, dtype=torch.float).view(-1, 1)
    
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=labels
    ).to(device)
    
    # 7. 设置掩码（不再遮蔽特征）
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
    model = EGAT(
        in_channels=data.x.shape[1],
        hidden_channels=128,
        out_channels=1,
        edge_dim=data.edge_attr.shape[1]
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    criterion = torch.nn.MSELoss()

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    train_losses = []
    val_losses = []
    phy_losses = []  # 新增物理损失记录

    train_loader = NeighborLoader(
        data,
        num_neighbors=[-1],
        batch_size=32,
        input_nodes=data.train_mask,
        shuffle=True,
    )
    
    best_val_loss = float('inf')
    patience = 30
    counter = 0
    
    for epoch in range(1, 1000):
        model.train()
        total_loss = 0
        total_phy_loss = 0  # 新增

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])

            # 计算物理损失
            # 计算总损失（添加物理约束项）
            phy_loss = model.conv1.phy_loss + model.conv2.phy_loss
            total_phy_loss += phy_loss.item()
            loss = criterion(out[batch.train_mask], batch.y[batch.train_mask]) + phy_loss
            '''# L1正则化
            l1_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l1_reg += torch.norm(param, 1)
            loss += 3e-6 * l1_reg'''
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        # 每个epoch结束后，更新学习率
        #scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        avg_phy_loss = total_phy_loss / len(train_loader)
        phy_losses.append(avg_phy_loss)

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_attr)
            val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
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
        ax.plot(phy_losses, label='Physics Loss', color='g', marker='^', markersize=3)
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
        preds = model(data.x, data.edge_index, data.edge_attr)
        
        preds_np = preds.cpu().numpy()
        true_np = data.y.cpu().numpy()
        preds_denorm = scaler.inverse_transform(preds_np)
        true_denorm = scaler.inverse_transform(true_np)
        
        metrics = {}
        for mask_name in ['train', 'val', 'test']:
            mask = getattr(data, f'{mask_name}_mask').cpu().numpy()
            y_true = true_denorm[mask]
            y_pred = preds_denorm[mask]

            metrics[mask_name] = {
                'mae': mean_absolute_error(y_true, y_pred),
                'mse': mean_squared_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred),
                'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if len(y_true) > 0 else np.nan
            }
            
            print(f"EGAT | {mask_name} Metrics:")
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
        print(f"Error loading model: {e}")
    
    metrics = evaluate_model(model, data, scaler)
    
    result_df = pd.DataFrame([metrics])
    result_df.to_csv("results/egat_results.csv", index=False)
    print("\n训练完成，结果已保存至 results/egat_results.csv")