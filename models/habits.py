import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


# ==========================================
# 1. 数据加载与高级特征工程
# ==========================================

def load_device_events(filename="../mocker/device_events.json"):
    with open(filename, "r") as f:
        events = json.load(f)
    return sorted(events, key=lambda x: x['timestamp'])


try:
    events = load_device_events()
    df = pd.DataFrame(events)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
except Exception as e:
    print(f"数据加载失败: {e}")
    exit()

# A. 编码器 (Embedding 需要原始索引)
device_encoder = LabelEncoder()
event_encoder = LabelEncoder()
df['dev_idx'] = device_encoder.fit_transform(df['device_id'])
df['evt_idx'] = event_encoder.fit_transform(df['event_type'])

num_devices = len(device_encoder.classes_)
num_classes = len(event_encoder.classes_)

# B. 周期性时间编码 (建议 B: Sin/Cos 变换)
df['hour'] = df['timestamp'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
df['dayofweek'] = df['timestamp'].dt.dayofweek / 6.0  # 简单归一化到 0-1

# C. 时间间隔特征 (建议 B: Time Delta)
# 计算当前事件距离上一个事件的秒数
df['t_delta'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
# Log 处理防止长间隔数值过大，并归一化
df['t_delta_log'] = np.log1p(df['t_delta'])
scaler = MinMaxScaler()
df['t_delta_norm'] = scaler.fit_transform(df[['t_delta_log']])


# ==========================================
# 2. 多任务时序数据提取 (建议 D)
# ==========================================
def create_advanced_sequences(df, time_steps=12):
    X_ids = []  # 包含 [dev_idx, evt_idx]
    X_cont = []  # 包含 [hour_sin, hour_cos, dayofweek, t_delta_norm]
    y_class = []  # 预测下一个事件类型
    y_time = []  # 预测下一个时间间隔 (辅助任务)

    dev_arr = df['dev_idx'].values
    evt_arr = df['evt_idx'].values
    cont_cols = ['hour_sin', 'hour_cos', 'dayofweek', 't_delta_norm']
    cont_arr = df[cont_cols].values

    for i in range(len(df) - time_steps):
        # 输入序列
        X_ids.append(np.stack([dev_arr[i:i + time_steps], evt_arr[i:i + time_steps]], axis=1))
        X_cont.append(cont_arr[i:i + time_steps])
        # 标签 (预测第 i+time_steps 个)
        y_class.append(evt_arr[i + time_steps])
        y_time.append(cont_arr[i + time_steps, 3])  # 预测下一个 t_delta_norm

    return (torch.tensor(np.array(X_ids), dtype=torch.long),
            torch.tensor(np.array(X_cont), dtype=torch.float32),
            torch.tensor(np.array(y_class), dtype=torch.long),
            torch.tensor(np.array(y_time), dtype=torch.float32).view(-1, 1))


time_steps = 15
X_ids, X_cont, y_class, y_time = create_advanced_sequences(df, time_steps)

# 划分训练/测试
train_size = int(len(X_ids) * 0.8)
train_dataset = TensorDataset(X_ids[:train_size], X_cont[:train_size], y_class[:train_size], y_time[:train_size])
test_dataset = TensorDataset(X_ids[train_size:], X_cont[train_size:], y_class[train_size:], y_time[train_size:])

# ==========================================
# 0. 配置
# ==========================================
config = {
    # 特征维度
    "dev_emb_dim": 32,  # 设备 ID 嵌入维度
    "evt_emb_dim": 32,  # 事件类型嵌入维度
    "cont_emb_dim": 32,  # 连续特征（时间/间隔）投影维度

    # --- LSTM 参数 ---
    "lstm_layers": 4,        # LSTM 的层数

    # Transformer 结构
    "hidden_size": 256,  # 变换器隐藏层维度 (d_model)
    "nhead": 16,  # 多头注意力的头数 (必须能被 hidden_size 整除)
    "num_layers": 4,  # Transformer 层数
    "dim_feedforward": 512,  # 前馈网络中间层维度
    "dropout": 0.1,  # 防止过拟合的随机失活率

    # 训练参数
    "lr": 0.001,
    "batch_size": 512,
    "epochs": 200
}

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)


# ==========================================
# 3. 模型定义
# ==========================================
class AdvancedHabitModel(nn.Module):
    def __init__(self, num_dev, num_evt, cfg):
        super(AdvancedHabitModel, self).__init__()

        # 保存配置
        self.cfg = cfg

        # 1. Embedding 层: 映射离散 ID 到向量
        self.dev_emb = nn.Embedding(num_dev, cfg["dev_emb_dim"])
        self.evt_emb = nn.Embedding(num_evt, cfg["evt_emb_dim"])

        # 2. 连续特征投射: 处理时间 (Sin/Cos/Delta)
        self.cont_fc = nn.Linear(4, cfg["cont_emb_dim"])

        # 3. 输入融合投影
        # 自动计算拼接后的维度
        combined_dim = cfg["dev_emb_dim"] + cfg["evt_emb_dim"] + cfg["cont_emb_dim"]
        self.input_proj = nn.Linear(combined_dim, cfg["hidden_size"])

        # 4. Transformer Encoder
        # batch_first=True 方便我们处理 [batch, seq, feature] 格式
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg["hidden_size"],
            nhead=cfg["nhead"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg["num_layers"])

        # 5. 多任务输出头
        self.classifier = nn.Linear(cfg["hidden_size"], num_evt)  # 任务1: 事件分类
        self.time_regressor = nn.Linear(cfg["hidden_size"], 1)  # 任务2: 时间预测

    def generate_causal_mask(self, sz):
        # 因果遮罩：防止时间步 t 看到 t+1 的数据
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, ids, cont):
        # ids: [batch, seq, 2], cont: [batch, seq, 4]
        d_emb = self.dev_emb(ids[:, :, 0])
        e_emb = self.evt_emb(ids[:, :, 1])
        c_feat = torch.relu(self.cont_fc(cont))  # 增加非线性以增强表达力

        # 拼接并投影到 hidden_size
        x = torch.cat([d_emb, e_emb, c_feat], dim=-1)
        x = self.input_proj(x)

        # 应用遮罩进行时序特征提取
        mask = self.generate_causal_mask(x.size(1)).to(x.device)
        x = self.transformer(x, mask=mask)

        # 取序列的最后一个状态用于预测
        last_step = x[:, -1, :]

        return self.classifier(last_step), self.time_regressor(last_step)


class HybridHabitModel(nn.Module):
    def __init__(self, num_dev, num_evt, cfg):
        super(HybridHabitModel, self).__init__()

        # 1. Embedding 层
        self.dev_emb = nn.Embedding(num_dev, cfg["dev_emb_dim"])
        self.evt_emb = nn.Embedding(num_evt, cfg["evt_emb_dim"])
        self.cont_fc = nn.Linear(4, cfg["cont_emb_dim"])

        combined_dim = cfg["dev_emb_dim"] + cfg["evt_emb_dim"] + cfg["cont_emb_dim"]

        # 2. 新增 LSTM 层：作为底层特征提取器，捕捉初步的时序规律
        # cfg["lstm_hidden"] 建议设为与 hidden_size 一致
        self.lstm = nn.LSTM(
            input_size=combined_dim,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["lstm_layers"],
            batch_first=True,
            dropout=cfg["dropout"] if cfg["lstm_layers"] > 1 else 0
        )

        # 3. Transformer 层：在 LSTM 的输出基础上，进行全局关联建模
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg["hidden_size"],
            nhead=cfg["nhead"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg["num_layers"])

        # 4. 多任务输出头
        self.classifier = nn.Linear(cfg["hidden_size"], num_evt)
        self.time_regressor = nn.Linear(cfg["hidden_size"], 1)

    def generate_causal_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, ids, cont):
        # 特征融合
        d_emb = self.dev_emb(ids[:, :, 0])
        e_emb = self.evt_emb(ids[:, :, 1])
        c_feat = torch.relu(self.cont_fc(cont))
        x = torch.cat([d_emb, e_emb, c_feat], dim=-1)  # [batch, seq, combined_dim]

        # A. 先过 LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq, hidden_size]

        # B. 再过 Transformer
        mask = self.generate_causal_mask(lstm_out.size(1)).to(lstm_out.device)
        x = self.transformer(lstm_out, mask=mask)  # [batch, seq, hidden_size]

        # C. 取最后一个点
        last_step = x[:, -1, :]

        return self.classifier(last_step), self.time_regressor(last_step)


# ==========================================
# 4. 训练与 GPU 配置 (自动化初始化)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 实例化模型
# model = AdvancedHabitModel(num_devices, num_classes, config).to(device)
model = HybridHabitModel(num_devices, num_classes, config).to(device)

# 打印模型参数量，方便你判断模型是否过大
total_params = sum(p.numel() for p in model.parameters())
print(f"模型总参数量: {total_params:,}")

# 损失函数与优化器
criterion_cls = nn.CrossEntropyLoss()
criterion_reg = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["lr"])

# 建议：在训练循环中直接使用 config["batch_size"] 和 config["epochs"]

print(f"开始在 {device} 上训练...")
for epoch in range(config["epochs"]):
    model.train()
    total_loss = 0
    for b_ids, b_cont, b_y_cls, b_y_time in train_loader:
        b_ids, b_cont = b_ids.to(device), b_cont.to(device)
        b_y_cls, b_y_time = b_y_cls.to(device), b_y_time.to(device)

        optimizer.zero_grad()
        out_cls, out_time = model(b_ids, b_cont)

        loss_cls = criterion_cls(out_cls, b_y_cls)
        loss_reg = criterion_reg(out_time, b_y_time)

        # 联合损失: 分类为主，回归为辅
        loss = loss_cls + 0.5 * loss_reg
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{config["epochs"]}, Avg Loss: {total_loss / len(train_loader):.4f}")

# ==========================================
# 5. 评估与可视化
# ==========================================
model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for b_ids, b_cont, b_y_cls, _ in test_loader:
        out_cls, _ = model(b_ids.to(device), b_cont.to(device))
        pred = torch.argmax(out_cls, dim=1).cpu().numpy()
        all_preds.extend(pred)
        all_true.extend(b_y_cls.numpy())

all_preds = np.array(all_preds)
all_true = np.array(all_true)

print("\n" + "=" * 30)
print(f"测试集准确率: {np.mean(all_preds == all_true):.2%}")
print("=" * 30)
print(classification_report(all_true, all_preds, target_names=event_encoder.classes_, zero_division=0))

# 绘图对比 (最后 50 个点)
plt.figure(figsize=(15, 6))
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

display_len = 100
plt.scatter(range(display_len), all_true[-display_len:], color='blue', label='真实事件', alpha=0.5, s=100)
plt.scatter(range(display_len), all_preds[-display_len:], color='red', marker='x', label='预测事件', s=100)
plt.yticks(ticks=range(num_classes), labels=event_encoder.classes_)
plt.title('进阶模型预测结果对比 (Embedding + Multi-Task)')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()
plt.show()


# 模拟输入
batch_size = 1
seq_len = 15
mock_ids = torch.zeros((batch_size, seq_len, 2), dtype=torch.long).to(device)
mock_cont = torch.zeros((batch_size, seq_len, 4), dtype=torch.float32).to(device)

from torchinfo import summary


# 2. 打印结构
print("\n" + "="*20 + " 模型结构分析 " + "="*20)
model_stats = summary(
    model,
    input_data=(mock_ids, mock_cont),
    col_names=["input_size", "output_size", "num_params", "mult_adds"],
    depth=3,  # 深度为3可以看到 LSTM 和 Transformer 内部细分层
    row_settings=["var_names"],
    verbose=0
)
print(model_stats)