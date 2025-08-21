import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle
import json
import os

# M1/M2 맥북에서 GPU 가속
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS 디바이스 사용!")
else:
    device = torch.device("cpu")
    print("CPU 사용")

# 🧪 실험 1 설정 - HIDDEN_SIZE만 변경
SEQ_LEN = 100      # 베이스라인 유지
HIDDEN_SIZE = 160  # 128 → 160 (유일한 변경!)
NUM_LAYERS = 1     # 베이스라인 유지
BATCH_SIZE = 32    # 베이스라인 유지
DROPOUT = 0.2      # 베이스라인 유지
EPOCHS = 10         # 베이스라인 유지
LEARNING_RATE = 0.001

class TrafficDataset(Dataset):
    def __init__(self, data, targets, seq_len=SEQ_LEN):
        self.data = data
        self.targets = targets
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]  # 시퀀스 (seq_len, features)
        y = self.targets[idx + self.seq_len]   # peak_volume 값
        return torch.FloatTensor(x), torch.FloatTensor([y])

class LSTMModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,  # 2층일 때만 dropout
            bidirectional=False  # 👈 이걸 False로!
        )
        # 양방향이므로 hidden_size * 2
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)  # 160 -> 80
        self.fc2 = nn.Linear(hidden_size // 2, 1)            # FC 2층만!
        self.relu = nn.ReLU()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size) ← *2 제거
        x = self.dropout(last_output)
        x = self.relu(self.fc1(x))        # hidden_size → hidden_size // 2
        x = self.dropout(x)               # dropout 추가로 정규화
        x = self.fc2(x)                   # hidden_size // 2 → 1 (최종 출력)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.L1Loss()(y_hat, y)  # MAE로 다시
        
        # MAE 계산 및 로깅
        mae = nn.L1Loss()(y_hat, y)
        self.log('train_loss', loss, prog_bar = True)
        self.log('train_mae', mae, prog_bar=True, logger = True)  # 진행바에 표시
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mae_loss = nn.L1Loss()(y_hat, y)
        
        self.log('val_loss', mae_loss)
        self.log('val_mae', mae_loss, prog_bar=True, logger = True)  # 동일한 값
        return mae_loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

        # 🆕 여기에 추가!
    def on_train_epoch_end(self):
        train_mae = self.trainer.callback_metrics.get('train_mae', 'N/A')
        print(f"Epoch {self.current_epoch + 1}: Train MAE = {train_mae}")

    def on_validation_epoch_end(self):
        val_mae = self.trainer.callback_metrics.get('val_mae', 'N/A') 
        print(f"Epoch {self.current_epoch + 1}: Val MAE = {val_mae}")

def train_model(model_name="traffic_lstm_v7", 
                csv_path="./task1_data/train_data.csv",
                save_path="./model"):
    os.makedirs(save_path, exist_ok=True)
    
    # 데이터 로드
    df = pd.read_csv(csv_path)
    print(f"데이터 로드 완료: {df.shape}")
    
    # 입력 특성 (27개, peak_volume 제외)
    input_cols = ['fwd_pkt_count', 'bwd_pkt_count', 'fwd_tcp_pkt_count', 'bwd_tcp_pkt_count',
                  'fwd_udp_pkt_count', 'bwd_udp_pkt_count', 'traffic_volume',
                  'fwd_tcp_flags_cwr_count', 'bwd_tcp_flags_cwr_count', 'fwd_tcp_flags_ecn_count',
                  'bwd_tcp_flags_ecn_count', 'fwd_tcp_flags_ack_count', 'bwd_tcp_flags_ack_count',
                  'fwd_tcp_flags_push_count', 'bwd_tcp_flags_push_count', 'fwd_tcp_flags_reset_count',
                  'bwd_tcp_flags_reset_count', 'fwd_tcp_flags_syn_count', 'bwd_tcp_flags_syn_count',
                  'fwd_tcp_flags_fin_count', 'bwd_tcp_flags_fin_count', 'fwd_tcp_window_size_avg',
                  'bwd_tcp_window_size_avg', 'fwd_tcp_window_size_max', 'bwd_tcp_window_size_max',
                  'fwd_tcp_window_size_min', 'bwd_tcp_window_size_min']
    
    # 데이터 전처리
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X = x_scaler.fit_transform(df[input_cols].values)  # 입력 특성
    y = y_scaler.fit_transform(df[['peak_volume']].values).flatten()  # 타겟
    
    # 데이터 분할
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # 데이터셋 생성
    train_dataset = TrafficDataset(X_train, y_train)
    val_dataset = TrafficDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 모델 생성 및 학습
    model = LSTMModel(len(input_cols), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT)
    trainer = pl.Trainer(
        max_epochs=EPOCHS, 
        enable_progress_bar=True, 
        accelerator="auto",    # 👈 이거 추가
        devices=1              # 👈 이거 추가)
    )
    trainer.fit(model, train_loader, val_loader)
    
    # 저장 경로 설정
    model_path = f'{save_path}/{model_name}.pth'
    x_scaler_path = f'{save_path}/{model_name}_scaler.pkl'
    y_scaler_path = f'{save_path}/{model_name}_y_scaler.pkl'
    meta_path = f'{save_path}/{model_name}_meta.json'
    
    # 저장
    torch.save(model.state_dict(), model_path)
    with open(x_scaler_path, 'wb') as f:
        pickle.dump(x_scaler, f)
    with open(y_scaler_path, 'wb') as f:
        pickle.dump(y_scaler, f)
    
    metadata = {
        'cols': input_cols, 
        'seq_len': SEQ_LEN, 
        'hidden_size': HIDDEN_SIZE,
        'input_size': len(input_cols),
        'num_layers': NUM_LAYERS,    # 수정
        'dropout': DROPOUT           # 수정
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f)
    
    print(f"모델 저장 완료:")
    print(f"- 모델: {model_path}")
    print(f"- X 스케일러: {x_scaler_path}")
    print(f"- Y 스케일러: {y_scaler_path}")
    print(f"- 메타데이터: {meta_path}")
    return model, x_scaler, y_scaler

if __name__ == "__main__":
    print("=== LSTM 모델 학습 시작 ===")
    model, x_scaler, y_scaler = train_model("traffic_lstm_v7")
    print("=== 학습 완료! ===")