import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Internal imports
from dataset import GaitDataset
from model import GaitLSTM # ResNet-1D

def log(msg, error=False):
    target = sys.stderr if error else sys.stdout
    target.write(f"[{'ERROR' if error else 'INFO'}] {msg}\n")
    target.flush()

# ---------------------------------------------------------------------------
# GLOBAL SCALER (Normalizes all 69 features for zero-loss stability)
# ---------------------------------------------------------------------------
class GlobalScaler:
    def __init__(self, feature_dim):
        self.mean = torch.zeros(feature_dim)
        self.std  = torch.ones(feature_dim)

    def fit(self, dataset):
        log("Fitting Scaler on all raw video files...")
        all_data = []
        # Fit on raw files directly for speed
        for path in dataset.data_paths:
            seq = np.load(path)
            all_data.append(torch.tensor(seq, dtype=torch.float32))
        all_data = torch.cat(all_data, dim=0)
        self.mean = all_data.mean(dim=0)
        self.std  = all_data.std(dim=0) + 1e-6
        log(f"Scaler Results: Mean range [{self.mean.min():.1f}, {self.mean.max():.1f}]")

    def transform(self, seq):
        return (seq - self.mean) / self.std

    def save(self, path):
        torch.save({'mean': self.mean, 'std': self.std}, path)

# ---------------------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------------------
def train_model(data_dir, epochs=500, lr=1e-3):
    log("="*50)
    log("  GAIT ZERO-LOSS MISSION: 500 EPOCHS + SLIDING WINDOW")
    log("="*50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")

    # 1. Dataset setup (Multi-crop enabled in dataset.py)
    full_train_ds = GaitDataset(data_dir, max_frames=60, is_train=True)
    val_ds        = GaitDataset(data_dir, max_frames=60, is_train=False)

    num_classes = full_train_ds.get_num_classes()
    sample_seq, _, _ = full_train_ds[0]
    input_size = sample_seq.shape[-1]
    
    log(f"Detected {input_size} features. Standardizing data...")
    
    scaler = GlobalScaler(input_size)
    scaler.fit(full_train_ds)
    os.makedirs('models', exist_ok=True)
    scaler.save('models/scaler.pth')

    train_loader = DataLoader(full_train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,        batch_size=8,  shuffle=False)

    # 2. Model (ResNet-1D)
    # dropout=0.5 for Good-Fit as per plan
    model = GaitLSTM(input_size=input_size, num_classes=num_classes, dropout=0.5).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-2)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs
    )

    history = {'t_loss': [], 'v_loss': [], 'v_acc': [], 'v_g_acc': []}
    best_v_loss = 1e9
    patience = 50
    no_improve_count = 0

    log(f"Starting Training ({len(full_train_ds)} virtual samples)...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for seqs, labels, gender_labels in train_loader:
            # Apply Scaler Normalization
            seqs = torch.stack([scaler.transform(s) for s in seqs])
            seqs, labels, gender_labels = seqs.to(device), labels.to(device), gender_labels.to(device)
            
            optimizer.zero_grad()
            id_out, g_out = model(seqs)
            
            loss_id = criterion(id_out, labels)
            loss_gender = criterion(g_out, gender_labels)
            loss = loss_id + loss_gender
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        v_loss = 0.0
        correct_id = 0
        correct_gen = 0
        total = 0
        with torch.no_grad():
            for seqs, labels, gender_labels in val_loader:
                seqs = torch.stack([scaler.transform(s) for s in seqs])
                seqs, labels, gender_labels = seqs.to(device), labels.to(device), gender_labels.to(device)
                
                id_out, g_out = model(seqs)
                
                loss_id = criterion(id_out, labels)
                loss_gender = criterion(g_out, gender_labels)
                v_loss += (loss_id + loss_gender).item()
                
                _, pred_id = torch.max(id_out, 1)
                _, pred_gen = torch.max(g_out, 1)
                
                total += labels.size(0)
                correct_id += (pred_id == labels).sum().item()
                correct_gen += (pred_gen == gender_labels).sum().item()
        
        v_loss /= len(val_loader)
        v_acc = 100 * correct_id / total
        v_g_acc = 100 * correct_gen / total
        
        history['t_loss'].append(train_loss)
        history['v_loss'].append(v_loss)
        history['v_acc'].append(v_acc)
        history['v_g_acc'].append(v_g_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log(f"Epoch {epoch+1:03d} | T-Loss: {train_loss:.4f} | V-Loss: {v_loss:.4f} | ID-Acc: {v_acc:.1f}% | G-Acc: {v_g_acc:.1f}%")

        if v_loss < best_v_loss:
            best_v_loss = v_loss
            no_improve_count = 0
            torch.save(model.state_dict(), 'models/best_gait_model.pth')
        else:
            no_improve_count += 1
            
        if no_improve_count >= patience:
            log(f"Early Stopping at epoch {epoch+1} (Good Fit reached).")
            break

    # Final curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['t_loss'], label='Train Loss')
    plt.plot(history['v_loss'], label='Val Loss')
    plt.title('Loss Reduction')
    plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history['v_acc'], label='ID Accuracy')
    plt.plot(history['v_g_acc'], label='Gender Accuracy')
    plt.title('Validation Accuracy')
    plt.legend(); plt.grid(True)
    plt.savefig('models/training_curves.png')
    
    log(f"COMPLETE. Best Val Loss: {best_v_loss:.4f}")

if __name__ == "__main__":
    data_path = os.path.join('dataset', 'processed')
    train_model(data_path)
