import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.fftpack import fft
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from tqdm import tqdm
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TimeFreqFeatExtractor:
    def extract_features(self, x):
        x = np.asarray(x)
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            x = np.nan_to_num(x)
        feats = [
            np.mean(x), np.std(x), np.max(x), np.min(x),
            np.median(x), np.percentile(x, 25), np.percentile(x, 75),
            np.max(x) - np.min(x), np.sum(np.diff(x) != 0) / (len(x) + 1e-6)
        ]
        xf = np.abs(fft(x))
        xf = xf[:len(x)//2]
        feats += [
            np.mean(xf), np.std(xf), np.max(xf), np.sum(xf>np.mean(xf)+2*np.std(xf)),
            np.sum(xf < np.mean(xf) - 2*np.std(xf))
        ]
        feats = np.nan_to_num(feats)
        return feats.astype(np.float32)

def extract_sequences(df, seq_len=64, stride=8, max_seqs=4000, augment=True):
    sig_cols = [c for c in df.columns if 'Signal' in c]
    label_col = 'Label' if 'Label' in df.columns else None
    extractor = TimeFreqFeatExtractor()
    X, y = [], []
    ids = df['ID'].unique()
    seqs_per_id = max(1, max_seqs // max(1,len(ids)))
    for cid in ids:
        sdf = df[df['ID']==cid]
        arr = sdf[sig_cols].values
        labels = sdf[label_col].values if label_col else np.zeros(len(sdf))
        for i in range(0, len(sdf)-seq_len+1, stride):
            window = arr[i:i+seq_len].flatten()
            feats = extractor.extract_features(window)
            label = 1 if np.any(labels[i:i+seq_len]) else 0
            if np.any(np.isnan(feats)) or np.any(np.isinf(feats)):
                continue
            X.append(feats)
            y.append(label)
            if (label==1 and augment):
                for _ in range(2):
                    X.append(feats + np.random.normal(0, 0.05, size=feats.shape))
                    y.append(1)
            elif (label==0 and augment and np.random.rand()<0.05):
                X.append(feats + np.random.normal(0, 0.01, size=feats.shape))
                y.append(0)
            if len(X) >= max_seqs: break
        if len(X) >= max_seqs: break
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    mask = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
    return X[mask], y[mask]

class CANSeqDataset(Dataset):
    def __init__(self, X, y, scaler=None):
        self.X = scaler.transform(X) if scaler is not None else X
        self.y = y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return torch.from_numpy(self.X[idx]), self.y[idx]

def balance_and_split(X, y, test_size=0.15):
    idx1 = np.where(y==1)[0]
    idx0 = np.where(y==0)[0]
    n1 = max(len(idx1), int(0.25*len(idx0)))
    if len(idx1) > 0:
        idx1 = np.random.choice(idx1, n1, replace=True)
    idx = np.concatenate([idx0, idx1])
    np.random.shuffle(idx)
    Xb, yb = X[idx], y[idx]
    return train_test_split(Xb, yb, test_size=test_size, stratify=yb, random_state=42)

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        self.U = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(rank, out_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
    def forward(self, x):
        x = x @ self.U @ self.V
        if self.bias is not None: x = x + self.bias
        return x

class LowRankAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], latent_dim=16, rank_factor=8, dropout=0.2):
        super().__init__()
        rank = max(1, min(hidden_dims[0], input_dim) // rank_factor)
        self.encoder = nn.Sequential(
            LowRankLinear(input_dim, hidden_dims[0], rank), nn.BatchNorm1d(hidden_dims[0]), nn.ReLU(), nn.Dropout(dropout),
            LowRankLinear(hidden_dims[0], hidden_dims[1], rank), nn.BatchNorm1d(hidden_dims[1]), nn.ReLU(), nn.Dropout(dropout),
            LowRankLinear(hidden_dims[1], latent_dim, rank)
        )
        self.decoder = nn.Sequential(
            LowRankLinear(latent_dim, hidden_dims[1], rank), nn.BatchNorm1d(hidden_dims[1]), nn.ReLU(), nn.Dropout(dropout),
            LowRankLinear(hidden_dims[1], hidden_dims[0], rank), nn.BatchNorm1d(hidden_dims[0]), nn.ReLU(), nn.Dropout(dropout),
            LowRankLinear(hidden_dims[0], input_dim, rank)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
    def compute_anomaly_score(self, x):
        with torch.no_grad():
            x_hat, _ = self.forward(x)
            score = F.mse_loss(x_hat, x, reduction='none').mean(1)
            score[torch.isnan(score)] = 0
            return score
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class StandardAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], latent_dim=16, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]), nn.BatchNorm1d(hidden_dims[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]), nn.BatchNorm1d(hidden_dims[1]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]), nn.BatchNorm1d(hidden_dims[1]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[0]), nn.BatchNorm1d(hidden_dims[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
    def compute_anomaly_score(self, x):
        with torch.no_grad():
            x_hat, _ = self.forward(x)
            score = F.mse_loss(x_hat, x, reduction='none').mean(1)
            score[torch.isnan(score)] = 0
            return score
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, input, target):
        loss = (input - target)**2
        pt = torch.exp(-loss)
        focal_weight = self.alpha * (1-pt) ** self.gamma
        return (focal_weight * loss).mean()

def train_autoencoder(model, loader, valloader, nepochs=30, lr=2e-3, wd=2e-4, patience=7, use_focal=True):
    optimz = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = FocalLoss() if use_focal else nn.MSELoss()
    best_loss, wait = float('inf'), 0
    best_state = None
    history = {'train_loss': [], 'val_loss': []}
    for epoch in range(nepochs):
        model.train()
        tr_loss = []
        for Xb, _ in loader:
            Xb = Xb.to(device)
            output, _ = model(Xb)
            loss = criterion(output, Xb)
            if not torch.isfinite(loss):
                continue
            optimz.zero_grad(); loss.backward(); optimz.step()
            tr_loss.append(loss.item())
        model.eval()
        val_loss = []
        with torch.no_grad():
            for Xb, _ in valloader:
                Xb = Xb.to(device)
                output, _ = model(Xb)
                loss = criterion(output, Xb)
                if not torch.isfinite(loss):
                    continue
                val_loss.append(loss.item())
        mean_tr_loss = np.mean(tr_loss) if tr_loss else float('inf')
        mean_val_loss = np.mean(val_loss) if val_loss else float('inf')
        history['train_loss'].append(mean_tr_loss)
        history['val_loss'].append(mean_val_loss)
        if not np.isnan(mean_val_loss) and mean_val_loss < best_loss:
            best_loss = mean_val_loss
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
        print(f"Epoch {epoch+1} train_loss={mean_tr_loss:.5f} val_loss={mean_val_loss:.5f}")
        if wait >= patience: break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("WARNING: Model never improved; returning last state.")
    return model, history

def optimize_threshold(scores, labels):
    fpr, tpr, thr = roc_curve(labels, scores)
    j_idx = np.argmax(2*tpr/(tpr+fpr+1e-8) - fpr)
    return thr[j_idx]

def eval_model(model, loader, thresh=None, optimize_thresh=False, measure_time=True, return_curves=False):
    model.eval()
    all_X, all_y = [], []
    time_start = time.time()
    with torch.no_grad():
        for Xb, yb in loader:
            all_X.append(Xb.cpu().numpy())
            all_y.append(yb.cpu().numpy())
        X = torch.tensor(np.concatenate(all_X)).float().to(device)
        y = np.concatenate(all_y)
        t_inf0 = time.time()
        scores = model.compute_anomaly_score(X).cpu().numpy()
        t_inf1 = time.time()
    if optimize_thresh:
        thresh = optimize_threshold(scores, y)
    elif thresh is None:
        normal_scores = scores[y==0]
        thresh = np.percentile(normal_scores, 92)
    preds = (scores > thresh).astype(int)
    t_end = time.time()
    inference_time = (t_inf1 - t_inf0) if measure_time else 0
    total_time = (t_end - time_start) if measure_time else 0
    fpr, tpr, _ = roc_curve(y, scores)
    precision, recall, _ = precision_recall_curve(y, scores)
    metrics = {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "roc_auc": auc(fpr, tpr),
        "pr_auc": auc(recall, precision),
        "ap": average_precision_score(y, scores),
        "confmat": confusion_matrix(y, preds),
        "thresh": thresh,
        "inference_time": inference_time,
        "total_time": total_time,
        "scores": scores,
        "fpr": fpr,
        "tpr": tpr,
        "pr": precision,
        "rec": recall
    }
    return metrics, scores, y, preds

def mem_MB(model):
    n_params = sum(p.numel() for p in model.parameters())
    return n_params * 4 / 1024 ** 2

def plot_training_curves(histories, model_names):
    plt.figure(figsize=(10,6))
    for history, name in zip(histories, model_names):
        plt.plot(history['train_loss'], label=f"{name} Train")
        plt.plot(history['val_loss'], label=f"{name} Val", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.close()

def plot_confusion_matrices_horizontal(results_df):
    models = ['LowRank', 'Standard']
    n_tests = len(results_df)
    fig, axes = plt.subplots(2, n_tests, figsize=(4*n_tests, 8))
    if n_tests == 1:
        axes = np.expand_dims(axes, axis=1)
    for col, row in enumerate(results_df.itertuples()):
        for row_idx, model in enumerate(models):
            cm = np.array(eval(getattr(row, f'{model.lower()}_confmat')))
            ax = axes[row_idx, col]
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
            ax.set_title(f"{model} - {row.testset}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrices_horizontal.png")
    plt.close()

def plot_score_distributions(results_df):
    all_data = []
    for _, row in results_df.iterrows():
        for model in ['LowRank', 'Standard']:
            scores = np.array(eval(row.get(f'{model.lower()}_scores','[]')))
            for s in scores:
                all_data.append({"Testset": row['testset'], "Model": model, "Score": s})
    df = pd.DataFrame(all_data)
    g = sns.FacetGrid(df, row="Testset", col="Model", sharex=True, sharey=False, height=2.8, aspect=2)
    g.map(sns.kdeplot, "Score", fill=True, bw_adjust=0.7, alpha=0.4, linewidth=2)
    g.set_titles(row_template='{row_name}', col_template='{col_name}')
    g.set_axis_labels("Anomaly Score", "Density")
    plt.subplots_adjust(top=0.92)
    g.fig.suptitle("Anomaly Score Distributions (per Testset/Model)", fontsize=14)
    g.savefig("score_distributions_facet.png")
    plt.close()

def plot_score_boxplots(results_df):
    all_data = []
    for _, row in results_df.iterrows():
        for model in ['LowRank', 'Standard']:
            scores = np.array(eval(row.get(f'{model.lower()}_scores','[]')))
            for s in scores:
                all_data.append({"Testset": row['testset'], "Model": model, "Score": s})
    df = pd.DataFrame(all_data)
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x="Testset", y="Score", hue="Model", showfliers=False, notch=True)
    plt.title("Anomaly Score Boxplot by Testset and Model")
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("score_boxplots.png")
    plt.close()

def plot_threshold_histogram(results_df):
    data = []
    for _, row in results_df.iterrows():
        for model in ['LowRank', 'Standard']:
            data.append({
                "Testset": row['testset'],
                "Model": model,
                "Threshold": float(row.get(f'{model.lower()}_thresh', np.nan))
            })
    df = pd.DataFrame(data)
    plt.figure(figsize=(8,4))
    sns.barplot(data=df, x="Testset", y="Threshold", hue="Model")
    plt.title("Chosen Anomaly Thresholds per Testset/Model")
    plt.tight_layout()
    plt.savefig("threshold_histogram.png")
    plt.close()

def plot_inference_time_bar(results_df):
    data = []
    for _, row in results_df.iterrows():
        for model in ['LowRank', 'Standard']:
            data.append({
                "Testset": row['testset'],
                "Model": model,
                "Inference time (s)": float(row.get(f'{model.lower()}_inftime', np.nan))
            })
    df = pd.DataFrame(data)
    plt.figure(figsize=(9,4))
    sns.barplot(data=df, x="Testset", y="Inference time (s)", hue="Model")
    plt.title("Inference Time per Testset/Model")
    plt.tight_layout()
    plt.savefig("inference_time_bar.png")
    plt.close()

def plot_f1_vs_threshold(results_df):
    data = []
    for _, row in results_df.iterrows():
        for model in ['LowRank', 'Standard']:
            data.append({
                "Testset": row['testset'],
                "Model": model,
                "F1": float(row.get(f'{model.lower()}_f1', np.nan)),
                "Threshold": float(row.get(f'{model.lower()}_thresh', np.nan))
            })
    df = pd.DataFrame(data)
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df, x="Threshold", y="F1", hue="Model", style="Testset", s=100)
    plt.title("F1 Score vs. Threshold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("f1_vs_threshold.png")
    plt.close()

def plot_precision_recall_bar(results_df):
    data = []
    for _, row in results_df.iterrows():
        for model in ['LowRank', 'Standard']:
            data.append({
                "Testset": row['testset'],
                "Model": model,
                "Precision": float(row.get(f'{model.lower()}_precision', np.nan)),
                "Recall": float(row.get(f'{model.lower()}_recall', np.nan))
            })
    df = pd.DataFrame(data)
    df = df.melt(id_vars=["Testset", "Model"], value_vars=["Precision", "Recall"], var_name="Metric", value_name="Value")
    plt.figure(figsize=(10,5))
    sns.barplot(data=df, x="Testset", y="Value", hue="Model", palette="Set2", ci=None, dodge=True, hue_order=["LowRank","Standard"], edgecolor="gray", linewidth=1.2)
    plt.title("Precision and Recall by Testset and Model")
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig("precision_recall_bar.png")
    plt.close()

def plot_f1_bar(results_df):
    data = []
    for _, row in results_df.iterrows():
        for model in ['LowRank', 'Standard']:
            data.append({
                "Testset": row['testset'],
                "Model": model,
                "F1": float(row.get(f'{model.lower()}_f1', np.nan))
            })
    df = pd.DataFrame(data)
    plt.figure(figsize=(8,5))
    sns.barplot(data=df, x="Testset", y="F1", hue="Model")
    plt.title("F1 Score per Testset and Model")
    plt.tight_layout()
    plt.savefig("f1_bar.png")
    plt.close()

def plot_ap_bar(results_df):
    data = []
    for _, row in results_df.iterrows():
        for model in ['LowRank', 'Standard']:
            data.append({
                "Testset": row['testset'],
                "Model": model,
                "AP": float(row.get(f'{model.lower()}_ap', np.nan))
            })
    df = pd.DataFrame(data)
    plt.figure(figsize=(8,5))
    sns.barplot(data=df, x="Testset", y="AP", hue="Model")
    plt.title("Average Precision per Testset and Model")
    plt.tight_layout()
    plt.savefig("ap_bar.png")
    plt.close()

def plot_pr_roc_curves(results_df):
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    for _, row in results_df.iterrows():
        for model in ['LowRank', 'Standard']:
            pr_col = f'{model.lower()}_pr'
            rec_col = f'{model.lower()}_rec'
            if pr_col in row and rec_col in row:
                try:
                    precision = np.array(eval(row[pr_col]))
                    recall = np.array(eval(row[rec_col]))
                    if len(precision) and len(recall):
                        plt.plot(recall, precision, label=f"{row['testset']} ({model})")
                except:
                    continue
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.subplot(1,2,2)
    for _, row in results_df.iterrows():
        for model in ['LowRank', 'Standard']:
            fpr_col = f'{model.lower()}_fpr'
            tpr_col = f'{model.lower()}_tpr'
            if fpr_col in row and tpr_col in row:
                try:
                    fpr = np.array(eval(row[fpr_col]))
                    tpr = np.array(eval(row[tpr_col]))
                    if len(fpr) and len(tpr):
                        plt.plot(fpr, tpr, label=f"{row['testset']} ({model})")
                except:
                    continue
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pr_roc_curves.png")
    plt.close()

def plot_performance_table(results_df, output_dir='.'):
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    records = []
    for _, row in results_df.iterrows():
        for model in ['LowRank', 'Standard']:
            rec = {'Testset': row['testset'], 'Model': model}
            for metric in metrics:
                rec[metric] = row[f"{model.lower()}_{metric}"]
            records.append(rec)
    df_perf = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(12, 1.5+len(df_perf)*0.5))
    ax.axis('off')
    tbl = ax.table(cellText=df_perf.round(3).values,
                   colLabels=df_perf.columns, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width(col=list(range(len(df_perf.columns))))
    plt.title("Performance Summary Table")
    plt.savefig("performance_table.png")
    plt.close()

def comprehensive_visualization(
    history_lowrank, history_standard,
    results_csv="results_summary.csv"
):
    results_df = pd.read_csv(results_csv)
    plot_training_curves(
        histories=[history_lowrank, history_standard],
        model_names=["LowRankAE", "StandardAE"]
    )
    plot_confusion_matrices_horizontal(results_df)
    plot_score_distributions(results_df)
    plot_score_boxplots(results_df)
    plot_threshold_histogram(results_df)
    plot_inference_time_bar(results_df)
    plot_f1_vs_threshold(results_df)
    plot_precision_recall_bar(results_df)
    plot_f1_bar(results_df)
    plot_ap_bar(results_df)
    if all(col in results_df.columns for col in ['lowrank_pr','lowrank_rec','lowrank_fpr','lowrank_tpr',
                                                 'standard_pr','standard_rec','standard_fpr','standard_tpr']):
        plot_pr_roc_curves(results_df)
    plot_performance_table(results_df)

def main():
    syncan_path = None
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            filepath = os.path.join(dirname, filename)
            if 'SynCAN-master' in filepath and filepath.endswith('.csv'):
                syncan_path = filepath.split('SynCAN-master')[0] + 'SynCAN-master'
                break
        if syncan_path:
            break
    if not syncan_path:
        print("Could not find SynCAN dataset!")
        return None
    print(f"Found SynCAN data at: {syncan_path}")

    train_files = [os.path.join(syncan_path, d, f"{d}.csv") for d in ["train_1"]]
    test_files = [os.path.join(syncan_path, d, f"{d}.csv") for d in [
        "test_normal","test_plateau","test_continuous","test_playback","test_suppress","test_flooding"]]
    train_df = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
    test_dfs = {os.path.basename(f).split('.')[0]: pd.read_csv(f) for f in test_files}

    print("Extracting features ...")
    X_train, y_train = extract_sequences(train_df, seq_len=64, stride=8, max_seqs=6000)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_tr, X_val, y_tr, y_val = balance_and_split(X_train, y_train, test_size=0.15)
    train_ds = CANSeqDataset(X_tr, y_tr)
    val_ds = CANSeqDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    print("PARAM COUNT AND MEMORY (MB):")
    input_dim = X_train.shape[1]
    lowrank_model = LowRankAutoencoder(input_dim=input_dim, hidden_dims=[128,64], latent_dim=16, rank_factor=8, dropout=0.18).to(device)
    std_model = StandardAutoencoder(input_dim=input_dim, hidden_dims=[128,64], latent_dim=16, dropout=0.18).to(device)
    print(f"LowRank: params={lowrank_model.count_params():,} mem={mem_MB(lowrank_model):.2f}MB")
    print(f"Standard: params={std_model.count_params():,} mem={mem_MB(std_model):.2f}MB")

    print("Training Low-Rank Autoencoder ...")
    t0 = time.time()
    lowrank_model, history_lowrank = train_autoencoder(lowrank_model, train_loader, val_loader, nepochs=32, lr=2e-3, wd=2e-4, patience=8, use_focal=True)
    t1 = time.time()
    print("Training Standard Autoencoder ...")
    std_model, history_standard = train_autoencoder(std_model, train_loader, val_loader, nepochs=32, lr=2e-3, wd=2e-4, patience=8, use_focal=True)
    t2 = time.time()

    print(f"LowRankAE train time: {t1-t0:.2f}s, StandardAE: {t2-t1:.2f}s")

    results = []
    for k, df in test_dfs.items():
        print(f"\nEvaluating on {k} ...")
        X_test, y_test = extract_sequences(df, seq_len=64, stride=8, max_seqs=2500, augment=False)
        X_test = scaler.transform(X_test)
        test_ds = CANSeqDataset(X_test, y_test)
        test_loader = DataLoader(test_ds, batch_size=256)

        lowrank_metrics, lowrank_scores, y_true, _ = eval_model(
            lowrank_model, test_loader, optimize_thresh=True, measure_time=True, return_curves=True)
        std_metrics, std_scores, _, _ = eval_model(
            std_model, test_loader, optimize_thresh=True, measure_time=True, return_curves=True)

        print(f"LowRankAE: acc={lowrank_metrics['accuracy']:.3f}  prec={lowrank_metrics['precision']:.3f}  rec={lowrank_metrics['recall']:.3f}  f1={lowrank_metrics['f1']:.3f}  roc_auc={lowrank_metrics['roc_auc']:.3f}  pr_auc={lowrank_metrics['pr_auc']:.3f}  AP={lowrank_metrics['ap']:.3f} inference_time={lowrank_metrics['inference_time']:.3f}s total_time={lowrank_metrics['total_time']:.3f}s")
        print(f"StandardAE: acc={std_metrics['accuracy']:.3f}  prec={std_metrics['precision']:.3f}  rec={std_metrics['recall']:.3f}  f1={std_metrics['f1']:.3f}  roc_auc={std_metrics['roc_auc']:.3f}  pr_auc={std_metrics['pr_auc']:.3f}  AP={std_metrics['ap']:.3f} inference_time={std_metrics['inference_time']:.3f}s total_time={std_metrics['total_time']:.3f}s")
        results.append({
            "testset": k,
            "lowrank_accuracy": lowrank_metrics['accuracy'],
            "lowrank_precision": lowrank_metrics['precision'],
            "lowrank_recall": lowrank_metrics['recall'],
            "lowrank_f1": lowrank_metrics['f1'],
            "lowrank_roc_auc": lowrank_metrics['roc_auc'],
            "lowrank_pr_auc": lowrank_metrics['pr_auc'],
            "lowrank_ap": lowrank_metrics['ap'],
            "lowrank_confmat": repr(lowrank_metrics['confmat'].tolist()),
            "lowrank_scores": repr(lowrank_metrics['scores'].tolist()),
            "lowrank_fpr": repr(lowrank_metrics['fpr'].tolist()),
            "lowrank_tpr": repr(lowrank_metrics['tpr'].tolist()),
            "lowrank_pr": repr(lowrank_metrics['pr'].tolist()),
            "lowrank_rec": repr(lowrank_metrics['rec'].tolist()),
            "lowrank_inftime": lowrank_metrics['inference_time'],
            "lowrank_totaltime": lowrank_metrics['total_time'],
            "standard_accuracy": std_metrics['accuracy'],
            "standard_precision": std_metrics['precision'],
            "standard_recall": std_metrics['recall'],
            "standard_f1": std_metrics['f1'],
            "standard_roc_auc": std_metrics['roc_auc'],
            "standard_pr_auc": std_metrics['pr_auc'],
            "standard_ap": std_metrics['ap'],
            "standard_confmat": repr(std_metrics['confmat'].tolist()),
            "standard_scores": repr(std_metrics['scores'].tolist()),
            "standard_fpr": repr(std_metrics['fpr'].tolist()),
            "standard_tpr": repr(std_metrics['tpr'].tolist()),
            "standard_pr": repr(std_metrics['pr'].tolist()),
            "standard_rec": repr(std_metrics['rec'].tolist()),
            "standard_inftime": std_metrics['inference_time'],
            "standard_totaltime": std_metrics['total_time'],
        })
    df_results = pd.DataFrame(results)
    df_results.to_csv("results_summary.csv", index=False)

    # Save histories for visualization
    pd.DataFrame(history_lowrank).to_csv("history_lowrank.csv", index=False)
    pd.DataFrame(history_standard).to_csv("history_standard.csv", index=False)

    # ---- VISUALIZATION ----
    comprehensive_visualization(
        history_lowrank, history_standard,
        results_csv="results_summary.csv"
    )

if __name__ == "__main__":
    main()
