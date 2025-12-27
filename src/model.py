import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from pytorch_lightning import LightningModule
import torch.nn as nn
import torchmetrics
from torch.optim import Adam
from pytorch_lightning import Trainer
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

trainer = Trainer(
    max_epochs=20,
    accelerator="auto",
    devices="auto",
    log_every_n_steps=10
)

#create the dataset class
class YTDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class MLP(LightningModule): 
    pass

#cambiare prima di pushare il path
data = pd.read_csv('C:/Users/dernj/Desktop/views_estimation/data/youtube_shorts_performance_dataset.csv')
y = np.log1p(data['views'].to_numpy())
X = data.drop('views', axis = 1)

ohe = OneHotEncoder(sparse_output = False)
X = ohe.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.2)
data_train = YTDataset(X_train, y_train)
data_test = YTDataset(X_test, y_test)

train_loader = DataLoader(
    data_train,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    data_test,
    batch_size=32,
    shuffle=False
)

print(X_train.shape)

class MLPRegressor(LightningModule):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1555, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            )


        # Loss consigliata per views (più robusta della MSE)
        self.criterion = nn.SmoothL1Loss(beta=1.0)

        # Metriche
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae   = torchmetrics.MeanAbsoluteError()
        self.val_mse   = torchmetrics.MeanSquaredError()

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)

        loss = self.criterion(preds, y)

        self.train_mae.update(preds, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mae", self.train_mae, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)

        mae = self.val_mae(preds, y)
        mse = self.val_mse(preds, y)

        self.log("val_mae", mae, prog_bar=True)
        self.log("val_mse", mse)
        return mae

    def configure_optimizers(self):
        return Adam(
            self.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )

model = MLPRegressor()

trainer.fit(
    model,
    train_dataloaders=train_loader,
    val_dataloaders=test_loader,
)

@torch.no_grad()
def evaluate_model(model, dataloader, device="cpu"):

    model = model.to(device)
    model.eval()

    y_true_log = []
    y_pred_log = []

    y_true_real = []
    y_pred_real = []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        out = model(x)
        y_pred_log.append(out.cpu().numpy())
        y_true_log.append(y.cpu().numpy())

        # convert back to real space
        pred_real = torch.expm1(out)
        true_real = torch.expm1(y)

        y_pred_real.append(pred_real.cpu().numpy())
        y_true_real.append(true_real.cpu().numpy())

    y_true_log  = np.concatenate(y_true_log)
    y_pred_log  = np.concatenate(y_pred_log)

    y_true_real = np.concatenate(y_true_real)
    y_pred_real = np.concatenate(y_pred_real)

    # ---- Metrics in log space ----
    log_mae = mean_absolute_error(y_true_log, y_pred_log)
    log_rmse = np.sqrt(mean_squared_error(y_true_log, y_pred_log))

    # ---- Metrics in real (views) space ----
    mae = mean_absolute_error(y_true_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
    r2 = r2_score(y_true_real, y_pred_real)

    mape = np.mean(np.abs((y_true_real - y_pred_real) / np.maximum(y_true_real, 1))) * 100

    # ---- Prediction table ----
    df_pred = pd.DataFrame({
        "views_true": y_true_real,
        "views_pred": y_pred_real,
        "views_error": y_pred_real - y_true_real
    })

    metrics = {
        "log_mae": log_mae,
        "log_rmse": log_rmse,
        "mae_real": mae,
        "rmse_real": rmse,
        "r2_real": r2,
        "mape_real_%": mape
    }

    return metrics, df_pred

metrics, df_pred = evaluate_model(model, test_loader)

print("=== Metrics (log space) ===")
print(f"MAE_log  = {metrics['log_mae']:.4f}")
print(f"RMSE_log = {metrics['log_rmse']:.4f}")

print("\n=== Metrics (real views space) ===")
print(f"MAE_real  = {metrics['mae_real']:.2f}")
print(f"RMSE_real = {metrics['rmse_real']:.2f}")
print(f"R2_real   = {metrics['r2_real']:.4f}")
print(f"MAPE_real = {metrics['mape_real_%']:.2f}%")

#le metriche del modello suggeriscono la natura complicata del target, sul log space funziona bene
#perchè riduce gli outliers, ma sullo spazio originale la rete non predice bene il target (in particolare è peggiore
#di un modello baseline che predice sempre la media).

#possiamo adottare due strategie: mantenere il modello e le predizioni sul log space o fare classificazione su un bucket
#di views su scala logaritmica