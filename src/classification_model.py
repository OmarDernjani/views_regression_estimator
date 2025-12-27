import pandas as pd
import numpy as np
from pytorch_lightning import LightningModule
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch.optim import Adam
import torchmetrics
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix


###############

@torch.no_grad()
def compute_confusion_matrix(model, dataloader):
    model.eval()

    all_preds = []
    all_targets = []

    for x, y in dataloader:
        logits = model(x)
        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.numpy())
        all_targets.append(y.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    return confusion_matrix(all_targets, all_preds)

################



data = pd.read_csv('C:/Users/dernj/Desktop/views_estimation/data/youtube_shorts_performance_dataset.csv')
views = data["views"].to_numpy()

X = data.drop('views', axis = 1)
ohe = OneHotEncoder(sparse_output = False)
X = ohe.fit_transform(X)

print("min:", views.min())
print("max:", views.max())
print("median:", np.median(views))
print("mean:", views.mean())


y_bucket, bins = pd.qcut(views, q=4, labels=False, retbins=True)
print(bins)
bucket_labels = [
    "low (1k–129k)",
    "mid-low (129k–255k)",
    "mid-high (255k–356k)",
    "high (356k–499k)"
]


class YTDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype = torch.float32)
        self.y = torch.tensor(y, dtype = torch.long)
    
    def __len__(self):
        return len(self.x)
    

    def __getitem__(self, idx):
        return self.x[idx],self.y[idx]

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y_bucket,
    stratify = y_bucket, 
    random_state = 42,
    test_size = 0.2)

train_dataset = YTDataset(X_train, y_train)
test_dataset = YTDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle = True)

print(X.shape)


class MLPClassification_easy(LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features = 1555, out_features = 200),
            nn.ReLU(),
            nn.Dropout(p = 0.6),
            nn.Linear(in_features = 200, out_features = 50),
            nn.ReLU(),
            nn.Dropout(p = 0.6),
            nn.Linear(in_features = 50, out_features = num_classes),
        )
        self.loss_fn = nn.CrossEntropyLoss()

        # accuracy metric
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=1555
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=1555
        )

    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar = True)
        return loss

    # ---------- VALIDATION ----------
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return {"preds": preds, "targets": y}

    # ---------- OPTIMIZER ----------
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)


class MLPClassification_hard(LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features = 1555, out_features = 800),
            nn.ReLU(),
            nn.Dropout(p = 0.6),
            nn.Linear(in_features = 800, out_features = 400),
            nn.ReLU(),
            nn.Dropout(p = 0.6),
            nn.Linear(in_features = 400, out_features = 200),
            nn.ReLU(),
            nn.Dropout(p = 0.6),
            nn.Linear(in_features = 200, out_features = 100),
            nn.ReLU(),
            nn.Dropout(p = 0.6),
            nn.Linear(in_features = 100, out_features = 50),
            nn.ReLU(),
            nn.Dropout(p = 0.6),
            nn.Linear(in_features = 50, out_features = num_classes),
        )
        self.loss_fn = nn.CrossEntropyLoss()

        # accuracy metric
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=1555
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=1555
        )


    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar = True)
        return loss

    # ---------- VALIDATION ----------
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return {"preds": preds, "targets": y}

    # ---------- OPTIMIZER ----------
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

model = MLPClassification_easy(num_classes = 5) #where num_classes is the number of butkets.

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="min"
)

trainer = Trainer(
    max_epochs=50,
    callbacks=[early_stop],
    log_every_n_steps=10
)

trainer.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=test_dataloader
)

cm_5 = compute_confusion_matrix(model, test_dataloader)
print(cm_5)


model = MLPClassification_hard(num_classes = 5)
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="min"
)

trainer = Trainer(
    max_epochs=50,
    callbacks=[early_stop],
    log_every_n_steps=10
)
trainer.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=test_dataloader
)

cm_5 = compute_confusion_matrix(model, test_dataloader)
print(cm_5)

#the model seems to overfit in the train set, we've used two different net, one simple net and the other more complicated, the result is the same,
#the problem could be with the bucket, they're too similar, let's try one more strategy.

# === CREATE 3 WIDE BUCKETS ===
y_bucket_3, bins_3 = pd.qcut(
    views,
    q=3,
    labels=False,
    retbins=True
)

print("3-bucket edges:", bins_3)

bucket_labels_3 = [
    "low",
    "mid",
    "high"
]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_bucket_3,
    stratify=y_bucket_3,
    random_state=42,
    test_size=0.2
)
train_dataset = YTDataset(X_train, y_train)
test_dataset  = YTDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader  = DataLoader(test_dataset, batch_size=32, shuffle=False)
model = MLPClassification_easy(num_classes = 3)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="min"
)

trainer = Trainer(
    max_epochs=50,
    callbacks=[early_stop],
    log_every_n_steps=10
)

trainer.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=test_dataloader
)

cm_5 = compute_confusion_matrix(model, test_dataloader)
print(cm_5)


model = MLPClassification_hard(num_classes = 3)
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="min"
)

trainer = Trainer(
    max_epochs=50,
    callbacks=[early_stop],
    log_every_n_steps=10
)
trainer.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=test_dataloader
)

cm_5 = compute_confusion_matrix(model, test_dataloader)
print(cm_5)

#from the last confusion_matrix we get that the high bucket isn't separable.

y_bucket_2, bins_2 = pd.qcut(
    views,
    q=2,
    labels=False,
    retbins=True
)

print("2-bucket edges:", bins_2)

bucket_labels_2 = [
    "low",
    "high"
]

num_classes = 2
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_bucket_2,
    stratify=y_bucket_2,
    random_state=42,
    test_size=0.2
)
train_dataset = YTDataset(X_train, y_train)
test_dataset  = YTDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader  = DataLoader(test_dataset, batch_size=32, shuffle=False)
model = MLPClassification_easy(num_classes = 2)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="min"
)

trainer = Trainer(
    max_epochs=50,
    callbacks=[early_stop],
    log_every_n_steps=10
)

trainer.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=test_dataloader
)

cm = compute_confusion_matrix(model, test_dataloader)
print("\nConfusion matrix (LOW vs HIGH):\n", cm)

model = MLPClassification_hard(num_classes = 2)

trainer.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=test_dataloader
)

cm = compute_confusion_matrix(model, test_dataloader)
print("\nConfusion matrix (LOW vs HIGH):\n", cm)
