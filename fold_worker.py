#!/usr/bin/env python3
"""
fold_worker.py — Exécute un fold du k-fold sur un GPU assigné.

Lancé par le notebook 02_approche_cnn_wbc.ipynb en sous-processus.
Chaque fold tourne dans un process séparé avec CUDA_VISIBLE_DEVICES
défini par le notebook parent.

Usage:
    python fold_worker.py --fold 0 --gpu 0 --config outputs/kfold_config.pkl
"""
import argparse
import os
import time
import copy
import pickle
import gc
import warnings

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import f1_score
import xgboost as xgb

warnings.filterwarnings("ignore")


# ─── Dataset ──────────────────────────────────────────────────────────────────

class WBCDataset(Dataset):
    def __init__(self, dataframe, transform, class_to_idx, col_label="label", has_labels=True):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.col_label = col_label
        self.has_labels = has_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.has_labels:
            return img, self.class_to_idx[row[self.col_label]]
        return img, -1


# ─── Backbone ─────────────────────────────────────────────────────────────────

class ConvNeXtTinyBackbone(nn.Module):
    def __init__(self, num_classes, dropout_p=0.3):
        super().__init__()
        backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.embed_dim = 768
        self.norm = nn.LayerNorm(self.embed_dim)
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(self.embed_dim, num_classes)
        )

    def get_embeddings(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.norm(x)

    def forward(self, x):
        return self.fc(self.get_embeddings(x))


# ─── Training functions ──────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return running_loss / len(loader.dataset), f1_score(all_labels, all_preds, average="macro")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return (running_loss / len(loader.dataset),
            f1_score(all_labels, all_preds, average="macro"),
            np.array(all_preds), np.array(all_labels))


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    all_emb, all_labels = [], []
    for imgs, labels in loader:
        emb = model.get_embeddings(imgs.to(device))
        all_emb.append(emb.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return np.vstack(all_emb), np.array(all_labels)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run one k-fold on a specific GPU")
    parser.add_argument("--fold", type=int, required=True, help="Fold index (0-based)")
    parser.add_argument("--gpu", type=int, required=True, help="GPU index (for logging)")
    parser.add_argument("--config", type=str, required=True, help="Path to config pickle")
    args = parser.parse_args()

    # CUDA_VISIBLE_DEVICES est déjà défini par le process parent via subprocess env.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load config ──
    with open(args.config, "rb") as f:
        cfg = pickle.load(f)

    fold_idx = args.fold
    k_folds = cfg["k_folds"]
    train_idx = cfg["splits"][fold_idx]["train_idx"]
    val_idx = cfg["splits"][fold_idx]["val_idx"]
    col_label = cfg["col_label"]

    # ── Seeds (unique par fold pour diversité) ──
    seed = cfg["seed"] + fold_idx
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False

    print(f"[Fold {fold_idx+1}/{k_folds}] GPU {args.gpu} → device={device} | seed={seed}")
    t_fold = time.time()

    # ── Transforms ──
    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    img_size = cfg["img_size"]
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    # ── Data ──
    df_train = cfg["df_train"]
    df_test = cfg["df_test"]
    c2i = cfg["class_to_idx"]
    nc = cfg["num_classes"]

    df_tr = df_train.iloc[train_idx].reset_index(drop=True)
    df_vl = df_train.iloc[val_idx].reset_index(drop=True)
    print(f"  Train: {len(df_tr)} | Val: {len(df_vl)}")

    # ── Datasets & Loaders ──
    bs = cfg["batch_size"]
    nw = cfg["num_workers"]
    train_ds = WBCDataset(df_tr, train_tf, c2i, col_label=col_label, has_labels=True)
    val_ds = WBCDataset(df_vl, val_tf, c2i, col_label=col_label, has_labels=True)

    # WeightedRandomSampler
    tr_labels = df_tr[col_label].values
    cc = pd.Series(tr_labels).value_counts()
    sw = np.array([1.0 / cc[lbl] for lbl in tr_labels])
    sampler = WeightedRandomSampler(torch.from_numpy(sw).double(), len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,
                              num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=True)

    # ── 1. Fine-tune backbone ──
    model = ConvNeXtTinyBackbone(nc, dropout_p=cfg["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    best_f1, best_wts, patience_ctr = 0.0, None, 0
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    patience = cfg["patience"]
    num_epochs = cfg["num_epochs"]

    for epoch in range(num_epochs):
        tr_loss, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_f1, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(vl_f1)

        imp = "✓" if vl_f1 > best_f1 else ""
        print(f"  Epoch {epoch+1:2d}/{num_epochs}  "
              f"tr_loss={tr_loss:.4f}  vl_loss={vl_loss:.4f}  "
              f"tr_F1={tr_f1:.4f}  vl_F1={vl_f1:.4f}  {imp}")

        if vl_f1 > best_f1:
            best_f1 = vl_f1
            best_wts = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping epoch {epoch+1}")
                break

    model.load_state_dict(best_wts)
    print(f"  Best backbone val_F1 = {best_f1:.4f}")

    # ── Save backbone checkpoint ──
    fold_dir = os.path.join(cfg["output_dir"], f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    torch.save(best_wts, os.path.join(fold_dir, "backbone_best.pth"))

    # ── 2. Extraction embeddings ──
    emb_tr_ds = WBCDataset(df_tr, val_tf, c2i, col_label=col_label, has_labels=True)
    emb_vl_ds = WBCDataset(df_vl, val_tf, c2i, col_label=col_label, has_labels=True)
    emb_tr_loader = DataLoader(emb_tr_ds, batch_size=bs, shuffle=False,
                               num_workers=nw, pin_memory=True)
    emb_vl_loader = DataLoader(emb_vl_ds, batch_size=bs, shuffle=False,
                               num_workers=nw, pin_memory=True)

    X_train, y_train = extract_embeddings(model, emb_tr_loader, device)
    X_val, y_val = extract_embeddings(model, emb_vl_loader, device)
    print(f"  Embeddings: train {X_train.shape}, val {X_val.shape}")

    # ── 3. XGBoost (CPU) ──
    xgb_params = cfg["xgb_params"].copy()
    xgb_clf = xgb.XGBClassifier(**xgb_params)
    xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_val_preds = xgb_clf.predict(X_val)
    xgb_val_f1 = f1_score(y_val, xgb_val_preds, average="macro")
    print(f"  XGBoost val F1 = {xgb_val_f1:.4f} ({xgb_clf.best_iteration} trees)")

    # ── 4. Test predictions ──
    test_ds = WBCDataset(df_test, val_tf, c2i, col_label=col_label, has_labels=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=nw, pin_memory=True)
    X_test, _ = extract_embeddings(model, test_loader, device)
    test_proba = xgb_clf.predict_proba(X_test)
    print(f"  Test probas: {test_proba.shape}")

    # ── Cleanup GPU ──
    del model, optimizer, criterion
    torch.cuda.empty_cache()
    gc.collect()

    elapsed = time.time() - t_fold
    print(f"  Fold {fold_idx+1} terminé en {elapsed/60:.1f} min")

    # ── Save results ──
    results = {
        "fold": fold_idx + 1,
        "gpu": args.gpu,
        "backbone_val_f1": best_f1,
        "xgb_val_f1": xgb_val_f1,
        "xgb_best_iter": xgb_clf.best_iteration,
        "val_preds": xgb_val_preds,
        "val_labels": y_val,
        "val_idx": val_idx,
        "test_proba": test_proba,
        "history": history,
        "elapsed_min": elapsed / 60,
    }

    results_path = os.path.join(fold_dir, "results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"  Résultats sauvegardés: {results_path}")


if __name__ == "__main__":
    main()
