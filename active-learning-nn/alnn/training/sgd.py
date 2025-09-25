from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange

def make_loaders(X_tr, y_tr, X_val, y_val, batch_size: int):
    ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    ds_val = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    return DataLoader(ds_tr, batch_size=batch_size, shuffle=True), DataLoader(ds_val, batch_size=1024, shuffle=False)

def get_loss(task: str):
    return nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()

def evaluate(model, loader, task: str):
    model.eval()
    criterion = get_loss(task)
    loss_sum, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.float())
            if task == "classification":
                loss = criterion(out, yb.long())
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                n += len(yb)
            else:
                yb = yb.view(-1,1).float()
                loss = criterion(out, yb)
                n += len(yb)
            loss_sum += loss.item() * len(xb)
    return {"loss": loss_sum / n, **({"acc": correct / n} if task=="classification" else {})}

def train_sgd(model, X_tr, y_tr, X_val, y_val, task: str, lr=1e-2, weight_decay=1e-4, batch_size=32, epochs=200, patience=20, device="cpu"):
    train_loader, val_loader = make_loaders(X_tr, y_tr, X_val, y_val, batch_size)
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    criterion = get_loss(task)

    best, best_state, wait = float("inf"), None, 0
    for _ in trange(epochs, desc="SGD", leave=False):
        model.train()
        for xb, yb in train_loader:
            xb = xb.float().to(device)
            if task == "classification":
                yb = yb.long().to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
            else:
                yb = yb.view(-1,1).float().to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
            opt.step()
            opt.zero_grad()

        val_metrics = evaluate(model, val_loader, task)
        cur = val_metrics["loss"]
        if cur < best - 1e-6:
            best, best_state, wait = cur, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, []
