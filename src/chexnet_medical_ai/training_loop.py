"""training loop"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import time
import sys
from chexnet_medical_ai.model_layers import DenseNet121


model = DenseNet121(num_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, device):
    scaler = GradScaler()
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # mixed precision
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # ---- simple loading bar ----
            bar_len = 30
            filled_len = int(bar_len * batch_idx / len(train_loader))
            bar = "=" * filled_len + "-" * (bar_len - filled_len)
            sys.stdout.write(
                f"\rEpoch [{epoch+1}/{epochs}] |{bar}| "
                f"Step {batch_idx}/{len(train_loader)} "
                f"Loss: {loss.item():.4f}"
            )
            sys.stdout.flush()

        train_acc = 100 * correct / total

        # validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        elapsed = time.time() - start_time

        # print epoch summary
        print(f"\n✅ Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}% "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}% "
              f"({elapsed:.1f} sec)")
