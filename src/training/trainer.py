import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import config
from src.training.loss import NeuromorphicLoss

class TrainEngine:
    def __init__(self, model, train_loader, val_loader, lr=config.LEARNING_RATE):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = NeuromorphicLoss()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in self.train_loader:
            batch_x, batch_y = batch_x.to(config.DEVICE), batch_y.to(config.DEVICE)
            
            self.optimizer.zero_grad()
            logits, spike_record = self.model(batch_x)
            
            loss = self.criterion(logits, batch_y, spike_record, self.model)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        epoch_loss = total_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x, batch_y = batch_x.to(config.DEVICE), batch_y.to(config.DEVICE)
                logits, spike_record = self.model(batch_x)
                
                loss = self.criterion(logits, batch_y, spike_record, self.model)
                total_loss += loss.item() * batch_x.size(0)
                
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        val_loss = total_loss / total
        val_acc = correct / total
        return val_loss, val_acc

    def run(self, epochs=config.EPOCHS):
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(1, epochs + 1):
            t_loss, t_acc = self.train_epoch(epoch)
            v_loss, v_acc = self.validate()

            history['train_loss'].append(t_loss)
            history['train_acc'].append(t_acc)
            history['val_loss'].append(v_loss)
            history['val_acc'].append(v_acc)

            print(f"Epoch [{epoch:02d}/{epochs:02d}] - "
                  f"Train Loss: {t_loss:.4f} | Train Acc: {t_acc*100:.2f}% | "
                  f"Val Loss: {v_loss:.4f} | Val Acc: {v_acc*100:.2f}%")

        return history