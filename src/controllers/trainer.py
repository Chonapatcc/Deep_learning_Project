"""
Controller: Training Logic
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import time


class Trainer:
    """Handle model training"""
    
    def __init__(self, model, device='cuda', learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', 
                                          factor=0.5, patience=5)
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc='Training')
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_loss = train_loss / train_total
        train_acc = 100. * train_correct / train_total
        
        return train_loss, train_acc
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc='Validation')
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_loss = val_loss / val_total
        val_acc = 100. * val_correct / val_total
        
        return val_loss, val_acc
    
    def train(self, train_loader, val_loader, num_epochs=50, 
              save_path='best_model.pth', patience=10):
        """
        Train model with early stopping
        
        Returns:
            history: Training history dictionary
        """
        best_val_acc = 0.0
        patience_counter = 0
        
        print("\n" + "="*60)
        print("TRAINING START")
        print("="*60)
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            epoch_time = time.time() - start_time
            
            # Print summary
            print(f'\nEpoch {epoch+1}/{num_epochs} - {epoch_time:.2f}s')
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {current_lr:.6f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, save_path)
                print(f'✓ Best model saved! (Val Acc: {val_acc:.2f}%)')
                patience_counter = 0
            else:
                patience_counter += 1
                print(f'No improvement. Patience: {patience_counter}/{patience}')
            
            # Early stopping
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                break
            
            print('-' * 60)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print("="*60)
        
        return self.history
