"""
Controller: Model Evaluation
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix


class Evaluator:
    """Handle model evaluation"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
    
    def evaluate(self, test_loader):
        """
        Evaluate model on test set
        
        Returns:
            accuracy, loss, predictions, true_labels
        """
        self.model.eval()
        
        test_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Evaluating'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_loss = test_loss / len(test_loader.dataset)
        accuracy = 100. * np.sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
        
        return accuracy, test_loss, np.array(all_predictions), np.array(all_labels)
    
    def print_classification_report(self, true_labels, predictions, class_names):
        """Print detailed classification report"""
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        
        report = classification_report(true_labels, predictions,
                                       target_names=class_names, digits=4)
        print(report)
        
        # Per-class accuracy
        cm = confusion_matrix(true_labels, predictions)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        print("\nPer-Class Accuracy:")
        print("-" * 40)
        for class_name, acc in zip(class_names, per_class_acc):
            print(f"{class_name}: {acc*100:.2f}%")
        
        # Worst performing classes
        worst_indices = np.argsort(per_class_acc)[:5]
        print("\nWorst Performing Classes:")
        print("-" * 40)
        for idx in worst_indices:
            print(f"{class_names[idx]}: {per_class_acc[idx]*100:.2f}%")
