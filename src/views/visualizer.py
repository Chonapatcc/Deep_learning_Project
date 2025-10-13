"""
View: Visualization and Display
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class Visualizer:
    """Handle visualization of results"""
    
    @staticmethod
    def plot_training_history(history, save_path='training_history.png'):
        """Plot training and validation metrics"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss plot
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        # Learning rate plot
        axes[2].plot(epochs, history['learning_rate'], 'g-')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_yscale('log')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(true_labels, predictions, class_names,
                             save_path='confusion_matrix.png'):
        """Plot confusion matrix"""
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        plt.show()
