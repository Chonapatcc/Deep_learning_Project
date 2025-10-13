"""
Utility: Data Loading and Saving
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


class DataHandler:
    """Handle data loading, saving, and splitting"""
    
    @staticmethod
    def save_processed_data(X, y, output_path='data/processed/processed_data.pkl'):
        """Save processed data to pickle file"""
        # Create directory if not exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump({'X': X, 'y': y}, f)
        print(f"Data saved to {output_path}")
    
    @staticmethod
    def load_processed_data(input_path='data/processed/processed_data.pkl'):
        """Load processed data from pickle file"""
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        return data['X'], data['y']
    
    @staticmethod
    def save_label_encoder(label_encoder, output_path='data/processed/label_encoder.pkl'):
        """Save label encoder to pickle file"""
        # Create directory if not exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        print(f"Label encoder saved to {output_path}")
    
    @staticmethod
    def load_label_encoder(input_path='data/processed/label_encoder.pkl'):
        """Load label encoder from pickle file"""
        with open(input_path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def filter_classes(X, y, min_samples_per_class=10):
        """
        Filter out classes with too few samples
        
        Returns:
            X_filtered, y_filtered, removed_classes
        """
        class_counts = {}
        for label in y:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        valid_classes = {label for label, count in class_counts.items() 
                        if count >= min_samples_per_class}
        
        removed = set(class_counts.keys()) - valid_classes
        
        if removed:
            print(f"⚠ Removing {len(removed)} classes with < {min_samples_per_class} samples: {sorted(removed)}")
            
            mask = np.array([label in valid_classes for label in y])
            X = X[mask]
            y = y[mask]
            
            print(f"Remaining samples: {len(X)}")
            print(f"Remaining classes: {len(valid_classes)}")
        
        return X, y, removed
    
    @staticmethod
    def split_data(X, y, test_size=0.3, val_size=0.5, random_state=42):
        """
        Split data into train, validation, and test sets
        
        Args:
            test_size: Proportion for temp split (val + test)
            val_size: Proportion of temp for validation
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
