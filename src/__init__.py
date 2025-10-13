"""
ASL Recognition Project - Source Code

Package structure:
- classifier.py: ASLClassifier neural network architecture
- dataset.py: Dataset handler
- config.py: Configuration settings
- controllers/: Training, prediction, evaluation logic
- utils/: Utility functions
- views/: Display and visualization
"""

__version__ = "1.0.0"

# Lazy imports to avoid importing torch/tensorflow when not needed
def __getattr__(name):
    if name == 'ASLClassifier':
        from .classifier import ASLClassifier
        return ASLClassifier
    elif name == 'ASLDataset':
        from .dataset import ASLDataset
        return ASLDataset
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['ASLClassifier', 'ASLDataset']
