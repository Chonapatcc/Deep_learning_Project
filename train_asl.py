"""
Main Training Script
Clean MVC architecture for ASL PyTorch training
"""

import torch
from torch.utils.data import DataLoader

from pytorch_asl.utils.preprocessor import ASLDataPreprocessor
from pytorch_asl.utils.data_handler import DataHandler
from pytorch_asl.models.dataset import ASLDataset
from pytorch_asl.models.classifier import ASLClassifier
from pytorch_asl.controllers.trainer import Trainer
from pytorch_asl.controllers.evaluator import Evaluator
from pytorch_asl.views.visualizer import Visualizer


def main():
    # ==================== CONFIGURATION ====================
    DATASET_PATH = "./datasets/asl_dataset/"
    PROCESSED_DATA_PATH = "asl_processed.pkl"
    MODEL_SAVE_PATH = "best_asl_model.pth"
    LABEL_ENCODER_PATH = "label_encoder.pkl"
    
    AUGMENT = True
    AUGMENT_FACTOR = 2
    FILTER_ALPHABET_ONLY = True
    MIN_SAMPLES_PER_CLASS = 10
    
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    PATIENCE = 10
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # ==================== STEP 1: PREPROCESS DATASET ====================
    print("\n" + "="*60)
    print("STEP 1: DATA PREPROCESSING")
    print("="*60)
    
    preprocessor = ASLDataPreprocessor()
    X, y = preprocessor.process_dataset(
        DATASET_PATH,
        augment=AUGMENT,
        augment_factor=AUGMENT_FACTOR,
        filter_alphabet_only=FILTER_ALPHABET_ONLY
    )
    preprocessor.close()
    
    # Save processed data
    DataHandler.save_processed_data(X, y, PROCESSED_DATA_PATH)
    
    # ==================== STEP 2: FILTER AND SPLIT DATA ====================
    print("\n" + "="*60)
    print("STEP 2: FILTERING AND SPLITTING DATA")
    print("="*60)
    
    # Filter classes with too few samples
    X, y, removed = DataHandler.filter_classes(X, y, MIN_SAMPLES_PER_CLASS)
    
    if len(X) == 0:
        print("❌ ERROR: No samples remaining after filtering!")
        return
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = DataHandler.split_data(X, y)
    
    # ==================== STEP 3: CREATE DATASETS ====================
    print("\n" + "="*60)
    print("STEP 3: CREATING DATASETS")
    print("="*60)
    
    train_dataset = ASLDataset(X_train, y_train)
    val_dataset = ASLDataset(X_val, y_val, label_encoder=train_dataset.label_encoder)
    test_dataset = ASLDataset(X_test, y_test, label_encoder=train_dataset.label_encoder)
    
    # Save label encoder
    DataHandler.save_label_encoder(train_dataset.label_encoder, LABEL_ENCODER_PATH)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # ==================== STEP 4: INITIALIZE MODEL ====================
    print("\n" + "="*60)
    print("STEP 4: MODEL INITIALIZATION")
    print("="*60)
    
    num_classes = len(train_dataset.label_encoder.classes_)
    model = ASLClassifier(input_size=63, num_classes=num_classes, dropout=0.3)
    
    print(f"Model architecture: ASLClassifier")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {train_dataset.label_encoder.classes_}")
    
    # ==================== STEP 5: TRAIN MODEL ====================
    print("\n" + "="*60)
    print("STEP 5: TRAINING")
    print("="*60)
    
    trainer = Trainer(model, device=DEVICE, learning_rate=LEARNING_RATE)
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        save_path=MODEL_SAVE_PATH,
        patience=PATIENCE
    )
    
    # ==================== STEP 6: VISUALIZE TRAINING ====================
    print("\n" + "="*60)
    print("STEP 6: VISUALIZATION")
    print("="*60)
    
    Visualizer.plot_training_history(history, save_path='training_history.png')
    
    # ==================== STEP 7: EVALUATE ON TEST SET ====================
    print("\n" + "="*60)
    print("STEP 7: FINAL EVALUATION")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    evaluator = Evaluator(model, device=DEVICE)
    test_acc, test_loss, predictions, true_labels = evaluator.evaluate(test_loader)
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Print classification report
    evaluator.print_classification_report(
        true_labels,
        predictions,
        train_dataset.label_encoder.classes_
    )
    
    # Plot confusion matrix
    Visualizer.plot_confusion_matrix(
        true_labels,
        predictions,
        train_dataset.label_encoder.classes_,
        save_path='confusion_matrix.png'
    )
    
    # ==================== COMPLETE ====================
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE")
    print("="*60)
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Label encoder saved to: {LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    main()
