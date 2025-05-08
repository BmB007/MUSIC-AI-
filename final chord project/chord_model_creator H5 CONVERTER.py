import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Import the ChordExtractionSystem class from the provided script
from chord_extraction import ChordExtractionSystem

def create_and_save_h5_model(output_path="chord_recognition_model.h5", samples=5000, epochs=30):
    """
    Create, train and save a chord recognition model in H5 format.
    
    Args:
        output_path (str): Path where the model will be saved
        samples (int): Number of synthetic samples to generate for training
        epochs (int): Number of training epochs
    """
    print(f"Creating chord recognition model with {samples} synthetic samples...")
    
    # Create chord extraction system
    chord_system = ChordExtractionSystem()
    


    # Generate synthetic dataset
    X_train, X_val, y_train, y_val = chord_system.create_synthetic_dataset(num_samples=samples)
    
    X = X_train  # Assign X_train to X
    y = y_train  # Assign y_train to y
    # Split into training and validation sets
    #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build the model
    chord_system.build_model()
    
    # Train the model
   # Train model with X and y
    #history = chord_system.train_model(X, y, epochs=epochs)
    #history = chord_system.train_model(X, y, epochs=epochs)
    history = chord_system.train_model(X, y, X_val, y_val, epochs=epochs)

    # Save the model in H5 format
    success = chord_system.save_model(output_path)
    
    if success:
        print(f"Model successfully saved to {output_path}")
        
        # Evaluate the model
        loss, accuracy = chord_system.model.evaluate(X_val, y_val)
        print(f"Validation accuracy: {accuracy:.4f}")
        
        # Print model summary
        chord_system.model.summary()
        
        return True
    else:
        print("Failed to save the model")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create and save a chord recognition model in H5 format')
    parser.add_argument('--output', default='chord_recognition_model.h5', help='Output path for the H5 model')
    parser.add_argument('--samples', type=int, default=5000, help='Number of synthetic samples to generate')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    
    args = parser.parse_args()
    
    create_and_save_h5_model(args.output, args.samples, args.epochs)
