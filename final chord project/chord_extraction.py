import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tempfile
import warnings
warnings.filterwarnings('ignore')

class ChordExtractionSystem:
    """
    A complete system for extracting chords from music audio recordings
    and generating chord charts for piano and guitar.
    """
    
    def __init__(self, model_path=None):
        # Constants for audio processing
        self.SR = 22050  # Sample rate
        self.HOP_LENGTH = 512  # Hop length for feature extraction
        self.N_FFT = 2048  # FFT window size
        self.N_CHROMA = 12  # Number of chroma features
        
        # Chord vocabulary (can be expanded)
        self.chord_vocabulary = [
            'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
            'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm',
            'C7', 'C#7', 'D7', 'D#7', 'E7', 'F7', 'F#7', 'G7', 'G#7', 'A7', 'A#7', 'B7',
            'Cmaj7', 'C#maj7', 'Dmaj7', 'D#maj7', 'Emaj7', 'Fmaj7', 'F#maj7', 'Gmaj7', 'G#maj7', 'Amaj7', 'A#maj7', 'Bmaj7',
            'N'  # No chord or silence
        ]
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.chord_vocabulary)
        
        # Guitar and piano chord mappings
        self.guitar_chord_mappings = self._create_guitar_chord_mappings()
        self.piano_chord_mappings = self._create_piano_chord_mappings()
        
        # Initialize or load model
        if model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            self.model = None
            print("No model loaded. You need to train a model first.")
    
    def _create_guitar_chord_mappings(self):
        """
        Create a dictionary mapping chord names to guitar fingerings.
        Returns a dictionary of chord_name -> fingering_pattern.
        """
        # Format: Strings from low E to high E (6 to 1)
        # X means don't play that string, 0 means open string
        guitar_chords = {
            # Major chords
            'C': '(x,3,2,0,1,0)',
            'C#': '(x,4,6,6,6,4)',
            'D': '(x,x,0,2,3,2)',
            'D#': '(x,x,1,3,4,3)',
            'E': '(0,2,2,1,0,0)',
            'F': '(1,3,3,2,1,1)',
            'F#': '(2,4,4,3,2,2)',
            'G': '(3,2,0,0,0,3)',
            'G#': '(4,6,6,5,4,4)',
            'A': '(x,0,2,2,2,0)',
            'A#': '(x,1,3,3,3,1)',
            'B': '(x,2,4,4,4,2)',
            
            # Minor chords
            'Cm': '(x,3,5,5,4,3)',
            'C#m': '(x,4,6,6,5,4)',
            'Dm': '(x,x,0,2,3,1)',
            'D#m': '(x,x,1,3,4,2)',
            'Em': '(0,2,2,0,0,0)',
            'Fm': '(1,3,3,1,1,1)',
            'F#m': '(2,4,4,2,2,2)',
            'Gm': '(3,5,5,3,3,3)',
            'G#m': '(4,6,6,4,4,4)',
            'Am': '(x,0,2,2,1,0)',
            'A#m': '(x,1,3,3,2,1)',
            'Bm': '(x,2,4,4,3,2)',
            
            # 7th chords
            'C7': '(x,3,2,3,1,0)',
            'C#7': '(x,4,3,4,2,0)',
            'D7': '(x,x,0,2,1,2)',
            'D#7': '(x,x,1,3,2,3)',
            'E7': '(0,2,0,1,0,0)',
            'F7': '(1,3,1,2,1,1)',
            'F#7': '(2,4,2,3,2,2)',
            'G7': '(3,2,0,0,0,1)',
            'G#7': '(4,6,4,5,4,4)',
            'A7': '(x,0,2,0,2,0)',
            'A#7': '(x,1,3,1,3,1)',
            'B7': '(x,2,1,2,0,2)',
            
            # Major 7th chords
            'Cmaj7': '(x,3,2,0,0,0)',
            'C#maj7': '(x,4,3,1,1,1)',
            'Dmaj7': '(x,x,0,2,2,2)',
            'D#maj7': '(x,x,1,3,3,3)',
            'Emaj7': '(0,2,1,1,0,0)',
            'Fmaj7': '(1,3,2,2,1,0)',
            'F#maj7': '(2,4,3,3,2,2)',
            'Gmaj7': '(3,2,0,0,0,2)',
            'G#maj7': '(4,6,5,5,4,4)',
            'Amaj7': '(x,0,2,1,2,0)',
            'A#maj7': '(x,1,3,2,3,1)',
            'Bmaj7': '(x,2,4,3,4,2)',
            
            # No chord
            'N': '(x,x,x,x,x,x)'
        }
        return guitar_chords
    
    def _create_piano_chord_mappings(self):
        """
        Create a dictionary mapping chord names to piano fingerings.
        Returns a dictionary of chord_name -> keys_to_play.
        """
        # Format: Notes to play in the chord (C3-C5 range)
        piano_chords = {
            # Major chords
            'C': '(C, E, G)',
            'C#': '(C#, F, G#)',
            'D': '(D, F#, A)',
            'D#': '(D#, G, A#)',
            'E': '(E, G#, B)',
            'F': '(F, A, C)',
            'F#': '(F#, A#, C#)',
            'G': '(G, B, D)',
            'G#': '(G#, C, D#)',
            'A': '(A, C#, E)',
            'A#': '(A#, D, F)',
            'B': '(B, D#, F#)',
            
            # Minor chords
            'Cm': '(C, D#, G)',
            'C#m': '(C#, E, G#)',
            'Dm': '(D, F, A)',
            'D#m': '(D#, F#, A#)',
            'Em': '(E, G, B)',
            'Fm': '(F, G#, C)',
            'F#m': '(F#, A, C#)',
            'Gm': '(G, A#, D)',
            'G#m': '(G#, B, D#)',
            'Am': '(A, C, E)',
            'A#m': '(A#, C#, F)',
            'Bm': '(B, D, F#)',
            
            # 7th chords
            'C7': '(C, E, G, A#)',
            'C#7': '(C#, F, G#, B)',
            'D7': '(D, F#, A, C)',
            'D#7': '(D#, G, A#, C#)',
            'E7': '(E, G#, B, D)',
            'F7': '(F, A, C, D#)',
            'F#7': '(F#, A#, C#, E)',
            'G7': '(G, B, D, F)',
            'G#7': '(G#, C, D#, F#)',
            'A7': '(A, C#, E, G)',
            'A#7': '(A#, D, F, G#)',
            'B7': '(B, D#, F#, A)',
            
            # Major 7th chords
            'Cmaj7': '(C, E, G, B)',
            'C#maj7': '(C#, F, G#, C)',
            'Dmaj7': '(D, F#, A, C#)',
            'D#maj7': '(D#, G, A#, D)',
            'Emaj7': '(E, G#, B, D#)',
            'Fmaj7': '(F, A, C, E)',
            'F#maj7': '(F#, A#, C#, F)',
            'Gmaj7': '(G, B, D, F#)',
            'G#maj7': '(G#, C, D#, G)',
            'Amaj7': '(A, C#, E, G#)',
            'A#maj7': '(A#, D, F, A)',
            'Bmaj7': '(B, D#, F#, A#)',
            
            # No chord
            'N': '()'
        }
        return piano_chords
    
    def load_audio(self, file_path):
        """
        Load and preprocess audio file.
        """
        print(f"Loading audio file: {file_path}")
        
        try:
            y, sr = librosa.load(file_path, sr=self.SR)
            print(f"Audio loaded successfully. Duration: {len(y)/sr:.2f}s")
            
            # Basic preprocessing - normalize audio
            y = librosa.util.normalize(y)
            return y, sr
        except Exception as e:
            print(f"Error loading audio: {str(e)}")
            return None, None
    
    def extract_features(self, y, sr=None):
        """
        Extract chromagram features from the audio signal.
        """
        if sr is None:
            sr = self.SR
            
        if y is None:
            print("Cannot extract features: No audio data")
            return None
        
        # Extract chromagram features
        chroma = librosa.feature.chroma_cqt(
            y=y, 
            sr=sr, 
            hop_length=self.HOP_LENGTH, 
            n_chroma=self.N_CHROMA
        )
        
        # Normalize features
        chroma = librosa.util.normalize(chroma)
        
        print(f"Extracted chromagram with shape: {chroma.shape}")
        return chroma

    def segment_features(self, chroma, segment_size=10):
        """
        Segment chromagram features into fixed-size chunks for chord analysis.
        """
        # Each segment will cover about segment_size frames
        segments = []
        
        for i in range(0, chroma.shape[1], segment_size):
            end_idx = min(i + segment_size, chroma.shape[1])
            segment = chroma[:, i:end_idx]
            
            # If segment is shorter than segment_size, pad it
            if segment.shape[1] < segment_size:
                pad_width = ((0, 0), (0, segment_size - segment.shape[1]))
                segment = np.pad(segment, pad_width, mode='constant')
            
            # Take the mean across time to get a single 12-dimensional vector
            segment_mean = np.mean(segment, axis=1)
            segments.append(segment_mean)
        
        segments = np.array(segments)
        print(f"Segmented features into {len(segments)} chunks")
        return segments
    
    def build_model(self):
        """
        Build a neural network model for chord recognition.
        """
        print("Building chord recognition model...")
        
        # Input shape: 12 chroma features
        inputs = layers.Input(shape=(self.N_CHROMA,))
        
        # Hidden layers
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        
        # Output layer - number of chord classes
        outputs = layers.Dense(len(self.chord_vocabulary), activation='softmax')(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("Model built successfully.")
        print(model.summary())
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        """
        Train the chord recognition model.
        """
        if self.model is None:
            self.build_model()
        
        print("Training chord recognition model...")
        
        # Training callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.0001)
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Model training completed.")
        return history
    
    def save_model(self, file_path):
        """
        Save the trained model to disk.
        """
        if self.model is None:
            print("No model to save.")
            return False
        
        try:
            self.model.save(file_path)
            print(f"Model saved to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def predict_chords(self, features):
        """
        Predict chords from feature segments.
        """
        if self.model is None:
            print("No model available for prediction. Train or load a model first.")
            return None
        
        # Make predictions
        predictions = self.model.predict(features)
        
        # Get the most likely chord for each segment
        chord_indices = np.argmax(predictions, axis=1)
        
        # Convert indices to chord names
        predicted_chords = self.label_encoder.inverse_transform(chord_indices)
        
        print(f"Predicted {len(predicted_chords)} chords")
        return predicted_chords
    
    def smooth_chord_sequence(self, chord_sequence, window_size=5):
        """
        Apply smoothing to the chord sequence to remove noise and quick changes.
        This uses a majority voting within a sliding window.
        """
        smoothed_sequence = []
        
        for i in range(len(chord_sequence)):
            # Define window boundaries
            start = max(0, i - window_size // 2)
            end = min(len(chord_sequence), i + window_size // 2 + 1)
            
            # Get chords in the current window
            window_chords = chord_sequence[start:end]
            
            # Count occurrences of each chord in the window
            unique_chords, counts = np.unique(window_chords, return_counts=True)
            
            # Get the most frequent chord
            most_common_chord = unique_chords[np.argmax(counts)]
            smoothed_sequence.append(most_common_chord)
        
        return np.array(smoothed_sequence)
    
    def create_synthetic_dataset(self, num_samples=2000):
        """
        Create a synthetic dataset for training the chord recognition model.
        Each sample consists of a chromagram simulating a chord and the corresponding chord label.
        """
        print(f"Creating synthetic dataset with {num_samples} samples...")
        
        X = []  # Will hold the chromagram features
        y = []  # Will hold the chord labels
        
        # Chord templates (idealized chroma patterns)
        chord_templates = {
            # Major chords
            'C': np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]),  # C, E, G
            'C#': np.array([0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]),  # C#, F, G#
            'D': np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]),  # D, F#, A
            'D#': np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]),  # D#, G, A#
            'E': np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]),  # E, G#, B
            'F': np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]),  # F, A, C
            'F#': np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]),  # F#, A#, C#
            'G': np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]),  # G, B, D
            'G#': np.array([1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]),  # G#, C, D#
            'A': np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]),  # A, C#, E
            'A#': np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]),  # A#, D, F
            'B': np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]),  # B, D#, F#
            
            # Minor chords
            'Cm': np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]),  # C, D#, G
            'C#m': np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]),  # C#, E, G#
            'Dm': np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]),  # D, F, A
            'D#m': np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]),  # D#, F#, A#
            'Em': np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]),  # E, G, B
            'Fm': np.array([1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]),  # F, G#, C
            'F#m': np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]),  # F#, A, C#
            'Gm': np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]),  # G, A#, D
            'G#m': np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1]),  # G#, B, D#
            'Am': np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]),  # A, C, E
            'A#m': np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]),  # A#, C#, F
            'Bm': np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]),  # B, D, F#
            
            # 7th chords (simplified patterns)
            'C7': np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),  # C, E, G, A#
            'C#7': np.array([0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]),  # C#, F, G#, B
            'D7': np.array([1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]),  # D, F#, A, C
            'D#7': np.array([0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]),  # D#, G, A#, C#
            'E7': np.array([0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1]),  # E, G#, B, D
            'F7': np.array([1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0]),  # F, A, C, D#
            'F#7': np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0]),  # F#, A#, C#, E
            'G7': np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1]),  # G, B, D, F
            'G#7': np.array([1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0]),  # G#, C, D#, F#
            'A7': np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]),  # A, C#, E, G
            'A#7': np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0]),  # A#, D, F, G#
            'B7': np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1]),  # B, D#, F#, A
            
            # Major 7th chords
            'Cmaj7': np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]),  # C, E, G, B
            'C#maj7': np.array([1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]),  # C#, F, G#, C
            'Dmaj7': np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]),  # D, F#, A, C#
            'D#maj7': np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]),  # D#, G, A#, D
            'Emaj7': np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]),  # E, G#, B, D#
            'Fmaj7': np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]),  # F, A, C, E
            'F#maj7': np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]),  # F#, A#, C#, F
            'Gmaj7': np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]),  # G, B, D, F#
            'G#maj7': np.array([1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]),  # G#, C, D#, G
            'Amaj7': np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]),  # A, C#, E, G#
            'A#maj7': np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]),  # A#, D, F, A
            'Bmaj7': np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]),  # B, D#, F#, A#
            
            # No chord
            'N': np.zeros(12)  # No active notes
        }
        
        # For each chord in our vocabulary
        for chord_name in self.chord_vocabulary:
            # Get the ideal template for this chord
            template = chord_templates[chord_name]
            
            # Create the specified number of samples per chord
            samples_per_chord = num_samples // len(self.chord_vocabulary)
            
            for _ in range(samples_per_chord):
                # Add some noise to make it more realistic
                noise_level = np.random.uniform(0.05, 0.3)
                noise = np.random.normal(0, noise_level, size=template.shape)
                
                # Create the sample
                sample = template + noise
                
                # Ensure non-negativity and normalize
                sample = np.maximum(sample, 0)
                if np.sum(sample) > 0:  # Avoid division by zero
                    sample = sample / np.sum(sample)
                
                # Add to datasets
                X.append(sample)
                y.append(self.label_encoder.transform([chord_name])[0])
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Dataset created. X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
        
        return X_train, X_val, y_train, y_val
    


    def visualize_chord_progression(self, chord_sequence, segment_duration=0.5):
        """
        Visualize the chord progression as a timeline.
        """
        time_points = np.arange(len(chord_sequence) + 1) * segment_duration  # One more time point for 'post' step

        # Create a colormap for the chords
        unique_chords = sorted(set(chord_sequence))
        chord_to_idx = {chord: idx for idx, chord in enumerate(unique_chords)}

        # Map chord sequence to indices
        y_vals = [chord_to_idx[chord] for chord in chord_sequence]
        y_vals.append(y_vals[-1])  # Repeat last value for proper step plot

        # Plot the chord changes
        plt.figure(figsize=(12, 4))
        plt.step(time_points, y_vals, where='post')

        # Set y-ticks to chord names
        plt.yticks(range(len(unique_chords)), unique_chords)

        plt.xlabel('Time (seconds)')
        plt.ylabel('Chord')
        plt.title('Chord Progression Timeline')
        plt.grid(True)
        plt.tight_layout()
        plt.show()