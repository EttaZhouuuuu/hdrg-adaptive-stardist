"""
Trainer for Adaptive Shape StarDist
====================================

Handles the training loop, data loading, and optimization.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm

from .loss import AdaptiveShapeLoss


class AdaptiveShapeTrainer:
    """
    Trainer class for Adaptive Shape StarDist
    
    Parameters:
    -----------
    model : AdaptiveShapeStarDist
        Model to train
    config : AdaptiveShapeConfig
        Training configuration
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Setup optimizer
        self.optimizer = keras.optimizers.Adam(
            learning_rate=config.train_learning_rate
        )
        
        # Setup loss function
        self.loss_fn = AdaptiveShapeLoss(
            prob_weight=1.0,
            dist_weight=1.0,
            complexity_weight=0.1,
            smoothness_weight=0.5,
            prior_weight=0.2
        )
        
        # Metrics
        self.train_loss_metric = keras.metrics.Mean(name='train_loss')
        self.val_loss_metric = keras.metrics.Mean(name='val_loss')
        
        # History
        self.history = {
            'loss': [],
            'val_loss': [],
        }
    
    def train(self, X_train, Y_train, validation_data=None, augmenter=None):
        """
        Train the model
        
        Parameters:
        -----------
        X_train : np.ndarray or list
            Training images
        Y_train : np.ndarray or list
            Training labels
        validation_data : tuple
            (X_val, Y_val)
        augmenter : callable
            Data augmentation function
            
        Returns:
        --------
        history : dict
            Training history
        """
        
        print("Starting training...")
        print(f"Training samples: {len(X_train)}")
        if validation_data is not None:
            print(f"Validation samples: {len(validation_data[0])}")
        
        # Create data generators
        train_generator = self._create_generator(
            X_train, Y_train,
            batch_size=self.config.train_batch_size,
            augment=augmenter is not None,
            augmenter=augmenter
        )
        
        if validation_data is not None:
            val_generator = self._create_generator(
                validation_data[0], validation_data[1],
                batch_size=self.config.train_batch_size,
                augment=False
            )
        else:
            val_generator = None
        
        # Training loop
        for epoch in range(self.config.train_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.train_epochs}")
            
            # Train for one epoch
            train_loss = self._train_epoch(train_generator)
            
            # Validate
            if val_generator is not None:
                val_loss = self._validate_epoch(val_generator)
            else:
                val_loss = None
            
            # Update history
            self.history['loss'].append(train_loss)
            if val_loss is not None:
                self.history['val_loss'].append(val_loss)
            
            # Print progress
            print(f"Train Loss: {train_loss:.4f}", end="")
            if val_loss is not None:
                print(f", Val Loss: {val_loss:.4f}")
            else:
                print()
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.model.save_model()
                print(f"Checkpoint saved at epoch {epoch + 1}")
        
        # Final save
        self.model.save_model()
        print("\nTraining complete!")
        
        return self.history
    
    def _train_epoch(self, generator):
        """Train for one epoch"""
        
        self.train_loss_metric.reset_states()
        
        # Progress bar
        pbar = tqdm(range(self.config.train_steps_per_epoch), desc="Training")
        
        for step in pbar:
            try:
                # Get batch
                batch = next(generator)
                X_batch, Y_batch = batch
                
                # Training step
                loss = self._train_step(X_batch, Y_batch)
                
                # Update metrics
                self.train_loss_metric.update_state(loss)
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss:.4f}'})
                
            except StopIteration:
                break
            except Exception as e:
                print(f"\nError in training step: {e}")
                continue
        
        return self.train_loss_metric.result().numpy()
    
    @tf.function
    def _train_step(self, X_batch, Y_batch):
        """Single training step"""
        
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self.model(X_batch, training=True)
            
            # Compute loss
            loss = self.loss_fn(Y_batch, predictions)
        
        # Backward pass
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss
    
    def _validate_epoch(self, generator):
        """Validate for one epoch"""
        
        self.val_loss_metric.reset_states()
        
        # Number of validation steps
        val_steps = max(1, len(generator) // self.config.train_batch_size)
        
        for step in range(val_steps):
            try:
                # Get batch
                batch = next(generator)
                X_batch, Y_batch = batch
                
                # Validation step
                loss = self._val_step(X_batch, Y_batch)
                
                # Update metrics
                self.val_loss_metric.update_state(loss)
                
            except StopIteration:
                break
            except Exception as e:
                print(f"\nError in validation step: {e}")
                continue
        
        return self.val_loss_metric.result().numpy()
    
    @tf.function
    def _val_step(self, X_batch, Y_batch):
        """Single validation step"""
        
        # Forward pass (no gradient)
        predictions = self.model(X_batch, training=False)
        
        # Compute loss
        loss = self.loss_fn(Y_batch, predictions)
        
        return loss
    
    def _create_generator(self, X, Y, batch_size=4, augment=False, augmenter=None):
        """
        Create data generator
        
        Yields batches of (images, labels) with proper formatting
        for the adaptive shape model.
        """
        
        num_samples = len(X)
        indices = np.arange(num_samples)
        
        while True:
            # Shuffle indices
            np.random.shuffle(indices)
            
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                X_batch = []
                Y_prob_batch = []
                Y_dist_batch = []
                
                for idx in batch_indices:
                    img = X[idx]
                    label = Y[idx]
                    
                    # Augmentation
                    if augment and augmenter is not None:
                        img, label = augmenter(img, label)
                    
                    # Normalize image
                    img = self._normalize_image(img)
                    
                    # Prepare labels
                    prob_map = self._create_prob_map(label)
                    dist_map = self._create_dist_map(label)
                    
                    X_batch.append(img)
                    Y_prob_batch.append(prob_map)
                    Y_dist_batch.append(dist_map)
                
                # Stack batches
                X_batch = np.stack(X_batch, axis=0)
                Y_prob_batch = np.stack(Y_prob_batch, axis=0)
                Y_dist_batch = np.stack(Y_dist_batch, axis=0)
                
                # Format as expected by loss function
                Y_batch = {
                    'prob': Y_prob_batch,
                    'dist': Y_dist_batch,
                    'mask': np.ones_like(Y_prob_batch),
                }
                
                yield (X_batch, Y_batch)
    
    def _normalize_image(self, img):
        """Normalize image to zero mean and unit variance"""
        
        img = img.astype(np.float32)
        
        # Ensure channel dimension
        if img.ndim == 2:
            img = img[..., np.newaxis]
        
        # Normalize
        mean = np.mean(img)
        std = np.std(img)
        img = (img - mean) / (std + 1e-7)
        
        return img
    
    def _create_prob_map(self, label):
        """
        Create probability map from instance labels
        
        Binary map: 1 for cell regions, 0 for background
        """
        
        prob_map = (label > 0).astype(np.float32)
        
        if prob_map.ndim == 2:
            prob_map = prob_map[..., np.newaxis]
        
        return prob_map
    
    def _create_dist_map(self, label):
        """
        Create distance map from instance labels
        
        For each pixel, compute distances to boundary in
        adaptive sampling directions.
        
        Simplified version - full implementation would use
        actual ray distances like StarDist.
        """
        from scipy.ndimage import distance_transform_edt
        
        # For simplicity, create distance transform
        # Full version would compute radial distances
        
        if label.ndim == 2:
            dist = distance_transform_edt(label > 0)
        else:
            dist = distance_transform_edt(label[..., 0] > 0)
        
        # Replicate across sampling points
        # [H, W] -> [H, W, N_points]
        dist = np.stack(
            [dist] * self.config.max_sampling_points,
            axis=-1
        )
        
        return dist.astype(np.float32)


class LearningRateScheduler:
    """
    Learning rate scheduler with warmup and decay
    """
    
    def __init__(
        self,
        initial_lr=1e-3,
        warmup_steps=1000,
        decay_steps=10000,
        decay_rate=0.96
    ):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
    
    def __call__(self, step):
        """Get learning rate for current step"""
        
        # Warmup
        if step < self.warmup_steps:
            return self.initial_lr * (step / self.warmup_steps)
        
        # Exponential decay
        step_adjusted = step - self.warmup_steps
        return self.initial_lr * (
            self.decay_rate ** (step_adjusted / self.decay_steps)
        )

