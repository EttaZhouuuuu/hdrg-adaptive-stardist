"""
Adaptive Shape StarDist Model
==============================

Main model that integrates all components:
- Backbone network for feature extraction
- Shape prior encoder for domain knowledge
- Adaptive shape encoder for flexible representation
- Multi-head output for instance segmentation

This model replaces StarDist's fixed-ray representation with
an adaptive, learned shape encoding.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path

from .backbone import UNetBackbone, ResNetBackbone
from .fpn_backbone import FPNBackbone
from ..core.shape_encoder import AdaptiveShapeEncoder
from ..core.shape_prior import ShapePriorEncoder


class AdaptiveShapeConfig:
    """
    Configuration for Adaptive Shape StarDist model
    
    Parameters:
    -----------
    n_channel_in : int
        Number of input channels
    backbone : str
        Backbone architecture ('unet' or 'resnet')
    n_depth : int
        Depth of backbone network
    n_filter_base : int
        Base number of filters
    min_sampling_points : int
        Minimum number of adaptive sampling points
    max_sampling_points : int
        Maximum number of adaptive sampling points
    use_shape_prior : bool
        Whether to use shape prior encoder
    num_shape_prototypes : int
        Number of learnable shape prototypes
    deformable_groups : int
        Number of deformable convolution groups
    grid : tuple
        Downsampling grid for output
    train_learning_rate : float
        Learning rate for training
    train_epochs : int
        Number of training epochs
    train_steps_per_epoch : int
        Steps per epoch
    train_batch_size : int
        Batch size
    """
    
    def __init__(
        self,
        n_channel_in=1,
        backbone='unet',
        n_depth=3,
        n_filter_base=32,
        min_sampling_points=32,
        max_sampling_points=128,
        use_shape_prior=True,
        num_shape_prototypes=16,
        deformable_groups=4,
        grid=(1, 1),
        # FPN specific parameters
        use_fpn=False,
        fpn_channels=256,
        fpn_levels=['p2', 'p3', 'p4', 'p5'],
        multiscale_prediction=False,
        train_learning_rate=3e-4,
        train_epochs=100,
        train_steps_per_epoch=100,
        train_batch_size=4,
        # Regularization
        weight_decay=1e-5,
        dropout_rate=0.3,
        label_smoothing=0.1,
        **kwargs
    ):
        # Model architecture
        self.n_channel_in = n_channel_in
        self.backbone = backbone.lower()
        self.n_depth = n_depth
        self.n_filter_base = n_filter_base
        
        # Adaptive shape encoding
        self.min_sampling_points = min_sampling_points
        self.max_sampling_points = max_sampling_points
        self.use_shape_prior = use_shape_prior
        self.num_shape_prototypes = num_shape_prototypes
        self.deformable_groups = deformable_groups
        
        # FPN configuration
        self.use_fpn = use_fpn
        self.fpn_channels = fpn_channels
        self.fpn_levels = fpn_levels
        self.multiscale_prediction = multiscale_prediction
        
        # Output configuration
        self.grid = grid
        
        # Training configuration
        self.train_learning_rate = train_learning_rate
        self.train_epochs = train_epochs
        self.train_steps_per_epoch = train_steps_per_epoch
        self.train_batch_size = train_batch_size
        
        # Regularization settings
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.label_smoothing = label_smoothing
        
        # Additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self):
        """Convert config to dictionary"""
        return vars(self)
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        return cls(**config_dict)


class AdaptiveShapeStarDist(keras.Model):
    """
    Adaptive Shape StarDist Model
    
    Main model for cell instance segmentation using adaptive shape encoding.
    
    Parameters:
    -----------
    config : AdaptiveShapeConfig
        Model configuration
    name : str
        Model name
    basedir : str
        Base directory for saving models
    """
    
    def __init__(self, config, name='adaptive_shape_stardist', basedir='.'):
        super(AdaptiveShapeStarDist, self).__init__(name=name)
        
        self.config = config
        self.basedir = Path(basedir)
        self.model_dir = self.basedir / name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Build model components
        self._build_model()
    
    def _build_model(self):
        """Build all model components"""
        
        # 1. Backbone network
        if self.config.use_fpn:
            # Use FPN backbone (overrides backbone setting)
            self.backbone = FPNBackbone(
                n_channel_in=self.config.n_channel_in,
                backbone='resnet34',  # FPN uses ResNet by default
                fpn_channels=self.config.fpn_channels,
                name='fpn_backbone'
            )
            self.is_fpn = True
        elif self.config.backbone == 'unet':
            self.backbone = UNetBackbone(
                n_depth=self.config.n_depth,
                n_filter_base=self.config.n_filter_base,
                n_channel_in=self.config.n_channel_in,
                name='backbone'
            )
            self.is_fpn = False
        elif self.config.backbone == 'resnet':
            self.backbone = ResNetBackbone(
                n_blocks=self.config.n_depth,
                n_filter_base=self.config.n_filter_base,
                n_channel_in=self.config.n_channel_in,
                name='backbone'
            )
            self.is_fpn = False
        else:
            raise ValueError(f"Unknown backbone: {self.config.backbone}")
        
        # 2. Shape prior encoder (optional)
        if self.config.use_shape_prior:
            self.shape_prior_encoder = ShapePriorEncoder(
                num_prototypes=self.config.num_shape_prototypes,
                embedding_dim=256,
                num_heads=8,
                num_layers=3,
                name='shape_prior_encoder'
            )
        else:
            self.shape_prior_encoder = None
        
        # 3. Adaptive shape encoder
        self.shape_encoder = AdaptiveShapeEncoder(
            min_sampling_points=self.config.min_sampling_points,
            max_sampling_points=self.config.max_sampling_points,
            feature_dims=[64, 128, 256],
            deformable_groups=self.config.deformable_groups,
            use_polar=True,
            name='shape_encoder'
        )
        
        # 4. Output heads
        if self.is_fpn and self.config.multiscale_prediction:
            self._build_multiscale_heads()
        else:
            self._build_output_heads()
    
    def _build_output_heads(self):
        """Build single-scale output prediction heads"""
        
        dropout_rate = getattr(self.config, 'dropout_rate', 0.3)
        
        # Instance probability head
        self.prob_head = keras.Sequential([
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Conv2D(1, 1, activation='sigmoid'),
        ], name='prob_head')
        
        # Distance/boundary head (predicts radii at sampling points)
        self.dist_head = keras.Sequential([
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Conv2D(self.config.max_sampling_points, 1, activation='relu'),
        ], name='dist_head')
        
        # Complexity head (predicts shape complexity)
        self.complexity_head = keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(1, activation='sigmoid'),
        ], name='complexity_head')
    
    def _build_multiscale_heads(self):
        """Build multi-scale output prediction heads for FPN"""
        
        dropout_rate = getattr(self.config, 'dropout_rate', 0.3)
        
        # Create separate heads for each FPN level
        self.prob_heads = {}
        self.dist_heads = {}
        
        for level in self.config.fpn_levels:
            # Probability head for this level
            self.prob_heads[level] = keras.Sequential([
                layers.Conv2D(128, 3, padding='same', activation='relu'),
                layers.Dropout(dropout_rate),
                layers.Conv2D(64, 3, padding='same', activation='relu'),
                layers.Dropout(dropout_rate),
                layers.Conv2D(1, 1, activation='sigmoid'),
            ], name=f'prob_head_{level}')
            
            # Distance head for this level
            self.dist_heads[level] = keras.Sequential([
                layers.Conv2D(128, 3, padding='same', activation='relu'),
                layers.Dropout(dropout_rate),
                layers.Conv2D(64, 3, padding='same', activation='relu'),
                layers.Dropout(dropout_rate),
                layers.Conv2D(self.config.max_sampling_points, 1, activation='relu'),
            ], name=f'dist_head_{level}')
        
        # Single complexity head (global, not scale-specific)
        self.complexity_head = keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(1, activation='sigmoid'),
        ], name='complexity_head')
    
    def call(self, inputs, training=None):
        """
        Forward pass
        
        Parameters:
        -----------
        inputs : tf.Tensor or dict
            Input image(s) [B, H, W, C] or dict with 'image' key
        training : bool
            Training mode flag
            
        Returns:
        --------
        output : dict
            If multiscale_prediction=False (default):
                - 'prob': Instance probability map [B, H, W, 1]
                - 'dist': Distance predictions [B, H, W, N_points]
                - 'sampling_points': Adaptive sampling points [B, N_points, 2]
                - 'complexity': Shape complexity [B, 1]
                - 'features': Intermediate features
            If multiscale_prediction=True (FPN):
                - 'multiscale_predictions': dict with keys 'p2', 'p3', 'p4', 'p5'
                  Each containing {'prob': ..., 'dist': ...}
                - 'complexity': Shape complexity [B, 1]
                - 'fpn_features': FPN feature pyramid
        """
        
        # Handle input format
        if isinstance(inputs, dict):
            image = inputs['image']
        else:
            image = inputs
        
        # 1. Extract features with backbone
        backbone_output = self.backbone(image, training=training)
        
        # Check if using FPN
        if self.is_fpn:
            return self._forward_fpn(backbone_output, training=training)
        else:
            return self._forward_standard(backbone_output, training=training)
    
    def _forward_standard(self, backbone_output, training=None):
        """Standard forward pass (single scale)"""
        features = backbone_output['features']
        
        # 2. Apply shape prior encoding (if enabled)
        if self.shape_prior_encoder is not None:
            prior_output = self.shape_prior_encoder(
                {'features': features},
                training=training
            )
            features = prior_output['prior_features']
            prototype_weights = prior_output['prototype_weights']
        else:
            prototype_weights = None
        
        # 3. Adaptive shape encoding
        shape_output = self.shape_encoder(
            {'features': features},
            training=training
        )
        
        shape_features = shape_output['shape_features']
        sampling_points = shape_output['sampling_points']
        complexity = shape_output['complexity']
        
        # 4. Predict outputs
        prob = self.prob_head(shape_features, training=training)
        dist = self.dist_head(shape_features, training=training)
        
        # 5. Assemble output
        output = {
            'prob': prob,  # [B, H, W, 1]
            'dist': dist,  # [B, H, W, N_points]
            'sampling_points': sampling_points,  # [B, N_points, 2]
            'point_features': shape_output['point_features'],  # [B, N_points, D]
            'complexity': complexity,  # [B, 1]
            'shape_descriptor': shape_output['shape_descriptor'],  # [B, 128]
            'features': shape_features,  # [B, H, W, C]
        }
        
        if prototype_weights is not None:
            output['prototype_weights'] = prototype_weights
        
        return output
    
    def _forward_fpn(self, backbone_output, training=None):
        """FPN forward pass (multi-scale)"""
        
        # FPN features: p2, p3, p4, p5
        fpn_features = backbone_output
        
        if self.config.multiscale_prediction:
            # Multi-scale predictions: predict at each FPN level
            # Apply shape prior encoding to all scales at once (multi-scale processing)
            if self.shape_prior_encoder is not None:
                prior_output = self.shape_prior_encoder(
                    {'fpn_features': fpn_features},
                    training=training
                )
                # Use multi-scale prior features
                prior_multiscale = prior_output.get('multiscale_prior_features', {})
                prototype_weights = prior_output['prototype_weights']
            else:
                prior_multiscale = {}
                prototype_weights = None
            
            multiscale_predictions = {}
            
            for level in self.config.fpn_levels:
                features = fpn_features[level]
                
                # Use prior features if available, otherwise use original features
                if level in prior_multiscale:
                    features = prior_multiscale[level]
                
                # Predict at this scale
                prob = self.prob_heads[level](features, training=training)
                dist = self.dist_heads[level](features, training=training)
                
                multiscale_predictions[level] = {
                    'prob': prob,
                    'dist': dist,
                }
            
            # Global complexity from finest scale (p2)
            complexity = self.complexity_head(fpn_features['p2'])
            
            output = {
                'multiscale_predictions': multiscale_predictions,
                'complexity': complexity,
                'fpn_features': fpn_features,
            }
            
            if prototype_weights is not None:
                output['prototype_weights'] = prototype_weights
            
            # Include learned prototypes for analysis
            if self.shape_prior_encoder is not None:
                output['learned_prototypes'] = prior_output.get('learned_prototypes')
                output['attention_maps'] = prior_output.get('attention_maps')
        
        else:
            # Single-scale prediction: use only p2 (highest resolution)
            features = fpn_features['p2']
            
            # Apply shape prior encoding (if enabled)
            if self.shape_prior_encoder is not None:
                prior_output = self.shape_prior_encoder(
                    {'features': features},
                    training=training
                )
                features = prior_output['prior_features']
                prototype_weights = prior_output['prototype_weights']
            else:
                prototype_weights = None
            
            # Adaptive shape encoding
            shape_output = self.shape_encoder(
                {'features': features},
                training=training
            )
            
            shape_features = shape_output['shape_features']
            sampling_points = shape_output['sampling_points']
            complexity = shape_output['complexity']
            
            # Predict outputs
            prob = self.prob_head(shape_features, training=training)
            dist = self.dist_head(shape_features, training=training)
            
            # Assemble output
            output = {
                'prob': prob,
                'dist': dist,
                'sampling_points': sampling_points,
                'point_features': shape_output['point_features'],
                'complexity': complexity,
                'shape_descriptor': shape_output['shape_descriptor'],
                'features': shape_features,
                'fpn_features': fpn_features,  # Include FPN features for analysis
            }
            
            if prototype_weights is not None:
                output['prototype_weights'] = prototype_weights
            
            # Include learned prototypes for analysis
            if self.shape_prior_encoder is not None:
                output['learned_prototypes'] = prior_output.get('learned_prototypes')
                output['attention_maps'] = prior_output.get('attention_maps')
        
        return output
    
    def predict_instances(self, image, prob_thresh=0.5, nms_thresh=0.3):
        """
        Predict instance masks from image
        
        Parameters:
        -----------
        image : np.ndarray
            Input image
        prob_thresh : float
            Probability threshold for detection
        nms_thresh : float
            Non-maximum suppression threshold
            
        Returns:
        --------
        labels : np.ndarray
            Instance label image
        details : dict
            Additional prediction details
        """
        
        # Ensure image has batch and channel dimensions
        if image.ndim == 2:
            image = image[np.newaxis, ..., np.newaxis]
        elif image.ndim == 3:
            image = image[np.newaxis, ...]
        
        # Normalize image
        image = image.astype(np.float32)
        image = (image - image.mean()) / (image.std() + 1e-7)
        
        # Predict
        output = self(image, training=False)
        
        # Extract predictions
        prob = output['prob'][0, ..., 0].numpy()  # [H, W]
        dist = output['dist'][0].numpy()  # [H, W, N_points]
        sampling_points = output['sampling_points'][0].numpy()  # [N_points, 2]
        
        # Threshold probability map
        mask = prob > prob_thresh
        
        # Find instance centers (local maxima)
        from scipy.ndimage import label, maximum_filter
        
        local_max = (prob == maximum_filter(prob, size=5)) & mask
        labeled_centers, num_centers = label(local_max)
        
        # For each center, reconstruct shape using adaptive sampling points
        labels = np.zeros(prob.shape, dtype=np.int32)
        
        for i in range(1, num_centers + 1):
            center_coords = np.where(labeled_centers == i)
            if len(center_coords[0]) == 0:
                continue
            
            cy, cx = center_coords[0][0], center_coords[1][0]
            
            # Get distance predictions at this center
            center_dist = dist[cy, cx, :]  # [N_points]
            
            # Reconstruct boundary from sampling points and distances
            # (Simplified - full implementation would use the learned sampling positions)
            angles = np.linspace(0, 2*np.pi, len(center_dist), endpoint=False)
            boundary_y = cy + center_dist * np.sin(angles)
            boundary_x = cx + center_dist * np.cos(angles)
            
            # Create mask for this instance
            from skimage.draw import polygon
            rr, cc = polygon(boundary_y, boundary_x, shape=prob.shape)
            labels[rr, cc] = i
        
        details = {
            'prob': prob,
            'dist': dist,
            'sampling_points': sampling_points,
            'complexity': output['complexity'][0].numpy(),
        }
        
        return labels, details
    
    def train_model(self, X_train, Y_train, validation_data=None, augmenter=None):
        """
        Train the model
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training images
        Y_train : np.ndarray
            Training labels
        validation_data : tuple
            Validation data (X_val, Y_val)
        augmenter : callable
            Data augmentation function
        """
        
        from ..training.trainer import AdaptiveShapeTrainer
        
        trainer = AdaptiveShapeTrainer(self, self.config)
        history = trainer.train(
            X_train, Y_train,
            validation_data=validation_data,
            augmenter=augmenter
        )
        
        return history
    
    def save_model(self, path=None):
        """Save model weights and configuration"""
        if path is None:
            path = self.model_dir
        else:
            path = Path(path)
        
        path.mkdir(parents=True, exist_ok=True)
        
        # Save weights
        self.save_weights(str(path / 'model_weights.weights.h5'))
        
        # Save config
        import json
        with open(path / 'config.json', 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        print(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path, custom_objects=None):
        """Load model from path"""
        import json
        
        path = Path(path)
        
        # Load config
        with open(path / 'config.json', 'r') as f:
            config_dict = json.load(f)
        
        config = AdaptiveShapeConfig.from_dict(config_dict)
        
        # Create model
        model = cls(config)
        
        # Build model by running a dummy forward pass
        dummy_input = tf.zeros((1, 256, 256, config.n_channel_in))
        _ = model(dummy_input, training=False)
        
        # Load weights
        model.load_weights(str(path / 'model_weights.weights.h5'))
        
        print(f"Model loaded from {path}")
        
        return model
    
    def get_config(self):
        """Get model configuration"""
        return self.config.to_dict()

