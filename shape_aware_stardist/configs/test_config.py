"""Test configuration for Shape-aware StarDist"""

TEST_CONFIG = {
    # Model Configuration
    'model': {
        'in_channels': 3,
        'n_rays': 32,
        'base_channels': 64,
        'n_blocks': 4,
        'use_shape_prior': True
    },
    
    # Training Configuration
    'training': {
        'batch_size': 4,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'validation_interval': 1
    },
    
    # Loss Configuration
    'loss': {
        'consistency_weight': 1.0,
        'smoothness_weight': 0.5,
        'adversarial_weight': 0.1,
        'use_focal_loss': True,
        'focal_gamma': 2.0,
        'use_adaptive_weights': True
    },
    
    # Data Configuration
    'data': {
        'input_size': (512, 512),
        'min_object_size': 100,
        'max_object_size': None,
        'augmentation': {
            'rotation_range': (-180, 180),
            'scale_range': (0.8, 1.2),
            'flip_probability': 0.5
        }
    },
    
    # Testing Configuration
    'testing': {
        'test_batch_size': 1,
        'save_predictions': True,
        'visualization': {
            'save_attention_maps': True,
            'save_shape_analysis': True
        }
    }
}
