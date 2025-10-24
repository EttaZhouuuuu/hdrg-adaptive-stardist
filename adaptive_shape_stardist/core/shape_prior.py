"""
Shape Prior Encoder
===================

Encodes domain-specific shape knowledge to guide the adaptive shape encoder.
This incorporates biological priors about cell morphology.

Key Features:
- Transformer-based global shape reasoning
- Shape prototype learning
- Multi-head attention for shape-feature fusion
- Learnable shape embeddings
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class ShapePriorEncoder(layers.Layer):
    """
    Shape Prior Encoder using Transformer architecture
    
    Learns and encodes domain-specific shape priors from the dataset.
    Uses attention mechanisms to capture global shape characteristics.
    
    Parameters:
    -----------
    num_prototypes : int
        Number of learnable shape prototypes
    embedding_dim : int
        Dimension of shape embeddings
    num_heads : int
        Number of attention heads
    num_layers : int
        Number of transformer layers
    """
    
    def __init__(
        self,
        num_prototypes=16,
        embedding_dim=256,
        num_heads=8,
        num_layers=3,
        dropout_rate=0.1,
        **kwargs
    ):
        super(ShapePriorEncoder, self).__init__(**kwargs)
        
        self.num_prototypes = num_prototypes
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        """Build shape prior components"""
        
        # Learnable shape prototypes (embeddings)
        self.shape_prototypes = self.add_weight(
            name='shape_prototypes',
            shape=(self.num_prototypes, self.embedding_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Feature projection to embedding space
        self.feature_projection = keras.Sequential([
            layers.Conv2D(self.embedding_dim, 1, activation='relu'),
            layers.BatchNormalization(),
        ], name='feature_projection')
        
        # Positional encoding for spatial features
        self.positional_encoding = PositionalEncoding2D(
            embedding_dim=self.embedding_dim,
            name='positional_encoding'
        )
        
        # Transformer encoder layers
        self.transformer_layers = []
        for i in range(self.num_layers):
            layer = TransformerBlock(
                embedding_dim=self.embedding_dim,
                num_heads=self.num_heads,
                ff_dim=self.embedding_dim * 4,
                dropout_rate=self.dropout_rate,
                name=f'transformer_{i}'
            )
            self.transformer_layers.append(layer)
        
        # Shape-feature fusion with cross-attention
        self.cross_attention = CrossAttentionFusion(
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            name='cross_attention'
        )
        
        # Output projection
        self.output_projection = keras.Sequential([
            layers.Dense(self.embedding_dim, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.embedding_dim),
        ], name='output_projection')
        
        super(ShapePriorEncoder, self).build(input_shape)
    
    def call(self, inputs, training=None):
        """
        Encode shape priors and fuse with input features
        
        Parameters:
        -----------
        inputs : tf.Tensor or dict
            If tensor: Input feature map [B, H, W, C]
            If dict: Must contain 'features' and optionally 'shape_hints'
        training : bool
            Training mode flag
            
        Returns:
        --------
        output : dict
            Dictionary containing:
            - 'prior_features': Features enhanced with shape priors [B, H, W, C]
            - 'prototype_weights': Weights for each prototype [B, num_prototypes]
            - 'attention_maps': Attention maps [B, H, W, num_heads]
        """
        
        # Handle different input formats
        if isinstance(inputs, dict):
            features = inputs['features']
            shape_hints = inputs.get('shape_hints', None)
        else:
            features = inputs
            shape_hints = None
        
        batch_size = tf.shape(features)[0]
        height = tf.shape(features)[1]
        width = tf.shape(features)[2]
        
        # Project features to embedding space
        embedded_features = self.feature_projection(features, training=training)
        
        # Add positional encoding
        embedded_features = self.positional_encoding(embedded_features)
        
        # Reshape features for transformer: [B, H*W, C]
        features_flat = tf.reshape(
            embedded_features,
            [batch_size, height * width, self.embedding_dim]
        )
        
        # Expand prototypes for batch: [B, num_prototypes, C]
        prototypes = tf.tile(
            tf.expand_dims(self.shape_prototypes, 0),
            [batch_size, 1, 1]
        )
        
        # Concatenate prototypes with features
        # This allows prototypes to interact with features
        combined = tf.concat([prototypes, features_flat], axis=1)
        
        # Apply transformer layers
        transformed = combined
        for transformer in self.transformer_layers:
            transformed = transformer(transformed, training=training)
        
        # Split back into prototypes and features
        transformed_prototypes = transformed[:, :self.num_prototypes, :]
        transformed_features = transformed[:, self.num_prototypes:, :]
        
        # Cross-attention fusion between prototypes and features
        fused_features, attention_maps = self.cross_attention(
            queries=transformed_features,
            keys=transformed_prototypes,
            values=transformed_prototypes,
            training=training
        )
        
        # Compute prototype weights (how much each prototype is used)
        prototype_weights = tf.reduce_mean(attention_maps, axis=1)  # [B, num_prototypes]
        
        # Project to output space
        output_features = self.output_projection(fused_features, training=training)
        
        # Reshape back to spatial format: [B, H, W, C]
        output_features = tf.reshape(
            output_features,
            [batch_size, height, width, self.embedding_dim]
        )
        
        # Compute attention visualization (average across heads)
        attention_viz = tf.reshape(
            tf.reduce_mean(attention_maps, axis=-1),  # Average over prototypes
            [batch_size, height, width, 1]
        )
        
        return {
            'prior_features': output_features,  # [B, H, W, C]
            'prototype_weights': prototype_weights,  # [B, num_prototypes]
            'attention_maps': attention_viz,  # [B, H, W, 1]
            'learned_prototypes': transformed_prototypes,  # [B, num_prototypes, C]
        }
    
    def get_config(self):
        """Get configuration"""
        config = super(ShapePriorEncoder, self).get_config()
        config.update({
            'num_prototypes': self.num_prototypes,
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
        })
        return config


class TransformerBlock(layers.Layer):
    """
    Standard Transformer encoder block
    
    Components:
    - Multi-head self-attention
    - Feed-forward network
    - Layer normalization
    - Residual connections
    """
    
    def __init__(self, embedding_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
    
    def build(self, input_shape):
        # Multi-head self-attention
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embedding_dim // self.num_heads,
            name='mha'
        )
        
        # Feed-forward network
        self.ffn = keras.Sequential([
            layers.Dense(self.ff_dim, activation='gelu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.embedding_dim),
        ], name='ffn')
        
        # Layer normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6, name='ln1')
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6, name='ln2')
        
        # Dropout
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)
        
        super(TransformerBlock, self).build(input_shape)
    
    def call(self, inputs, training=None):
        # Self-attention with residual
        attn_output = self.attention(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward with residual
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
        })
        return config


class CrossAttentionFusion(layers.Layer):
    """
    Cross-attention mechanism for fusing shape priors with features
    """
    
    def __init__(self, embedding_dim, num_heads, **kwargs):
        super(CrossAttentionFusion, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
    
    def build(self, input_shape):
        self.cross_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embedding_dim // self.num_heads,
            name='cross_mha'
        )
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        super(CrossAttentionFusion, self).build(input_shape)
    
    def call(self, queries, keys, values, training=None):
        """
        Apply cross-attention
        
        Parameters:
        -----------
        queries : tf.Tensor
            Query features [B, N, C]
        keys : tf.Tensor
            Key features [B, M, C]
        values : tf.Tensor
            Value features [B, M, C]
            
        Returns:
        --------
        output : tf.Tensor
            Fused features [B, N, C]
        attention_weights : tf.Tensor
            Attention weights [B, N, M]
        """
        # Apply cross-attention
        attn_output, attn_weights = self.cross_attention(
            query=queries,
            key=keys,
            value=values,
            return_attention_scores=True,
            training=training
        )
        
        # Residual connection and layer norm
        output = self.layernorm(queries + attn_output)
        
        return output, attn_weights
    
    def get_config(self):
        config = super(CrossAttentionFusion, self).get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
        })
        return config


class PositionalEncoding2D(layers.Layer):
    """
    2D positional encoding for spatial feature maps
    
    Adds sinusoidal position information to feature maps
    to help the network understand spatial relationships.
    """
    
    def __init__(self, embedding_dim, max_height=512, max_width=512, **kwargs):
        super(PositionalEncoding2D, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.max_height = max_height
        self.max_width = max_width
    
    def build(self, input_shape):
        # Create positional encoding lookup table
        self.pos_encoding = self._create_positional_encoding()
        super(PositionalEncoding2D, self).build(input_shape)
    
    def _create_positional_encoding(self):
        """Create 2D sinusoidal positional encoding"""
        # Create position indices
        y_pos = np.arange(self.max_height)[:, np.newaxis]
        x_pos = np.arange(self.max_width)[np.newaxis, :]
        
        # Create frequency bands
        div_term = np.exp(
            np.arange(0, self.embedding_dim, 2) * 
            -(np.log(10000.0) / self.embedding_dim)
        )
        
        # Allocate encoding array
        pos_encoding = np.zeros((self.max_height, self.max_width, self.embedding_dim))
        
        # Apply sine to even indices
        pos_encoding[:, :, 0::2] = np.sin(
            y_pos[:, :, np.newaxis] * div_term
        ) + np.sin(
            x_pos[:, :, np.newaxis] * div_term
        )
        
        # Apply cosine to odd indices
        if self.embedding_dim > 1:
            pos_encoding[:, :, 1::2] = np.cos(
                y_pos[:, :, np.newaxis] * div_term
            ) + np.cos(
                x_pos[:, :, np.newaxis] * div_term
            )
        
        return tf.constant(pos_encoding, dtype=tf.float32)
    
    def call(self, inputs):
        """Add positional encoding to inputs"""
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        
        # Crop positional encoding to match input size
        pos_enc = self.pos_encoding[:height, :width, :]
        
        # Add to input
        return inputs + pos_enc
    
    def get_config(self):
        config = super(PositionalEncoding2D, self).get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
            'max_height': self.max_height,
            'max_width': self.max_width,
        })
        return config

