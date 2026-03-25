"""
Backbone Networks
=================

Provides various backbone architectures for feature extraction.

Available Backbones:
- UNet: Standard U-Net architecture
- ResNet: Residual network with skip connections
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class UNetBackbone(keras.Model):
    """
    U-Net backbone for feature extraction
    
    Parameters:
    -----------
    n_depth : int
        Number of down/up-sampling levels
    n_filter_base : int
        Base number of filters (doubled at each level)
    kernel_size : int
        Convolution kernel size
    n_channel_in : int
        Number of input channels
    """
    
    def __init__(
        self,
        n_depth=3,
        n_filter_base=32,
        kernel_size=3,
        n_channel_in=1,
        **kwargs
    ):
        super(UNetBackbone, self).__init__(**kwargs)
        
        self.n_depth = n_depth
        self.n_filter_base = n_filter_base
        self.kernel_size = kernel_size
        self.n_channel_in = n_channel_in
        
        # Build encoder path
        self.encoder_blocks = []
        self.pool_layers = []
        
        for i in range(n_depth):
            n_filters = n_filter_base * (2 ** i)
            
            # Double convolution block
            block = keras.Sequential([
                layers.Conv2D(n_filters, kernel_size, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv2D(n_filters, kernel_size, activation='relu', padding='same'),
                layers.BatchNormalization(),
            ], name=f'encoder_block_{i}')
            
            self.encoder_blocks.append(block)
            
            if i < n_depth - 1:
                self.pool_layers.append(layers.MaxPooling2D(pool_size=2))
        
        # Build decoder path
        self.decoder_blocks = []
        self.upconv_layers = []
        
        for i in range(n_depth - 1):
            n_filters = n_filter_base * (2 ** (n_depth - 2 - i))
            
            # Upsampling
            upconv = layers.Conv2DTranspose(
                n_filters, 
                kernel_size=2, 
                strides=2, 
                padding='same',
                name=f'upconv_{i}'
            )
            self.upconv_layers.append(upconv)
            
            # Double convolution block
            block = keras.Sequential([
                layers.Conv2D(n_filters, kernel_size, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv2D(n_filters, kernel_size, activation='relu', padding='same'),
                layers.BatchNormalization(),
            ], name=f'decoder_block_{i}')
            
            self.decoder_blocks.append(block)
    
    def call(self, inputs, training=None):
        """
        Forward pass
        
        Returns:
        --------
        output : dict
            Dictionary containing:
            - 'features': Final feature map
            - 'encoder_features': List of encoder features (for skip connections)
            - 'decoder_features': List of decoder features
        """
        encoder_features = []
        
        # Encoder path
        x = inputs
        for i, block in enumerate(self.encoder_blocks):
            x = block(x, training=training)
            encoder_features.append(x)
            
            if i < len(self.pool_layers):
                x = self.pool_layers[i](x)
        
        # Decoder path
        decoder_features = []
        for i, (upconv, block) in enumerate(zip(self.upconv_layers, self.decoder_blocks)):
            x = upconv(x)
            
            # Skip connection from encoder
            skip_features = encoder_features[-(i + 2)]
            
            # Crop skip features if needed to match size (graph-compatible)
            target_shape = tf.shape(x)[1:3]
            skip_shape = tf.shape(skip_features)[1:3]
            
            # Use tf.cond for conditional resize (graph-compatible)
            needs_resize = tf.logical_or(
                tf.not_equal(skip_shape[0], target_shape[0]),
                tf.not_equal(skip_shape[1], target_shape[1])
            )
            
            skip_features = tf.cond(
                needs_resize,
                lambda: tf.image.resize(skip_features, target_shape),
                lambda: skip_features
            )
            
            # Concatenate
            x = tf.concat([x, skip_features], axis=-1)
            
            # Apply convolutions
            x = block(x, training=training)
            decoder_features.append(x)
        
        return {
            'features': x,
            'encoder_features': encoder_features,
            'decoder_features': decoder_features,
        }
    
    def get_config(self):
        return {
            'n_depth': self.n_depth,
            'n_filter_base': self.n_filter_base,
            'kernel_size': self.kernel_size,
            'n_channel_in': self.n_channel_in,
        }


class ResNetBackbone(keras.Model):
    """
    ResNet backbone for feature extraction
    
    Parameters:
    -----------
    n_blocks : int
        Number of residual blocks
    n_filter_base : int
        Base number of filters
    kernel_size : int
        Convolution kernel size
    n_channel_in : int
        Number of input channels
    """
    
    def __init__(
        self,
        n_blocks=4,
        n_filter_base=64,
        kernel_size=3,
        n_channel_in=1,
        **kwargs
    ):
        super(ResNetBackbone, self).__init__(**kwargs)
        
        self.n_blocks = n_blocks
        self.n_filter_base = n_filter_base
        self.kernel_size = kernel_size
        self.n_channel_in = n_channel_in
        
        # Initial convolution
        self.initial_conv = keras.Sequential([
            layers.Conv2D(n_filter_base, 7, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=3, strides=2, padding='same'),
        ], name='initial_conv')
        
        # Residual blocks
        self.res_blocks = []
        for i in range(n_blocks):
            n_filters = n_filter_base * (2 ** i)
            stride = 2 if i > 0 else 1
            
            block = ResidualBlock(
                n_filters=n_filters,
                kernel_size=kernel_size,
                stride=stride,
                name=f'res_block_{i}'
            )
            self.res_blocks.append(block)
    
    def call(self, inputs, training=None):
        """Forward pass"""
        features_list = []
        
        x = self.initial_conv(inputs, training=training)
        features_list.append(x)
        
        for block in self.res_blocks:
            x = block(x, training=training)
            features_list.append(x)
        
        return {
            'features': x,
            'multi_scale_features': features_list,
        }
    
    def get_config(self):
        return {
            'n_blocks': self.n_blocks,
            'n_filter_base': self.n_filter_base,
            'kernel_size': self.kernel_size,
            'n_channel_in': self.n_channel_in,
        }


class ResidualBlock(layers.Layer):
    """
    Residual block with skip connection
    """
    
    def __init__(self, n_filters, kernel_size=3, stride=1, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
    
    def build(self, input_shape):
        # Main path
        self.conv1 = layers.Conv2D(
            self.n_filters, 
            self.kernel_size, 
            strides=self.stride,
            padding='same'
        )
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        
        self.conv2 = layers.Conv2D(
            self.n_filters, 
            self.kernel_size,
            padding='same'
        )
        self.bn2 = layers.BatchNormalization()
        
        # Skip connection
        if self.stride != 1 or input_shape[-1] != self.n_filters:
            self.skip_conv = layers.Conv2D(
                self.n_filters,
                kernel_size=1,
                strides=self.stride,
                padding='same'
            )
            self.skip_bn = layers.BatchNormalization()
        else:
            self.skip_conv = None
        
        self.relu2 = layers.ReLU()
        
        super(ResidualBlock, self).build(input_shape)
    
    def call(self, inputs, training=None):
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Skip connection
        if self.skip_conv is not None:
            skip = self.skip_conv(inputs)
            skip = self.skip_bn(skip, training=training)
        else:
            skip = inputs
        
        # Add and activate
        x = x + skip
        x = self.relu2(x)
        
        return x
    
    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({
            'n_filters': self.n_filters,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
        })
        return config

