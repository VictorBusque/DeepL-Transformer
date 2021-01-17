import os
import random
import re
import string
from typing import Dict, Tuple, List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
This file contains implementations for several building blocks for a Transformer decoder according to
the "Attention is all you need" paper by Ashish Vaswani et. al. (https://arxiv.org/pdf/1706.03762.pdf)

The Transformer block (which is a Transformer decoder), does not include the encoder-decoder Multi-Head attention layer
since this implementation won't be used in encoder-decoder architectures.

Starting from code available at: https://keras.io/examples/generative/text_generation_with_miniature_gpt/

Working tf version: 1.15.0
Author: Víctor Busqué
"""


class TokenAndPositionEmbedding(layers.Layer):
    """tf.keras.layer.Layer extension that represents a fusion of the token + position embedding layers for transformers.

    Attributes:
        token_emb (tf.keras.layers.Embedding): Embedding layer that encodes tokens of a vocabulary to a vector of some dimensionality.
        pos_emb (tf.keras.layers.Embedding): Embedding layer that encodes a token position of a vocabulary to a vector of some dimensionality.

    Note:
        token_emb and pos_emb resulting embeddings dimensionality must be the same.
    """
    def __init__(self, 
                max_seq_length: int, 
                vocab_size: int, 
                embed_dim: int):
        """Instantiates a TokenAndPositionEmbedding layer.

        Args:
            max_seq_length (int): Length of the longest sentence that the model should process.
            vocab_size (int): Amount of different tokens that the model should learn.
            embed_dim (int): Dimensionality of the corresponding embeddings.
        """
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=max_seq_length, output_dim=embed_dim)

    def get_config(self) -> Dict:
        """Generates the layer's configuration so that it can be saved by tfs.keras.model.Model afterwads.

        Returns:
            Dict: Configuration of the layer (i.e the attributes) so that they can be saved.
        """
        config = super().get_config().copy()
        config.update({
            "token_emb": self.token_emb,
            "pos_emb": self.pos_emb
        })
        return config

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Computes token and position embeddings and returns a semantic representation for each word.

        Args:
            x (tf.Tensor): Input vector. It is normally a sparse vector from a one-hot representation (i.e. [1,5,3,12,4], ids for words).

        Returns:
            tf.Tensor: Vector representing each word semantically + encoding its position in the sentence.
        """
        max_seq_length = tf.shape(x)[-1]
        
        # Positions is the position embedding input vector, which is a sorted list of indexes (i.e. [0,1,2,3,4,5,...]) 
        positions = tf.range(start=0, limit=max_seq_length, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)

        # Bitwise addition, not concatenation.
        return x + positions


class MultiHeadSelfAttention(layers.Layer):
    """tf.keras.layer.Layer for multi-head self-attention.

    Attributes:
        embed_dim (int): Dimensionality for intermediate representations and for the final context-sensitive token representation.
        num_heads (int): How many heads will be used.
        use_mask (bool): Defines whether we use future-token masking while computing attention.
        projection_dim (int): Dimensionality for the Q,V,K projections.
        query_dense (tf.keras.layers.Dense): Q matrix weights.
        key_dense (tf.keras.layers.Dense): K matrix weights.
        value_dense (tf.keras.layers.Dense): V matrix weights.
        combine_heads (tf.keras.layers.Dense): Layer that will combine results to an embed_dim dimensionality.
    """
    def __init__(self, 
                embed_dim: int, 
                num_heads: int=8, 
                use_mask: bool=True):
        """Instantiates a Multi-Head Self-Attention layer.

        Args:
            embed_dim (int): Dimensionality for intermediate representations and for the final context-sensitive token representation.
            num_heads (int, optional): How many heads will be used.. Defaults to 8.
            use_mask (bool, optional): Defines whether we use future-token masking while computing attention.

        Raises:
            ValueError: When num_heads is not divisible by embed_dim to compute the projection dimensionality.
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_mask = use_mask
        if embed_dim % num_heads != 0:
            raise ValueError(f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}")
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def get_config(self) -> Dict:
        """Generates the layer's configuration so that it can be saved by tf.keras.model.Model afterwads.

        Returns:
            Dict: Configuration of the layer (i.e the attributes) so that they can be saved.
        """
        config = super().get_config().copy()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "use_mask": self.use_mask,
            "projection_dim": self.projection_dim,
            "query_dense": self.query_dense,
            "key_dense": self.key_dense,
            "value_dense": self.value_dense,
            "combine_heads": self.combine_heads
        })
        return config

    @staticmethod
    def causal_attention_mask(n_dest: int, n_src: int) -> tf.Tensor:
        """Performs a causal attention mask so that future tokens are not taken into account.

        Args:
            n_dest (int): Number of columns in the matrix.
            n_src (int): Number of rows in the matrix.

        Returns:
            tf.Tensor: Matrix (n_src x n_dest) with 1's in the lower triangle, counting from the lower right corner.
                        Mask the upper half of the dot product matrix in self attention.
                        This prevents flow of information from future tokens to current token.
                        
        """
        # i = tf.range(n_dest)[:, None]
        # j = tf.range(n_src)
        # m = i >= j - n_src + n_dest
        # return tf.cast(m, dtype)
        return tf.linalg.band_part(tf.ones((n_src, n_dest)), -1, 0)

    def attention(self, 
                query: tf.Tensor, 
                key: tf.Tensor, 
                value: tf.Tensor
                ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Performs Scaled Dot-Product Attention. (https://paperswithcode.com/media/methods/SCALDE.png)

        Attention(Q, K, V) = softmax( (QK^T) / sqrt(d_k) ) * V

        Args:
            query (tf.Tensor): Query input vector (input -> query_dense = query). Dimensionality: (batch_size, seq_len, embed_dim)
            key (tf.Tensor): Key input vector (input -> key_dense = key). Dimensionality: (batch_size, seq_len, embed_dim)
            value (tf.Tensor): Query input vector (input -> value_dense = value). Dimensionality: (batch_size, seq_len, embed_dim)

        Returns:
            Tuple[tf.Tensor, tf.Tensor]:
                - Tensor containing a per-token representation that is context-aware. Dimensionality: (batch_size, seq_len, embed_dim)
                - Tensor containing a per-token influence value (attention weights). Dimensionality: (batch_size, seq_len, seq_len, (1))
        """
        # MatMul(Q, K^T)
        score = tf.matmul(query, key, transpose_b=True)
        
        # Scale
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)

        # Mask (opt)
        if self.use_mask:
            shape = tf.shape(scaled_score)
            dim_dest, dim_src = shape[2], shape[3]
            attention_mask = self.causal_attention_mask(dim_dest, dim_src)
            attention_mask = tf.reshape(attention_mask, [1, 1, dim_dest, dim_src])
            scaled_score = scaled_score * attention_mask - 1e4 * (1 - attention_mask)

        # SoftMax
        weights = tf.nn.softmax(scaled_score, axis=-1)

        #MatMul (W, V)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        """For an input tensor, splits the last dimension (embed_dim) to a matrix of (num_heads, projection_dim).

        Args:
            x (tf.Tensor): Input tensor to be separated. Dimensionality: (batch_size, seq_len, embed_dim)
            batch_size (int): Batch size used for training/predictions.

        Returns:
            tf.Tensor: Input Tensor with a new dimension. Dimensionality: (batch_size, num_heads, seq_len, projection_dim)
        """
        # Turn (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, num_heads, projection_dim)
        x = tf.reshape(tensor=x, shape=(batch_size, -1, self.num_heads, self.projection_dim))

        # Permutate dimensions: (batch_size, seq_len, num_heads, projection_dim) -> (batch_size, num_heads, seq_len, projection_dim)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Performs Multi-Head self-attention. (https://paperswithcode.com/media/methods/multi-head-attention_l1A3G7a.png)
        
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h)*W^O
            where head_i = Attention( Q*(W_i^Q), K*(W_i^K), V*(W_i^V) )


        Args:
            inputs (tf.Tensor): Input tensor to the multi-head self-attention layer (it's usually the output from an Token&Position embedding).
                                Dimensionality: (batch_size, seq_len, embedding_dim)

        Returns:
            tf.Tensor: The result of applying multi-head self-attention on the input
        """
        batch_size = tf.shape(inputs)[0]

        # Compute Q, K, V
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)

        # Split last dimension in different heads
        query = self.separate_heads(query, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(key, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(value, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)

        # Scaled Dot-Product Attention
        attention, weights = self.attention(query, key, value)
        # Undo permutation done when splitting heads: (batch_size, num_heads, seq_len, projection_dim) -> (batch_size, seq_len, num_heads, projection_dim).
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, projection_dim)

        # Concatenation of heads
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))  # (batch_size, seq_len, embed_dim)

        # Linear
        output = self.combine_heads(concat_attention)  # (batch_size, seq_len, embed_dim)
        # return output, weights # To see what was attended to.
        return output 


class TransformerBlock(layers.Layer):
    """tf.keras.layer.Layer that represents a transformer  decoder block. (https://deepfrench.gitlab.io/deep-learning-project/resources/transformer.png, right-side)

    Besides Masked Multi-Head Self-Attention, this includes a Feed-Forward net, that computes:
    FFN(x) = max(0, xW1 + b1)W2 + b2 

    Attributes:
        mmhsa (MultiHeadSelfAttention): Masked Multi-Head Self-Attention block.
        mmhsa_dropout (tf.keras.layers.Dropout): Dropout layer for mmhsa.
        add_and_norm_1 (tf.keras.layers.LayerNormalization): Add & Norm layer for mmhsa.
        ffn (List[tf.keras.layers.Dense]): Position-wise Feed-Forward Network.
        ffn_dropout (tf.keras.layers.Dropout): Dropout layer for the position wise ffn.
        add_and_norm_2 (tf.keras.layers.LayerNormalization): Add & Norm layer for ffn.
    """
    def __init__(self, 
                embed_dim: int, 
                num_heads: int, 
                ff_dim: int, 
                dropout_rate: float=0.1,
                bidirectional: bool=False):
        """Instantiates a Transformer Decoder Block.

        Args:
            embed_dim (int): Dimensionality of the embeddings.
            num_heads (int): Number of heads for multi-head attention.
            ff_dim (int): Dimensionality for the FFN. Defaults to 4*embed_dim, as used in the original paper.
            dropout_rate (float, optional): Defines the dropout rate for both dropout layers. Defaults to 0.1.
        """
        super(TransformerBlock, self).__init__()
        self.mmhsa = MultiHeadSelfAttention(embed_dim, num_heads, use_mask=not bidirectional)
        self.mmhsa_dropout = layers.Dropout(dropout_rate)
        self.add_and_norm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.ffn_dropout = layers.Dropout(dropout_rate)
        self.add_and_norm_2 = layers.LayerNormalization(epsilon=1e-6)

    def get_config(self) -> Dict:
        """Generates the layer's configuration so that it can be saved by tf.keras.model.Model afterwads.

        Returns:
            Dict: Configuration of the layer (i.e the attributes) so that they can be saved.
        """
        config = super().get_config().copy()
        config.update({
            "mmhsa": self.mmhsa,
            "ffn": self.ffn,
            "add_and_norm_1": self.add_and_norm_1,
            "add_and_norm_2": self.add_and_norm_2,
            "mmhsa_dropout": self.mmhsa_dropout,
            "ffn_dropout": self.ffn_dropout
        })
        return config

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Performs a pass into the transformer's decoder block.

        Args:
            inputs (tf.Tensor): Input to be passed into the block.

        Returns:
            tf.Tensor: Output of the pass.
        """
        attention_output = self.mmhsa(inputs)
        attention_output = self.mmhsa_dropout(attention_output)
        out1 = self.add_and_norm_1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.ffn_dropout(ffn_output)
        return self.add_and_norm_2(out1 + ffn_output)
