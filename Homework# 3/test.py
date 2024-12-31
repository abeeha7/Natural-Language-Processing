import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LayerNormalization, Attention, Add, Dense, Dropout
from tensorflow.keras.models import Model
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LayerNormalization, MultiHeadAttention, Add, Dense, Dropout
from tensorflow.keras.models import Model
import numpy as np

# Function to create a causal mask for decoder
def create_causal_mask(seq_len):
    mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
    return tf.convert_to_tensor(mask)

def build_improved_transformer(input_vocab_size, target_vocab_size, input_seq_len, target_seq_len, embedding_dim=512, num_heads=8, ff_dim=512, num_layers=4):
    # Encoder
    encoder_inputs = Input(shape=(input_seq_len,))
    encoder_embeddings = Embedding(input_vocab_size, embedding_dim)(encoder_inputs)
    positional_encoding = get_positional_encoding(input_seq_len, embedding_dim)  # Positional Encoding
    encoder_embeddings += positional_encoding  # Adding Positional Encoding to Embeddings
    encoder_output = encoder_embeddings

    for _ in range(num_layers):  # Transformer layers
        encoder_output = LayerNormalization()(encoder_output)
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(encoder_output, encoder_output)
        attention_output = Add()([encoder_output, attention_output])  # Residual connection
        encoder_output = Dense(embedding_dim, activation='relu')(attention_output)  # Aligns with embedding_dim
        encoder_output = Dropout(0.2)(encoder_output)

    # Decoder
    decoder_inputs = Input(shape=(target_seq_len,))
    decoder_embeddings = Embedding(target_vocab_size, embedding_dim)(decoder_inputs)
    positional_encoding_dec = get_positional_encoding(target_seq_len, embedding_dim)  # Positional Encoding for Decoder
    decoder_embeddings += positional_encoding_dec  # Adding Positional Encoding to Decoder Embeddings
    decoder_output = decoder_embeddings

    # Create the causal mask for the decoder
    causal_mask = create_causal_mask(target_seq_len)

    for _ in range(num_layers):  # Transformer layers with masking
        decoder_output = LayerNormalization()(decoder_output)
        # Apply masked attention using MultiHeadAttention with attention_mask
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(decoder_output, decoder_output, attention_mask=causal_mask)
        attention_output = Add()([decoder_output, attention_output])  # Residual connection
        decoder_output = Dense(embedding_dim, activation='relu')(attention_output)  # Aligns with embedding_dim
        decoder_output = Dropout(0.2)(decoder_output)

    # Output Layer
    final_output = Dense(target_vocab_size, activation='softmax')(decoder_output)

    transformer_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=final_output)
    return transformer_model