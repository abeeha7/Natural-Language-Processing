import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

def preprocess_data(file_path, test_size=0.2, val_size=0.1):
    # Load the data from CSV
    data = pd.read_csv(file_path, encoding='utf-8')
    english_texts = data['English words/sentences'].apply(lambda x: x.lower())
    french_texts = data['French words/sentences'].apply(lambda x: x.lower())

    # Initialize tokenizers
    eng_tokenizer = Tokenizer()
    french_tokenizer = Tokenizer()

    # Fit tokenizers on the texts
    eng_tokenizer.fit_on_texts(english_texts)
    french_tokenizer.fit_on_texts(french_texts)

    # Convert texts to sequences
    eng_sequences = eng_tokenizer.texts_to_sequences(english_texts)
    french_sequences = french_tokenizer.texts_to_sequences(french_texts)

    # Pad sequences to ensure equal length
    eng_padded = pad_sequences(eng_sequences, padding='post')
    french_padded = pad_sequences(french_sequences, padding='post')

    # Get vocabulary sizes
    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    french_vocab_size = len(french_tokenizer.word_index) + 1

    # Split data into training and remaining sets (test + validation)
    X_train, X_remaining, y_train, y_remaining = train_test_split(eng_padded, french_padded, test_size=(test_size + val_size), random_state=42)

    # Split the remaining data into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(X_remaining, y_remaining, test_size=(test_size / (test_size + val_size)), random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test, eng_tokenizer, french_tokenizer, eng_vocab_size, french_vocab_size


def build_and_train_model(X_train, y_train, X_val, y_val, eng_vocab_size, french_vocab_size, max_eng_length, max_french_length, batch_size=64, epochs=1, embedding_dim=256):
    """
    Build and train an encoder-decoder model with LSTM layers, including teacher forcing.
    
    Parameters:
    X_train: The training data for the English sentences (input).
    y_train: The training data for the French sentences (output).
    X_val: The validation data for the English sentences (input).
    y_val: The validation data for the French sentences (output).
    eng_vocab_size: Vocabulary size for the English language.
    french_vocab_size: Vocabulary size for the French language.
    max_eng_length: Maximum length of the English sentences.
    max_french_length: Maximum length of the French sentences.
    batch_size: Batch size for training.
    epochs: Number of epochs for training.
    embedding_dim: Dimension of the embedding layer.
    """
    
    # Define the Encoder
    encoder_inputs = layers.Input(shape=(max_eng_length,))
    enc_emb = layers.Embedding(eng_vocab_size, embedding_dim)(encoder_inputs)
    encoder_lstm = layers.LSTM(256, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
    encoder_states = [state_h, state_c]
    
    # Define the Decoder
    decoder_inputs = layers.Input(shape=(max_french_length - 1,))
    dec_emb_layer = layers.Embedding(french_vocab_size, embedding_dim)
    dec_emb = dec_emb_layer(decoder_inputs)
    
    # LSTM decoder with the encoder's state as initial state
    decoder_lstm = layers.LSTM(256, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
    
    # Output layer with softmax to predict next word
    decoder_dense = layers.Dense(french_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    # Compile the model with Adam optimizer and sparse categorical crossentropy loss
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Define the teacher forcing mechanism
    # Teacher forcing: During training, feed the true previous word as the next input.
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        
        # Training loop with teacher forcing
        for batch in range(0, len(X_train), batch_size):
            # Get a batch of data
            batch_X = X_train[batch:batch + batch_size]
            batch_y = y_train[batch:batch + batch_size]
            batch_X_val = X_val[batch:batch + batch_size]
            batch_y_val = y_val[batch:batch + batch_size]
            
            # Ensure the decoder input and target sequence are correctly shaped
            # Here we're using the same y as both target sequence and labels for teacher forcing
            decoder_input_data = batch_y[:, :-1]
            decoder_target_data = batch_y[:, 1:]
            
            # Train the model
            model.fit([batch_X, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=1, validation_data=([batch_X_val, batch_y_val[:, :-1]], batch_y_val[:, 1:]),verbose=1)
        
        # Save the model after each epoch if desired
        model.save(f'model_epoch_{epoch + 1}.h5')
    
    return model

file_path = 'eng-french.csv'
X_train, X_val, X_test, y_train, y_val, y_test, eng_tokenizer, french_tokenizer, eng_vocab_size, french_vocab_size = preprocess_data(file_path)
max_eng_length = X_train.shape[1]
max_french_length = y_train.shape[1]
model = build_and_train_model(X_train, y_train, X_val, y_val, eng_vocab_size, french_vocab_size,max_eng_length, max_french_length, batch_size=64, epochs=20, embedding_dim=256)


