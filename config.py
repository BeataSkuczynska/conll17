from keras.layers import GRU

params = {
    'rnn': GRU,
    'output_dim_rnn': 200,
    'activation_rnn': 'relu',
    'dropout': 0.5,
    'optimizer': 'adam',
    'trainable_embeddings': True

}
