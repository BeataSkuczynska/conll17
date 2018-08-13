from keras.layers import GRU

params = {
    'epochs': 8,
    'rnn': GRU,
    'output_dim_rnn': 200,
    'activation_rnn': 'relu',
    'dropout': 0.5,
    'optimizer': 'adam',
    'model_name': 'generated/zabawa.h5',
    'predict_output_path': 'generated/output_testix.conllu',
    'emb_size': 300,
    'trainable_embeddings': True
}
