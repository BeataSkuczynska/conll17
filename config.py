from keras.layers import GRU

params = {
    'epochs': 30,
    'rnn': GRU,
    'output_dim_rnn': 200,
    'activation_rnn': 'relu',
    'dropout': 0.5,
    'optimizer': 'adam',
    'model_name': 'generated/zabawa.h5',
    'predict_output_path': 'generated/output.conllu',
    'emb_size': 50
}
