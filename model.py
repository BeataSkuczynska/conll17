from keras import Input, Model
from keras.layers import Bidirectional, Reshape, Embedding, Concatenate
from keras.layers.core import Dense, Dropout

from constants import RELS_DICT, POS_DICT


def create_model(emb, maxlen, params):
    no_of_POS_features = len(POS_DICT) + 1
    pos_input = Input(shape=(maxlen, no_of_POS_features,), name='POS')
    poses = Reshape((maxlen, no_of_POS_features,))(pos_input)

    emb_input = Input(shape=(maxlen, params['emb_size'],), name='emb_input')
    embedding = Embedding(emb.shape[0], params['emb_size'], input_length=maxlen, weights=[emb],
                          trainable=params['trainable_embeddings'])(emb_input)
    concat = Concatenate(axis=-1)([emb_input, pos_input])

    lstm = Bidirectional(
        params['rnn'](units=params['output_dim_rnn'], activation=params['activation_rnn'], return_sequences=True),
        input_shape=(maxlen, no_of_POS_features + params['emb_size'],))(concat)

    dropout = Dropout(params['dropout'])(lstm)

    out1 = Dense(maxlen+1, activation='sigmoid', name='parents')(dropout)
    out2 = Dense(len(RELS_DICT) + 1, activation='sigmoid', name='relations')(dropout)

    model = Model(inputs=[emb_input, pos_input], outputs=[out1, out2])

    model.compile(optimizer=params['optimizer'], loss='binary_crossentropy', metrics=['acc'])
    from keras.utils import plot_model
    plot_model(model, to_file='generated/model.png', show_shapes=True)
    model.save(params['model_name'])
    return model
