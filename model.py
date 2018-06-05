from keras import Input, Model
from keras.layers import Bidirectional, Reshape
from keras.layers.core import Dense, Dropout

from constants import RELS_DICT, POS_DICT


def create_model(maxlen, params):
    # maxlen = 38
    no_of_input_features = len(POS_DICT) + 1
    pos_input = Input(shape=(maxlen, no_of_input_features,), name='pos')
    poses = Reshape((maxlen, no_of_input_features,))(pos_input)
    lstm = Bidirectional(
        params['rnn'](units=params['output_dim_rnn'], activation=params['activation_rnn'], return_sequences=True),
        input_shape=(maxlen, no_of_input_features,))(poses)
    dropout = Dropout(params['dropout'])(lstm)
    out1 = Dense(maxlen+1, activation='sigmoid')(dropout)
    out2 = Dense(len(RELS_DICT) + 1, activation='sigmoid')(dropout)

    model = Model(inputs=[pos_input], outputs=[out1, out2])

    model.compile(optimizer=params['optimizer'], loss='binary_crossentropy', metrics=['acc'])
    from keras.utils import plot_model
    plot_model(model, to_file='generated/model.png', show_shapes=True)
    model.save('generated/model.h5')
    return model
