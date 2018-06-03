from keras import Input, Model
from keras.layers import Bidirectional, Concatenate, Reshape, Flatten
from keras.layers.core import Dense, Dropout


def create_model(maxlen, params):
    # maxlen = 38
    pos_input = Input(shape=(maxlen, 18,), name='pos')
    poses = Reshape((maxlen, 18,))(pos_input)
    lstm = Bidirectional(params['rnn'](units=params['output_dim_rnn'], activation=params['activation_rnn'], return_sequences=True),
                         input_shape=(maxlen, 18,))(poses)
    dropout = Dropout(params['dropout'])(lstm)
    out1 = Dense(maxlen, activation='sigmoid')(dropout)
    out2 = Dense(maxlen-1, activation='sigmoid')(dropout)

    model = Model(inputs=[pos_input], outputs=[out1, out2])

    model.compile(optimizer=params['optimizer'], loss='binary_crossentropy', metrics=['acc'])
    from keras.utils import plot_model
    plot_model(model, to_file='model.png', show_shapes=True)
    model.save('generated/model.h5')
    return model
