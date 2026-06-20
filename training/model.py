import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def CALSigNet():
    inp = Input(shape=(128,128,1))

    x = Conv2D(32, (3,3), activation='relu', padding='same')(inp)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)

    x = Reshape((1, -1))(x)
    x = LSTM(64, return_sequences=True)(x)

    attn = Attention()([x, x])
    x = Flatten()(attn)

    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model