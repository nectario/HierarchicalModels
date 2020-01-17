import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate, RepeatVector, Activation, Dot, Bidirectional, Flatten, Embedding, Input, SpatialDropout1D, LSTM, Dropout, Lambda, Conv2D, Conv1D, Attention, AdditiveAttention, GlobalAveragePooling1D, TimeDistributed, AveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.backend import backend
import tensorflow_hub as hub
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow_datasets as tfds

def cnn_classifier():
    # Encode each timestep
    # embedding = Embedding(10000, 300, trainable=True)(input)

    embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
    input = Input(shape=(None,), name="Input")
    hub_layer = hub.KerasLayer(embedding, trainable=True)(input)
    cnn = Conv1D(64,3, padding="same", activation="relu")(hub_layer)
    cnn = Conv1D(32, 3, padding="same", activation="relu")(cnn)
    cnn = Conv1D(16, 3, padding="same", activation="relu")(cnn)
    cnn = Flatten()(cnn)
    output = Dense(1, activation="sigmoid")(cnn)

    model = Model(input, output)
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    model.summary()

    return model

#input_1 = np.random.rand(3,3,3,1 ) #np.array([[1, 2, 3, 4, 5, 6, 7],[1, 2, 3, 4, 5, 6, 7],[1, 2, 3, 4, 5, 6, 7]])


# (x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
#                                                       num_words=None,
#                                                       skip_top=0,
#                                                       maxlen=None,
#                                                       seed=113,
#                                                       start_char=1,
#                                                       oov_char=2,
#                                                       index_from=3)

#X_train = pad_sequences(x_train, maxlen=1100, dtype='int32', padding='post', truncating='post', value=0.0)
#X_test = pad_sequences(x_test, maxlen=1100, dtype='int32', padding='post', truncating='post', value=0.0)

model = cnn_classifier()

train_validation_split = tfds.Split.TRAIN.subsplit([8, 2])

(train_data, validation_data), test_data = tfds.load(
    name="imdb_reviews",
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=True)

train_data = pad_sequences(train_data, maxlen=1100, dtype='int32', padding='post', truncating='post', value=0.0)
test_data = pad_sequences(test_data, maxlen=1100, dtype='int32', padding='post', truncating='post', value=0.0)

#model.fit(X_train, y_train, epochs=100, validation_data=(X_test,y_test), verbose=1)

model.fit(train_data.shuffle(25000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=10)
