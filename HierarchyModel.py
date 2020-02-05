from tensorflow.keras.layers import Dense, Embedding, Input, LSTM, TimeDistributed
import pickle

from tensorflow.keras.layers import Dense, Embedding, Input, LSTM, TimeDistributed
from tensorflow.keras.models import Model
from common.data_utils import *
from MultiVectorizer import *

class HierarchyModel:

    def __init__(self, vectorizer=None):
        self.vectorizer = vectorizer

    def sequence_to_sequence_model(self):
        # Encode each timestep
        in_sentence = Input(shape=(None,), dtype='int64', ragged=True, name="Input1")
        embedded_sentence = Embedding(vectorizer.get_vocabulary_size(), 300, trainable=True)(in_sentence)
        lstm_sentence = LSTM(500)(embedded_sentence)
        encoded_model = Model(in_sentence, lstm_sentence)

        section_input = Input(shape=(None, None), dtype='int64', ragged=True, name="Input2")
        section_encoded = TimeDistributed(encoded_model)(section_input)
        section_encoded = LSTM(400)(section_encoded)
        section_encoded = Dense(1)(section_encoded)

        model = Model(section_input, section_encoded)
        model.compile(loss='categorical_crossentropy',
                               optimizer='rmsprop',
                               metrics=['accuracy'])

        print(encoded_model.summary())
        print(model.summary())
        return model

    def get_model(self):
        model = self.sequence_to_sequence_model()
        return model


    def fit(self, input=None, output=None, validation_data=None, batch_size=100, epochs=200):
        self.model = self.get_model()
        self.model.fit(input, output, validation_data=validation_data, use_multiprocessing=True, epochs=epochs, batch_size=batch_size, verbose=2)
        return

if __name__ == "__main__":

    vectorizer = MultiVectorizer()
    model = HierarchyModel(vectorizer=vectorizer)

    imdb_data_df = load_data("C:/Development/Projects/IMDB/IMDB Dataset.csv", rows=400)
    imdb_data_df["review"] = imdb_data_df["review"].apply(get_sentences)
    imdb_data_df["labels"] = imdb_data_df["sentiment"].replace("positive",1).replace("negative",0)

    X = imdb_data_df["review"].values

    y = imdb_data_df["labels"]

    X_train = X[0:300]
    y_train = y[0:300]
    X_train = vectorizer.fit(X_train)
    X_train = pad_list_of_lists(X_train, shape=(len(X_train), 50,150))

    X_test = X[300:-1]
    y_test = y[300:-1]
    X_test = vectorizer.fit(X_test)
    X_test = pad_list_of_lists(X_test, shape=(len(X_test), 40,150))

    print("Done")

    model.fit(input=X_train, output=y_train, validation_data=(X_test, y_test))

