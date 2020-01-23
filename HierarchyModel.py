from tensorflow.keras.layers import Dense, Embedding, Input, LSTM, TimeDistributed
import pickle

from tensorflow.keras.layers import Dense, Embedding, Input, LSTM, TimeDistributed
from tensorflow.keras.models import Model
from common.data_utils import *

class HierarchyModel:

    def example(self):
        # Encode each timestep
        in_sentence = Input(shape=(None,), dtype='int64', ragged=True, name="Input1")
        embedded_sentence = Embedding(1000, 300, trainable=False)(in_sentence)
        lstm_sentence = LSTM(300)(embedded_sentence)
        encoded_model = Model(in_sentence, lstm_sentence)

        section_input = Input(shape=(None, None), dtype='int64', ragged=True, name="Input2")
        section_encoded = TimeDistributed(encoded_model)(section_input)
        section_encoded = LSTM(300)(section_encoded)
        section_encoded = Dense(1)(section_encoded)

        model = Model(section_input, section_encoded)
        model.compile(loss='categorical_crossentropy',
                               optimizer='rmsprop',
                               metrics=['accuracy'])

        print(encoded_model.summary())
        print(model.summary())
        return model

    def get_model(self):
        model = self.example()
        return model


    def fit(self, input=None, output=None, batch_size=100, epochs=200):
        self.model = self.get_model()
        self.model.fit(input, output, use_multiprocessing=True, epochs=epochs, batch_size=batch_size, verbose=2)
        return

if __name__ == "__main__":

    model = HierarchyModel()

    imdb_data_df = load_data("C:/Development/Projects/IMDB/IMDB Dataset.csv")
    imdb_data_df["review"] = imdb_data_df["review"].apply(get_sentences)
    imdb_data_df["labels"] = imdb_data_df["sentiment"].replace("positive",1).replace("negative",0)

    X = imdb_data_df["review"].values
    X = pad_nested_sequences(X)
    y = imdb_data_df["labels"]
    print("Done")

    model.fit(input=X, output=y)

