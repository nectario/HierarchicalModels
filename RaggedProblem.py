from tensorflow.keras.layers import Dense, Embedding, Input, LSTM, TimeDistributed
import pickle

from tensorflow.keras.layers import Dense, Embedding, Input, LSTM, TimeDistributed
from tensorflow.keras.models import Model


class RaggedProblem:

    def example(self):
        # Encode each timestep
        in_sentence = Input(shape=(None,), dtype='int64', ragged=True, name="Input1")
        embedded_sentence = Embedding(1000, 300, trainable=False)(in_sentence)
        lstm_sentence = LSTM(300)(embedded_sentence)
        encoded_model = Model(in_sentence, lstm_sentence)

        section_input = Input(shape=(None, None), dtype='int64', ragged=True, name="Input2")
        section_encoded = TimeDistributed(encoded_model)(section_input)
        section_encoded = LSTM(300)(section_encoded)
        section_model = Model(section_input, section_encoded)

        document_input = Input(shape=(None, None, None), dtype='int64', ragged=True, name="Input3")
        document_encoded = TimeDistributed(section_model)(document_input)
        document_encoded = LSTM(300, return_sequences=True)(document_encoded)
        document_encoded = Dense(1)(document_encoded)
        document_model = Model(document_input, document_encoded)
        document_model.compile(loss='categorical_crossentropy',
                               optimizer='rmsprop',
                               metrics=['accuracy'])

        print(section_model.summary())
        print(document_model.summary())
        return document_model

    def get_model(self):
        model = self.example()
        return model


    def fit(self, input=None, output=None, batch_size=100, epochs=200):
        self.model = self.get_model()
        self.model.fit(input, output, use_multiprocessing=True, epochs=epochs, batch_size=batch_size, verbose=2)
        return

if __name__ == "__main__":

    qa_model = RaggedProblem()

    input = pickle.load(open("data/X.dat", "rb"))
    output = pickle.load(open("data/y.dat", "rb"))

    qa_model.fit(input=input, output=output)

