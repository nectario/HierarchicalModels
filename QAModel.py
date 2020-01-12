
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Input,  Conv1D, AdditiveAttention, GlobalAveragePooling1D, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta

import pickle

class QuestionAnswer:

    def get_text_model(self, embedding, use_attention=False, question_model=None):
        text_input = Input(shape=(None,), name="text_input")
        text_embedding = embedding(text_input)
        output = Conv1D(128, 4, padding="same", activation="relu", strides=1)(text_embedding)

        if use_attention:
            attention = AdditiveAttention()([output, question_model])
            output = GlobalAveragePooling1D()(attention)

        model = Model(text_input, output)
        return model

    def get_section_model(self, sentence_model, question_model):
        section_input = Input(shape=(None, None), name="section_input")
        section_encoded = TimeDistributed(sentence_model)(section_input)
        section_encoded = Conv1D(128, 4, padding="same", activation="relu", strides=1)(section_encoded)
        attention = AdditiveAttention()([section_encoded, question_model])
        output = GlobalAveragePooling1D()(attention)
        model = Model(section_input,  output)
        return model

    def get_document_model(self, section_model, question_model):
        document_input = Input(shape=(None, None, None), name="document_input")
        document_encoded = TimeDistributed(section_model)(document_input)
        cnn_1d = Conv1D(128, 4, padding="same", activation="relu", strides=1)(document_encoded)
        attention = AdditiveAttention()([cnn_1d, question_model])
        output = GlobalAveragePooling1D()(attention)
        model = Model(document_input, output)
        return model

    def get_model(self):

        embedding = Embedding(5000, 300, mask_zero=True, trainable=True)

        self.question_model = self.get_text_model(embedding)
        self.sentence_model = self.get_text_model(embedding, use_attention=True)

        self.section_model = self.get_section_model(self.sentence_model, self.question_model)
        self.document_model = self.get_document_model(self.section_model, self.question_model)

        optimizer = Adadelta()

        loss_metrics = "binary_crossentropy"

        self.document_model.compile(loss=loss_metrics, optimizer=optimizer, metrics=[loss_metrics])
        self.document_model.summary()


    def fit(self, input=None, output=None, batch_size=100, epochs=200):

        self.model = self.get_model()

        self.model.fit(input, output, use_multiprocessing=True, epochs=epochs, batch_size=batch_size, verbose=2)
        return



if __name__ == "__main__":

    qa_model = QuestionAnswer()

    input = pickle.load(open("data/X.dat", "rb"))
    output = pickle.load(open("data/y.dat", "rb"))

    qa_model.fit(input=input, output=output)

