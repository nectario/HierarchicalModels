#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#from tensorflow_core.python.keras.layers.core import RepeatVector, Activation
#from tensorflow_core.python.keras.layers.merge import Concatenate
from collections import Counter

import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate, RepeatVector, Activation, Dot, Bidirectional, Embedding, Input, SpatialDropout1D, LSTM, Dropout, Lambda, Conv1D, Attention, AdditiveAttention, GlobalAveragePooling1D, TimeDistributed, AveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from MultiVectorizer import MultiVectorizer
import numpy as np
import pandas as pd
from tensorflow.keras import backend as K
from common.utils import jsonl_to_df
from common.data_utils import pad_nested_sequences
import re
from bs4 import BeautifulSoup
from collections import OrderedDict
document_max_size = 6000
section_max_size = 500
question_max_size = 110
answer_max_size = 15000
sentence_max_size = 100
and_pattern = re.compile(r"&[ ]?amp;")
import py_compile
from time import time
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pickle

class QuestionAnswer:

    def __init__(self, embedding_size=300, encoder_output=300, decoder_output=None, glove_path=None, use_cnn=False, embedding_matrix=None,
                 question_max_size=question_max_size, answer_max_size=answer_max_size, section_max_size=section_max_size, sentence_max_size=sentence_max_size,
                 document_max_size=document_max_size, tokenizer=None):

        self.embedding_size =embedding_size
        self.encoder_output = encoder_output
        self.max_output_sequence = document_max_size

        if decoder_output is None:
            self.decoder_output = 2*self.encoder_output
        self.use_cnn = use_cnn

        self.embedding_matrix = embedding_matrix
        self.vocabulary_size = None
        self.use_masking = False
        self.question_text_size = question_max_size
        self.answer_text_size = answer_max_size
        self.document_max_size = document_max_size
        self.sentence_max_size = sentence_max_size
        self.section_max_size = section_max_size

        self.vectorizer = MultiVectorizer(glove_path=glove_path, tokenizer=tokenizer)

        self.weight_path = None
        self.vectorizer_path = None
        self.model = None

    def example(self):

        # Encode each timestep
        in_sentence = Input(shape=(None,),  dtype='int64', name="Input1")
        embedded_sentence = Embedding(1000, 300, trainable=False)(in_sentence)
        lstm_sentence = LSTM(300)(embedded_sentence)
        encoded_model = Model(in_sentence, lstm_sentence)

        section_input = Input(shape=(None, None), dtype='int64', name="Input2")
        section_encoded = TimeDistributed(encoded_model)(section_input)
        section_encoded = LSTM(300)(section_encoded)
        section_model = Model(section_input, section_encoded)

        document_input = Input(shape=(None, None, None), dtype='int64', name="Input3")
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


    def get_sentence_model(self, embedding, use_attention=False, question_model=None):
        text_input = Input(shape=(None,), ragged=True, dtype="int64", name="text_input")
        text_embedding = embedding(text_input)
        output = Conv1D(128, 4, padding="same", activation="relu", strides=1)(text_embedding)

        if use_attention:
            attention = AdditiveAttention()([output, question_model])
            output = GlobalAveragePooling1D()(attention)

        model = Model(text_input, output)
        return model

    def get_section_model(self, sentence_model, question_model):
        section_input = Input(shape=(None, None), ragged=True, name="section_input")
        section_encoded = TimeDistributed(sentence_model)(section_input)
        section_encoded = Conv1D(128, 4, padding="same", activation="relu", strides=1)(section_encoded)
        attention = AdditiveAttention()([section_encoded, question_model])
        output = GlobalAveragePooling1D()(attention)
        model = Model(section_input,  output)
        return model

    def get_document_model(self, section_model, question_model):
        document_input = Input(shape=(None, None, None), ragged=True, name="document_input")
        document_encoded = TimeDistributed(section_model)(document_input)
        cnn_1d = Conv1D(128, 4, padding="same", activation="relu", strides=1)(document_encoded)
        attention = AdditiveAttention()([cnn_1d, question_model])
        output = GlobalAveragePooling1D()(attention)
        model = Model(document_input, output)
        return model

    def get_model(self):
        self.vocabulary_size = self.vectorizer.get_vocabulary_size()
        self.embedding_matrix = self.vectorizer.get_embedding_matrix()

        embedding = Embedding(self.vocabulary_size, self.embedding_size, mask_zero=False, trainable=True,
                                   weights=None if self.embedding_matrix is None else [self.embedding_matrix])

        self.question_model = self.get_sentence_model(embedding)
        self.sentence_model = self.get_sentence_model(embedding, use_attention=True)

        self.section_model = self.get_section_model(self.sentence_model, self.question_model)
        self.document_model = self.get_document_model(self.section_model, self.question_model)

        optimizer = Adadelta()

        loss_metrics = "binary_crossentropy"

        self.document_model.compile(loss=loss_metrics, optimizer=optimizer, metrics=[loss_metrics])
        self.document_model.summary()

    def preprocess(self, X, y):
        X = pad_nested_sequences(X) #pad_sequences(X, padding="post", truncating="post", maxlen=self.question_text_size, value=0.0)
        y = pad_nested_sequences(y) #pad_sequences(y, padding="post", truncating="post", maxlen=self.document_max_size, value=0.0)
        return X, y

    def fit(self, input=None, output=None, batch_size=100, epochs=200):

        if self.vectorizer_path != None:
            self.vectorizer = MultiVectorizer.load(self.vectorizer_path)

        if self.weight_path != None:
            self.model.load_weights(self.weight_path)

        self.model = self.get_model()

        #input_padded, output_padded = self.preprocess(input, output)
        #self.example().fit(input_padded, output_padded, use_multiprocessing=True, epochs=10, batch_size=1)
        callback = self.CallbackActions(vectorizer=self.vectorizer)

        self.model.fit(input, output, use_multiprocessing=True, callbacks=[callback], epochs=epochs, batch_size=batch_size, verbose=2)
        return

    def save(self, weight_path="data/weights/weights.h5", vectorizer_path="data/weights/vectorizer/"):
        self.model.save_weights(weight_path, save_format='h5')
        self.vectorizer.save(vectorizer_path)
        return

    def load(self, weight_path=None, vectorizer_path=None):
        if weight_path is not None:
            self.weight_path = weight_path
            if self.model != None:
                self.model.load_weights(weight_path)
        if vectorizer_path is not None:
            self.vectorizer_path = vectorizer_path
            self.vectorizer = MultiVectorizer.load(vectorizer_path)
        return

    def get_sentences(self, text):
        sentences = text.split(" . ")
        #sentences_tensor = tf.ragged.constant([sentences])
        return sentences

    def get_sections(self, text, tags=["p", "table", "tr", "li", "ol", "ul", "dl"], remove_html_tags=False, parse_sentences=False):

        soup = BeautifulSoup(text, "lxml")
        tag_contents = soup.find_all(tags, recursive=True)

        document_structure = []
        for tag_content in tag_contents:
            if remove_html_tags:
                tag_content = str(tag_content.get_text(strip=True))
            else:
                tag_content = str(tag_content.encode(formatter=None).decode("utf-8"))

            if parse_sentences:
                tag_content = self.get_sentences(tag_content)

            document_structure.append(tag_content)
        return document_structure

    def get_tag_distribution(self, answer_texts):
        first_html_tag_pattern = re.compile(r"^<[a-zA-Z]+>")
        counter = Counter()
        for text in answer_texts:
            tag = "".join([text[x.start() :x.end()] for x in first_html_tag_pattern.finditer(text)])
            counter[tag] += 1

        return OrderedDict(counter)

    def load_data(self, directory, filename="simplified-nq-train.jsonl", n_rows=-1):
        f = time()
        data_df = pd.DataFrame()

        train = jsonl_to_df(directory + filename, n_rows=n_rows, truncate=False)
        #test = jsonl_to_df(directory + 'simplified-nq-test.jsonl', load_annotations=False, n_rows=-1, truncate=False)

        self.questions = train["question_text"].values
        self.documents = train["document_text"].values
        self.ids = train["example_id"].astype(str)

        labels = zip(train["long_answer_start"].values, train["long_answer_end"].values, train["short_answer_start"].values, train["short_answer_end"].values,
                     train["yes_no_answer"].values, self.documents)

        print("Max Document Size:",train.document_text.str.len().max())

        #train.to_excel("data/json_data.xlsx")
        filter_pattern = re.compile(r'_colspan="[0-9][0-9]*"')

        self.long_answers = []
        self.short_answers = []
        self.yes_no_answers = []
        self.documents_h = []
        self.matches = []
        self.long_answer_spans = []
        self.short_answer_spans = []
        self.target_values = []
        long_answer  = None
        short_answer = None

        b = time()
        row_num = 0
        self.section_lengths = []
        self.detected_span = []
        for (long_answer_start, long_answer_end, short_answer_start, short_answer_end, yes_no_answer, document) in labels:

            document = re.sub(filter_pattern, "", document)
            doc_tokens = self.get_doc_tokens(document)
            y = []
            start = -1
            end = -1

            if (long_answer_start != -1 and long_answer_end != -1):
                long_answer = " ".join(doc_tokens[long_answer_start:long_answer_end])
                long_answer_tokens = doc_tokens[long_answer_start:long_answer_end]
                long_answer = re.sub(filter_pattern,"", long_answer)
                start, end = self.get_span_range(text_tokens=long_answer_tokens, doc_tokens=doc_tokens)
                self.long_answers.append(long_answer)
            else:
                self.long_answers.append("")

            if (short_answer_start != -1 and short_answer_end != -1):
                short_answer = " ".join(doc_tokens[short_answer_start:short_answer_end])
                short_answer = re.sub(filter_pattern, "", short_answer)
                self.short_answers.append(short_answer)
            else:
                self.short_answers.append("")

            self.long_answer_spans.append((long_answer_start,long_answer_end))
            self.short_answer_spans.append((short_answer_start, short_answer_end))
            self.detected_span.append((start, end))
            self.yes_no_answers.append(yes_no_answer)

            sections = self.get_sections(document, parse_sentences=True)

            self.section_lengths.append(len(sections))
            self.documents_h.append(sections)

            num_matches = 0

            for tag_text in sections:
                tag_text = " . ".join(tag_text)
                if fuzz.ratio(long_answer.lower(),tag_text.lower()) > 96:
                    num_matches += 1
                    y.append(1.0)
                else:
                    y.append(0.0)

            self.target_values.append(y)
            self.matches.append(num_matches)
            row_num += 1

        print("Max Sections:", max(self.section_lengths))

        data_df["Id"] = self.ids
        data_df["Question"] = self.questions
        data_df["Number of Matches"] = self.matches
        data_df["Long Answer"] = self.long_answers
        data_df["Long Answer Span"] = self.long_answer_spans
        data_df["Detected Span"] = self.detected_span
        data_df["Short Answer"] = self.short_answers
        data_df["Short Answer Span"] = self.short_answer_spans
        data_df[r"Yes/No Answer"] = self.yes_no_answers
        data_df["Document"] = self.documents
        data_df["Sections"] = self.documents_h
        data_df["Number of Sections"] = self.section_lengths

        data_df.to_excel("data/Training_Data.xlsx", index=False)

        docs = self.vectorizer.fit(self.documents_h)
        #document_h_ragged = tf.ragged.constant(docs)
        qs = self.vectorizer.fit(self.questions)
        #questions_ragged = qs #tf.ragged.constant(qs)
        targets = self.target_values

        #target_values_ragged = tf.ragged.constant(targets)

        #document_dataset =  tf.data.Dataset.from_tensor_slices(document_h_ragged)
        #question_dataset = tf.data.Dataset.from_tensor_slices(questions_ragged)


        #target_values_dataset = tf.data.Dataset.from_tensor_slices(target_values_ragged)

        #X  = [document_h_ragged, questions_ragged]
        #y = target_values_ragged

        #dataset = tf.data.Dataset.from_tensor_slices((input_data, target_values_dataset))
        #dataset = tf.data.Dataset.zip((input_dataset, target_values_dataset))

        tag_dist = self.get_tag_distribution(self.long_answers)
        tag_dist_df = pd.DataFrame(columns=["Tag", "Count"])
        tag_dist_df["Tag"] = tag_dist.keys()
        tag_dist_df["Count"] = tag_dist.values()
        tag_dist_df.to_excel("TagInfo.xlsx")

        return  data_df, docs, qs, targets

    def get_span_range(self, text=None, document=None, text_tokens=None, doc_tokens=None):
        if text_tokens is None and text is not None:
            text_tokens = text.split(" ")
        if doc_tokens is None and document is not None:
            doc_tokens = self.get_doc_tokens(document)
        n = len(text_tokens)
        token_span = [text_tokens == doc_tokens[i:i + n] for i in range(len(doc_tokens) - n + 1)]
        res = [j for j, val in enumerate(token_span) if val]

        value = any(token_span)

        if value:
            output = (res[0], res[0]+len(text_tokens))
        else:
            output = (-1,-1)
        return output

    def get_doc_tokens(self, document):
        doc_tokens = document.split(" ")
        return doc_tokens

    class CallbackActions(Callback):
        def __init__(self, vectorizer):
            self.vectorizer = vectorizer
            return

        def on_train_begin(self, logs={}):
            return

        def on_train_end(self, logs={}):
            return

        def on_epoch_begin(self, epoch, logs={}):
            return

        def on_epoch_end(self, epoch, logs={}):
            return

if __name__ == "__main__":
    print("Eager Execution:", tf.executing_eagerly())
    a = time()
    #qa_model = QuestionAnswer(glove_path="D:/Development/Embeddings/Glove/glove.6B.300d.txt", tokenizer=str.split)
    qa_model = QuestionAnswer(tokenizer=str.split)

    data_df, documents, questions, y = qa_model.load_data("data/jsonl_data/", filename="simplified-nq-train.jsonl", n_rows=5)


    y = []
    for doc in documents:
        tocs = []
        for token in doc:
            tocs.append(np.random.random_integers(0, high=1))
        y.append(tocs)

    output = y
    qa_model.fit(input=documents, output=output)