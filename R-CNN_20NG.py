# Prerna Khurana (prerna22khurana@gmail.com)
# My keras implementation of the Recurrent Convolutional Neural Network (RCNN) found in [1] 
# on the 20 Newsgroups data.
# [1] Siwei, L., Xu, L., Kang, L., and Zhao, J. 2015. Recurrent convolutional
#         neural networks for text classification. In AAAI, pp. 2267-2273.
#         http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745

import numpy as np
import os
import os.path
import sys
from sklearn.metrics import f1_score

from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Dense, LSTM, Bidirectional, Lambda, Input, TimeDistributed
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

BASE_DIR = ''
TRAIN_DATA_DIR = BASE_DIR + '/DATA set/20news-bydate/20news-bydate-train'
TEST_DATA_DIR = BASE_DIR + '/DATA set/20news-bydate/20news-bydate-test'
GLOVE_DIR = BASE_DIR + '/glove.6B/'
input_length = 800
MAX_NB_WORDS = 20000
w2vDimension = 100
VALIDATION_SPLIT = 0.1

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

def load_data_labels(TEXT_DATA_DIR):
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        path = os.path.join(TEXT_DATA_DIR, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath, encoding='latin-1')
                    t = f.read()
                    i = t.find('\n\n')  # skip header
                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                    f.close()
                    labels.append(label_id)
    return texts, labels, labels_index
    
texts, labels, labels_index = load_data_labels(TRAIN_DATA_DIR)
texts_test, labels_test, labels_indeX_test = load_data_labels(TEST_DATA_DIR)
print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor

tokenizer = Tokenizer(MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

#tokenizer.fit_on_texts(texts_test)
sequences_test = tokenizer.texts_to_sequences(texts_test)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=input_length)
data_test = pad_sequences(sequences_test, maxlen=input_length)

labels_cat = to_categorical(np.asarray(labels))

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels_cat.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels_cat = labels_cat[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

X_train = data[:-num_validation_samples]
y_train_cat = labels_cat[:-num_validation_samples]
X_val = data[-num_validation_samples:]
y_val_cat = labels_cat[-num_validation_samples:]

y_train = np.asarray(y_train_cat.argmax(axis = 1))
y_val = np.asarray(y_val_cat.argmax(axis = 1))

X_test = data_test
y_test = np.asarray(labels_test)
y_test_cat = to_categorical(y_test)

classes = len(labels_index)
print('Preparing embedding matrix.')

n_symbols = min(MAX_NB_WORDS, len(word_index))
embedding_weights = np.zeros((n_symbols, w2vDimension))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_weights[i] = embedding_vector

    
def def_model(w2vDimension,n_symbols,embedding_weights,input_length):
    model = Sequential() 
    model.add(Embedding(output_dim=w2vDimension,
                        input_dim=n_symbols,
                        mask_zero=False,
                        weights=[embedding_weights],
                        input_length=input_length,
                        trainable=False)) 
       
    model.add(Bidirectional(LSTM(hidden_dim_1, return_sequences=True)))  
    model.add(TimeDistributed(Dense(hidden_dim_2, activation = "tanh")))
    
    return model
    
hidden_dim_1 = 150
hidden_dim_2 = 150
NUM_CLASSES = y_train_cat.shape[1]
bs = 64
ne = 100
callbacks = [EarlyStopping(monitor='val_loss',patience=3,verbose=0)]

base_network = def_model(w2vDimension,n_symbols,embedding_weights,input_length)
input_a = Input(shape=(input_length,))
processed_a = base_network(input_a)    
pool_rnn = Lambda(lambda x: K.max(x, axis = 1), output_shape = (hidden_dim_2, ))(processed_a)
output = Dense(NUM_CLASSES, input_dim = hidden_dim_2, activation = "softmax")(pool_rnn)
rcnn_model = Model(input = input_a, output = output)
        
rcnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ["accuracy"])

hist = rcnn_model.fit(X_train, y_train_cat, batch_size=bs, nb_epoch=ne,
                                          verbose=2,validation_data=(X_val, y_val_cat), callbacks=callbacks)
y_proba = rcnn_model.predict(X_test, verbose=0)
y_pred = y_proba.argmax(axis = 1)
F1_score = f1_score(y_test, y_pred, average='macro')  
print("Macro F1 score for classification model - ", F1_score)
