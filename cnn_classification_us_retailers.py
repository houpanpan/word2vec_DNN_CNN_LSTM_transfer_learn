import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.metrics import classification_report

from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import models 
from tensorflow.contrib.keras import callbacks # "ModelCheckpoint

import data_helpers

# Get US retailer embedding as np array
embedding_path = '/temp0/dev/models/cnn_text/us_ret_all/'  #Contains the embedding trained on all the US retailer data
model_path = '/temp0/dev/models/keras/us_ret/cnn/'

input_filename ="/temp0/dev/models/US-traingview_0418_small_train.dat"
input_filename_test ="/temp0/dev/models/US-traingview_0418_small_test.dat"

# Load small train & validation data  - Preprocessed 
x_train, y_train, class_bal_dict = data_helpers.load_data_and_labels_US_ret(input_filename)  
x_test, y_test, class_bal_dict = data_helpers.load_data_and_labels_US_ret(input_filename_test)

# Label Encoded for the Industry Class text
le = preprocessing.LabelEncoder()

le.fit(y_train)
y_train = le.transform(y_train)
y_test = le.transform(y_test)

# Pickle the transform for future use - offline
joblib.dump(le,model_path + '/US_RET_LabelEncoder.pkl')

# Load the embedding, dictionary & reverse disctionary which was saved from training word2vec on the Large training US retailer
embeddings = joblib.load(embedding_path + '/128_1/embeddings.pkl')
dictionary = joblib.load(embedding_path + '/dictionary.pkl')


# Intanciate tokenizer 
tokenizer = tf.contrib.keras.preprocessing.text.Tokenizer(num_words=None, filters= '',lower=False,split=" ",char_level=False)
#fit only the training X
tokenizer.fit_on_texts(x_train)

##########################
# Initialize the embedding from the word2vec trained embedding from the Large(All) US Retailer training data
####################
W = np.random.uniform(-1.0,1.0, size=(len(tokenizer.word_counts)+1, 128))
for word, v in tokenizer.word_index.items():
    try:
        W[v,:] = embeddings[dictionary[word]] 
    except: 
        pass
###########################

### generate word index vectors for training
x_train = tf.contrib.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=100,  dtype='int32', padding='pre' , truncating='pre', value=0)

#### generate word index vectors for validation X
x_test = tf.contrib.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(x_test), maxlen=100, dtype='int32', padding='pre', truncating='pre', value=0)


##################
# Define & Compile CNN, Dense Model
###################
#input layer
main_input = layers.Input(shape=(100,), dtype='int32', name='main_input')
#Embedding layer
x = layers.Embedding(len(tokenizer.word_counts)+1,128,weights=[W],input_length=100, trainable=True,name='word_embedding')(main_input)
# Conv layer
conv1 =layers.Conv1D(filters=32, kernel_size =2, strides=1, padding='same',activation='relu', use_bias=True)(x)
conv2 =layers.Conv1D(filters=32, kernel_size =3, strides=1, padding='same',activation='relu', use_bias=True)(x)
conv3 =layers.Conv1D(filters=32, kernel_size =4, strides=1, padding='same',activation='relu', use_bias=True)(x)

#Pool layer
pool1 = layers.MaxPooling1D(pool_size=2, strides=None, padding='valid')(conv1)
pool2 = layers.MaxPooling1D(pool_size=2, strides=None, padding='valid')(conv2)
pool3 = layers.MaxPooling1D(pool_size=2, strides=None, padding='valid')(conv3)

#concat & Flatten layer to feed to Dense layer
concat = layers.concatenate([pool1, pool2, pool3], axis=1)
out1 = layers.Reshape((4800,))(concat)

# Dense layer
dense1 = layers.Dense(100, activation='relu')(out1)
# Softmax layer
main_output = layers.Dense(33, activation='softmax', name='main_output')(dense1)
# Model - put together Input & Output
model = models.Model(inputs=[main_input], outputs=[main_output])
# Compile Model
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
################################
# Train the model
############################
hist = model.fit(x_train, y_train,epochs=1, batch_size=512,validation_data = [x_test, y_test])
#####################

#### Evaluate the f1 score by class for comparison with SVC
pred =model.predict([x_test],batch_size =len(y_test))

#### Generate Classification Report
classification_report_ =  classification_report(y_test, np.argmax(pred,axis=1), target_names=le.classes_, digits = 6)

print classification_report_




