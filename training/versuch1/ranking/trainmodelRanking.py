import ptvsd
ptvsd.enable_attach(address=('0.0.0.0', 5678))
# ptvsd.wait_for_attach()

import os
import sys
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.text import tokenizer_from_json

data_directory = sys.argv[1]
tokenizer_directory = sys.argv[2]
model_directory = sys.argv[3]
tokenizer_character_count = int(sys.argv[4])
tokenizer_word_count = int(sys.argv[5])

training_data_path = data_directory + "trainingRanking.csv"
test_data_path = data_directory + "testRanking.csv"
validation_data_path = data_directory + "validationRanking.csv"

model_onnx_file_path = model_directory + "model.onnx"
model_keras_file_path = model_directory + "model.h5"
model_png_file_path = model_directory + "model.png"
model_history_file_path = model_directory + "history.json"
training_csv_file_path = model_directory + "training_history.csv"
model_evaluation_file_path = model_directory + "evaluation.json"

model_checkpoint_directory = model_directory + "checkpoints/"
if not os.path.exists(model_checkpoint_directory):
        os.makedirs(model_checkpoint_directory)

tokenizer_names = ["structure01",
                       "structure02",
                       "structure03",
                       "structure04",
                       "structure05",
                       "structure06",
                       "structure07",
                       "structure08"]

#training_callBacks = [
#	callbacks.CSVLogger(training_csv_file_path),
#	callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.001, patience=3, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0),
#    callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')]

def read_text(path):
    with open(path, "r", encoding="utf-8") as text_file:
        text = text_file.read()
    return text

def write_text(path, text):
    with open(path, "w", encoding="utf-8") as text_file:
        print(text, file=text_file)

def load_tokenizers():
    jsons_characters = [(read_text("{0}tokenizer_{1}.json".format(tokenizer_directory, name)), name) for name in tokenizer_names]
    jsons_words = [(read_text("{0}tokenizer_{1}_words.json".format(tokenizer_directory, name)), "{0}_words".format(name)) for name in tokenizer_names]
    jsons = jsons_characters + jsons_words
    tokenizers = {name: tokenizer_from_json(json) for json, name in jsons}

    for key, tokenizer in tokenizers.items(): # fix string interpretation after loading tokenizer
        tokenizer.num_words = int(tokenizer.num_words)

    return tokenizers
		
print("load data")
training_set_features = pd.read_csv(training_data_path, delimiter='\t', converters={i: str for i in range(0, 100)})
validation_set_features = pd.read_csv(validation_data_path, delimiter='\t', converters={i: str for i in range(0, 100)})
test_set_features = pd.read_csv(test_data_path, delimiter='\t', converters={i: str for i in range(0, 100)})

header_index_with_template_categories = [i for i, x in enumerate(training_set_features.columns.values) if x.startswith("Template_category")]

print("load tokenizer")
tokenizers = load_tokenizers()

print("encode features")
def encode_features(dataset):
	features = {}
	for name in tokenizer_names:
		encoded_template = tokenizers[name].texts_to_matrix(dataset['Template_{0}'.format(name)], mode="tfidf")
		features['Template_{0}'.format(name)] = encoded_template
		
		if name != "structure08":
			encoded_basic = tokenizers[name].texts_to_matrix(dataset['Basic_{0}'.format(name)], mode="tfidf")
			features['Basic_{0}'.format(name)] = encoded_basic
	for name in tokenizer_names:
		encoded_template = tokenizers['{0}_words'.format(name)].texts_to_matrix(dataset['Template_{0}_words'.format(name)], mode="tfidf")
		features['Template_{0}_words'.format(name)] = encoded_template
		
		if name != "structure08":
			encoded_basic = tokenizers['{0}_words'.format(name)].texts_to_matrix(dataset['Basic_{0}_words'.format(name)], mode="tfidf")
			features['Basic_{0}_words'.format(name)] = encoded_basic
	return features

encoded_features_training = encode_features(training_set_features)
encoded_features_validation = encode_features(validation_set_features)
encoded_features_test = encode_features(test_set_features)

print("load x any y")
def get_x_y(dataset, encoded_features):
	matching = np.array(dataset["ranking"])
	y = matching.astype(np.int)

	x = encoded_features
	x["Template_Count"] = np.array([[int(x)] for x in dataset["Template_count"].values])
	x["Template_Categories"] = np.array(dataset.iloc[ : , header_index_with_template_categories ].values)
	x["Template_Categories"] = x["Template_Categories"].astype(np.int)
	return (x, y)
x_train, y_train = get_x_y(training_set_features, encoded_features_training)
x_validation, y_validation = get_x_y(validation_set_features, encoded_features_validation)
x_test, y_test = get_x_y(test_set_features, encoded_features_test)

print("creating model")
from tensorflow import keras
from tensorflow.keras import layers, metrics
from tensorflow.keras.utils import plot_model

len_template_categories = len(header_index_with_template_categories)

dropout_rate = 0.2
input_template_categories = layers.Input(shape=(len_template_categories,), name='Template_Categories')
dense_template_categories = layers.Dense(6, activation='relu')(input_template_categories)

input_placement_count = layers.Input(shape=(1,), name='Template_Count')

input_words = []
convolutional_basic_words = []
convolutional_template_words = []        

template_words_convolutional_kernel_size = 8
basic_words_convolutional_kernel_size = 4
template_words_convolutional_count = 64
basic_words_convolutional_count = 48
        
template_words_second_convolutional = False
basic_words_second_convolutional = True

        
template_words_second_convolutional_kernel_size = 6
template_words_second_convolutional_count = 24
        
basic_words_second_convolutional_kernel_size = 6
basic_words_second_convolutional_count = 24

for name in tokenizer_names:
	input_layer = layers.Input(shape=(tokenizer_word_count,), name='Template_{0}_words'.format(name))
	embedding_layer = layers.Embedding(tokenizer_word_count, 4)(input_layer)
	convolutional_layer = layers.Conv1D(template_words_convolutional_count, kernel_size=template_words_convolutional_kernel_size, activation='relu')(embedding_layer)
	if template_words_second_convolutional:
		convolutional_layer = layers.Conv1D(template_words_second_convolutional_count, kernel_size=template_words_second_convolutional_kernel_size, activation='relu')(convolutional_layer)
	input_words.append(input_layer)
	convolutional_basic_words.append(convolutional_layer)
#
	if name != "structure08":
		input_layer = layers.Input(shape=(tokenizer_word_count,), name='Basic_{0}_words'.format(name))
		embedding_layer = layers.Embedding(tokenizer_word_count, 4)(input_layer)
		convolutional_layer = layers.Conv1D(basic_words_convolutional_count, kernel_size=basic_words_convolutional_kernel_size, activation='relu')(embedding_layer)
		if basic_words_second_convolutional:
			convolutional_layer = layers.Conv1D(basic_words_second_convolutional_count, kernel_size=basic_words_second_convolutional_kernel_size, activation='relu')(convolutional_layer)
		input_words.append(input_layer)
		convolutional_template_words.append(convolutional_layer)
#
input_chars = []
convolutional_template_char = []
convolutional_basic_char = []
template_char_second_convolutional = False
basic_char_second_convolutional = True

template_char_convolutional_kernel_size = 8
basic_char_convolutional_kernel_size = 10
template_char_convolutional_count = 32
basic_char_convolutional_count = 64
        

template_char_second_convolutional_kernel_size = 6
template_char_second_convolutional_count = 24
        
basic_char_second_convolutional_kernel_size = 10
basic_char_second_convolutional_count = 24

for name in tokenizer_names:
	input_layer = layers.Input(shape=(tokenizer_character_count,), name='Template_{0}'.format(name))
	embedding_layer = layers.Embedding(tokenizer_character_count, 4)(input_layer)
	convolutional_layer = layers.Conv1D(template_char_convolutional_count, kernel_size=template_char_convolutional_kernel_size, activation='relu')(embedding_layer)
	if template_char_second_convolutional:
		convolutional_layer = layers.Conv1D(template_char_second_convolutional_count, kernel_size=template_char_second_convolutional_kernel_size, activation='relu')(convolutional_layer)
	input_chars.append(input_layer)
	convolutional_template_char.append(convolutional_layer)
#
	if name != "structure08":
		input_layer = layers.Input(shape=(tokenizer_character_count,), name='Basic_{0}'.format(name))
		embedding_layer = layers.Embedding(tokenizer_character_count, 4)(input_layer)
		convolutional_layer = layers.Conv1D(basic_char_convolutional_count, kernel_size=basic_char_convolutional_kernel_size, activation='relu')(embedding_layer)
		if basic_char_second_convolutional:
			convolutional_layer = layers.Conv1D(basic_char_second_convolutional_count, kernel_size=basic_char_second_convolutional_kernel_size, activation='relu')(convolutional_layer)
		input_chars.append(input_layer)
		convolutional_basic_char.append(convolutional_layer)
#
layer_template_words = layers.concatenate(convolutional_template_words)
layer_template_words = layers.Flatten()(layer_template_words)
layer_template_words = layers.Dense(64, activation='relu')(layer_template_words)
layer_template_words = layers.Dropout(dropout_rate)(layer_template_words)

layer_basic_words = layers.concatenate(convolutional_basic_words)
layer_basic_words = layers.Flatten()(layer_basic_words)        
layer_basic_words = layers.Dense(128, activation='relu')(layer_basic_words)

layer_words = layers.concatenate([layer_template_words, layer_basic_words])
layer_words = layers.Dense(128, activation='relu')(layer_words)

#
layer_template_char = layers.concatenate(convolutional_template_char)
layer_template_char = layers.Flatten()(layer_template_char)
layer_template_char = layers.Dense(320, activation='relu')(layer_template_char)
layer_template_char = layers.Dropout(dropout_rate)(layer_template_char)

layer_basic_char = layers.concatenate(convolutional_basic_char)
layer_basic_char = layers.Flatten()(layer_basic_char)        
layer_basic_char = layers.Dense(128, activation='relu')(layer_basic_char)

layer_chars = layers.concatenate([layer_template_char, layer_basic_char])
layer_chars = layers.Dense(128, activation='relu')(layer_chars)

#
layer = layers.concatenate([layer_words, layer_chars])
layer = layers.Dense(128, activation='relu')(layer)
layer = layers.Dropout(dropout_rate)(layer)
layer = layers.concatenate([dense_template_categories, input_placement_count, layer])
layer = layers.Dense(128, activation='relu')(layer)
layer = layers.Dropout(dropout_rate)(layer)
layer = layers.Dense(1)(layer)
inputs = [input_template_categories, input_placement_count] + input_words + input_chars

model = keras.Model(inputs=inputs, outputs=[layer])
model.compile(
    optimizer=keras.optimizers.Adagrad(0.01),
    loss='huber_loss',
    metrics=[metrics.RootMeanSquaredError(), metrics.MeanAbsoluteError()])
			
# print(model.summary())
print("plotting model")
plot_model(
	model,
	to_file=model_png_file_path,
    show_shapes=True,
    show_layer_names=False,
    rankdir='LR'
) 

print("start training")
metrics = model.metrics_names
checkpoint_filepath = model_checkpoint_directory + 'trainmodel_weights.{epoch:02d}-{' + metrics[1] + ':.2f}.h5'

keras_callbacks =[keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath)]
history = model.fit(x = x_train, y=y_train, verbose=0, callbacks=keras_callbacks, batch_size=32, epochs=20, validation_data=(x_validation, y_validation))

print("export to onnx")
import keras2onnx
import onnx
onnx_model = keras2onnx.convert_keras(model)
onnx.save_model(onnx_model, model_onnx_file_path)

print("save model")
model.save(model_keras_file_path)
#
#####################

import json
write_text(model_history_file_path,json.dumps(str(history.history)))

print("evaluate model")

evaluation = model.evaluate(x = x_test, y = y_test, verbose=0)
print(evaluation)
print(model.metrics_names)
write_text(model_evaluation_file_path, '{0}\n{1}'.format(evaluation,model.metrics_names ))
