import os
import zipfile
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import layers

import pprint
import numpy as np
print(tf.__version__)

gdrive_data_directory = "/content/drive/My Drive/Data"
gdrive_training_path = gdrive_data_directory + "/training.zip"
gdrive_test_path = gdrive_data_directory + "/test.zip"
gdrive_validation_path = gdrive_data_directory + "/validation.zip"

data_tmp_directory = gdrive_data_directory
data_tmp_training = data_tmp_directory + "/training.csv"
data_tmp_test = data_tmp_directory + "/test.csv"
data_tmp_validation = data_tmp_directory + "/validation.csv"


batch_size = 512

default_values = ["0.0" for a in range (460)]

training_data_length = 8066567 # get_data_length(data_tmp_training)
test_data_length = 2558554 #get_data_length(data_tmp_test)
validation_data_length = 2016641 #2016648# get_data_length(data_tmp_validation)

print ('traininglength:', training_data_length)
print('test:', test_data_length)
print('validation', validation_data_length)


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as text_file:
        print(text, file=text_file)

def _parse_line(line):
    # Decode the line into its fields
    fields = tf.io.decode_csv(line, default_values, field_delim='\t')

    # Pack the result into a dictionary
    features = [tf.strings.to_number(i, out_type=tf.dtypes.float32) for i in fields] #dict(zip(columns,fields))
    label = features[0] #.pop('0matching')
    
    categories = features[1:21]
    count = features[21:22]
    structure_basic_01 = features[22:67]
    structure_basic_02 = features[67:112]
    structure_basic_03 = features[112:157]
    structure_basic_05 = features[157:200]
    structure_template_01 = features[200:245]
    structure_template_02 = features[245:290]
    structure_template_03 = features[290:335]
    structure_template_05 = features[335:370]
    structure_template_06 = features[370:415]
    structure_template_08 = features[415:460]
    features = {
        'categories':categories, 
        'templatecount':count, 
        'inputStructureBasic01':structure_basic_01,
        'inputStructureBasic02':structure_basic_02, 
        'inputStructureBasic03':structure_basic_03, 
        'inputStructureBasic05':structure_basic_05, 
        'inputStructureTemplate01':structure_template_01, 
        'inputStructureTemplate02':structure_template_02, 
        'inputStructureTemplate03':structure_template_03, 
        'inputStructureTemplate05':structure_template_05, 
        'inputStructureTemplate06':structure_template_06, 
        'inputStructureTemplate08':structure_template_08}
    # Separate the label from the features

    return features, label

def createDataTensor(filePath):
    dataset = tf.data.TextLineDataset(filePath).skip(1)
    dataset = dataset.map(_parse_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    # dataset = dataset.shuffle(dataLength, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

leakyRelu_alpha = 0.01

def getStructureLayer(inputName, shape_x):
  input_structure = keras.Input(shape=(shape_x,), name=inputName)
  structure_layer = layers.RepeatVector(3)(input_structure)
  structure_layer = layers.Conv1D(filters=2, kernel_size=3)(structure_layer)
  structure_layer = layers.LeakyReLU(alpha=leakyRelu_alpha)(structure_layer)
  structure_layer = layers.Flatten()(structure_layer)	
  structure_layer = layers.Dropout(0.2)(structure_layer)

  structure_layer_dense = layers.Dense(24)(input_structure)
  structure_layer_dense = layers.LeakyReLU(alpha=leakyRelu_alpha)(structure_layer_dense)
  return input_structure, structure_layer, structure_layer_dense

def make_model():
  input_structure_basic_01, layer_structure_basic_01, layer_structure_dense_basic_01 = getStructureLayer("inputStructureBasic01", 45)
  input_structure_basic_02, layer_structure_basic_02, layer_structure_dense_basic_02 = getStructureLayer("inputStructureBasic02", 45)
  input_structure_basic_03, layer_structure_basic_03, layer_structure_dense_basic_03 = getStructureLayer("inputStructureBasic03", 45)
  input_structure_basic_05, layer_structure_basic_05, layer_structure_dense_basic_05 = getStructureLayer("inputStructureBasic05", 43)
  input_structure_template_01, layer_structure_template_01, layer_structure_dense_template_01 = getStructureLayer("inputStructureTemplate01", 45)
  input_structure_template_02, layer_structure_template_02, layer_structure_dense_template_02 = getStructureLayer("inputStructureTemplate02", 45)
  input_structure_template_03, layer_structure_template_03, layer_structure_dense_template_03 = getStructureLayer("inputStructureTemplate03", 45)
  input_structure_template_05, layer_structure_template_05, layer_structure_dense_template_05 = getStructureLayer("inputStructureTemplate05", 35)
  input_structure_template_06, layer_structure_template_06, layer_structure_dense_template_06 = getStructureLayer("inputStructureTemplate06", 45)
  layer_structures = layers.concatenate([layer_structure_basic_01, layer_structure_basic_02, layer_structure_basic_03, layer_structure_basic_05,layer_structure_template_01, layer_structure_template_02, layer_structure_template_03, layer_structure_template_05, layer_structure_template_06])
  layer_structures = layers.Dense(24)(layer_structures)
  layer_structures = layers.LeakyReLU(alpha=leakyRelu_alpha)(layer_structures)

  input_structure_template_8 = keras.Input(shape=(45,), name='inputStructureTemplate08')
  layer_structure_template_8 = layers.Dense(24)(input_structure_template_8)
  layer_structure_template_8 = layers.LeakyReLU(alpha=leakyRelu_alpha)(layer_structure_template_8)
  layer_structure_template_8 = layers.Dropout(0.2)(layer_structure_template_8)

  layer_structure_dense = layers.concatenate([layer_structure_dense_basic_01,layer_structure_dense_basic_02,layer_structure_dense_basic_03,layer_structure_dense_basic_05,layer_structure_dense_template_01, layer_structure_dense_template_02, layer_structure_dense_template_03, layer_structure_dense_template_05, layer_structure_dense_template_06, layer_structure_template_8 ])
  layer_structure_dense = layers.Dense(128)(layer_structure_dense)
  layer_structure_dense = layers.LeakyReLU(alpha=leakyRelu_alpha)(layer_structure_dense)
  layer_structure_dense = layers.Dense(64)(layer_structure_dense)
  layer_structure_dense = layers.LeakyReLU(alpha=leakyRelu_alpha)(layer_structure_dense)
  layer_structure_dense = layers.Dropout(0.2)(layer_structure_dense)

  layer_structures = layers.concatenate([layer_structure_dense, layer_structures])
  layer_structures = layers.Dense(32)(layer_structures)
  layer_structures = layers.LeakyReLU(alpha=leakyRelu_alpha)(layer_structures)
  layer_structures = layers.Dropout(0.2)(layer_structures)

  inputCategories = keras.Input(shape=(20,), name='categories')
  layersCategories = layers.Dense(8)(inputCategories)
  layersCategories = layers.LeakyReLU(alpha=leakyRelu_alpha)(layersCategories)

  inputTemplateCount = keras.Input(shape=(1,), name='templatecount')

  x = layers.concatenate([layersCategories, inputTemplateCount, layer_structures])
  x = layers.Dense(32)(x)
  x = layers.LeakyReLU(alpha=leakyRelu_alpha)(x)
  x = layers.Dropout(0.2)(x)
  x = layers.Dense(8)(x)
  x = layers.LeakyReLU(alpha=leakyRelu_alpha)(x)
  output = layers.Dense(1, activation='sigmoid', name='predictions')(x)

  inputs = [input_structure_basic_01, input_structure_basic_02, input_structure_basic_03, input_structure_basic_05,input_structure_template_01, input_structure_template_02, input_structure_template_03, input_structure_template_05, input_structure_template_06,input_structure_template_8, inputCategories, inputTemplateCount ]

  model = keras.Model(inputs=inputs, outputs=output)

  model.compile(optimizer=keras.optimizers.Nadam(),
              loss='binary_crossentropy',
              # List of metrics to monitor
              metrics=[keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC(curve='PR', name='AUPRC')])#, keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()])
  return model

model = make_model()
model.save('/content/drive/My Drive/Data/untrained_model.h5')

trainingDataset = createDataTensor(data_tmp_training)
validationDataset = createDataTensor(data_tmp_validation)

for next_element in trainingDataset:
    tf.print(next_element)
    break

training_steps = training_data_length // batch_size
validation_steps = validation_data_length // batch_size

print(training_steps)
print(validation_steps)

metrics = model.metrics_names
checkpoint_filepath = '/content/drive/My Drive/Data/checkpoints/trainmodel_weights.{epoch:02d}-{' + metrics[3] + ':.2f}.h5'

callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath)]

print('# Fit model on training data')
history = model.fit(trainingDataset,
                    epochs=20,
                    steps_per_epoch=training_steps,
                    callbacks=callbacks,
                    verbose = 2,
                    validation_data=validationDataset,
                    validation_steps=validation_steps)

model_onnx_file_path = '/content/drive/My Drive/Data/model.onnx'
model_keras_file_path = '/content/drive/My Drive/Data/model.h5'
import keras2onnx
import onnx
onnx_model = keras2onnx.convert_keras(model)
onnx.save_model(onnx_model, model_onnx_file_path)

model.save(model_keras_file_path)
print('saved')

history_file_path = '/content/drive/My Drive/Data/model_history.txt'

print('\nhistory dict:', history.history)
write_text(history_file_path, history.history)


history_file_path = '/content/drive/My Drive/Data/evaluation.txt'
testDataset = createDataTensor(data_tmp_test) 
test_steps = test_data_length // batch_size
results = model.evaluate(testDataset, steps= test_steps)
resultValuesSting = np.array2string(np.array(results))
resultMetrics_string = str (model.metrics_names)
resultString = resultValuesSting + '\n' + resultMetrics_string
write_text(history_file_path, resultString)