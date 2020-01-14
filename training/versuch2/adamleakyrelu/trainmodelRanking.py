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
gdrive_training_path = gdrive_data_directory + "/trainingRanking.zip"
gdrive_test_path = gdrive_data_directory + "/testRanking.zip"
gdrive_validation_path = gdrive_data_directory + "/validationRanking.zip"

data_tmp_directory = gdrive_data_directory
data_tmp_training = data_tmp_directory + "/trainingRanking.csv"
data_tmp_test = data_tmp_directory + "/testRanking.csv"
data_tmp_validation = data_tmp_directory + "/validationRanking.csv"


batch_size = 32

default_values = ["0.0" for a in range (460)]


training_data_length = 59922 # get_data_length(data_tmp_training)
test_data_length = 18767 #get_data_length(data_tmp_test)
validation_data_length = 15032#15148# get_data_length(data_tmp_validation)

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
    # label = features[0:15] #.pop('0matching')
    
    # categories = features[15:35]
    # count = features[35:36]
    # structure_basic_01 = features[36:81]
    # structure_basic_02 = features[81:126]
    # structure_basic_03 = features[126:171]
    # structure_basic_05 = features[171:215]
    # structure_template_01 = features[215:260]
    # structure_template_02 = features[260:305]
    # structure_template_03 = features[305:350]
    # structure_template_05 = features[350:385]
    # structure_template_06 = features[385:430]
    # structure_template_08 = features[430:475]
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

def createDataTensor(filePath, dataLength):
    dataset = tf.data.TextLineDataset(filePath).skip(1)
    dataset = dataset.map(_parse_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(dataLength, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    print(dataset)
    return dataset

leakyRelu_alpha = 0.01

def getStructureLayer(inputName, shape_x):
  input_structure = keras.Input(shape=(shape_x,), name=inputName)

  structure_layer_dense = layers.Dense(24, kernel_regularizer=keras.regularizers.l2(l=0.001))(input_structure)
  structure_layer_dense = layers.LeakyReLU(alpha=leakyRelu_alpha)(structure_layer_dense)
  return input_structure, structure_layer_dense

def make_model():
  input_structure_basic_01, layer_structure_dense_basic_01 = getStructureLayer("inputStructureBasic01", 45)
  input_structure_basic_02, layer_structure_dense_basic_02 = getStructureLayer("inputStructureBasic02", 45)
  input_structure_basic_03, layer_structure_dense_basic_03 = getStructureLayer("inputStructureBasic03", 45)
  input_structure_basic_05, layer_structure_dense_basic_05 = getStructureLayer("inputStructureBasic05", 43)
  input_structure_template_01, layer_structure_dense_template_01 = getStructureLayer("inputStructureTemplate01", 45)
  input_structure_template_02, layer_structure_dense_template_02 = getStructureLayer("inputStructureTemplate02", 45)
  input_structure_template_03, layer_structure_dense_template_03 = getStructureLayer("inputStructureTemplate03", 45)
  input_structure_template_05, layer_structure_dense_template_05 = getStructureLayer("inputStructureTemplate05", 35)
  input_structure_template_06, layer_structure_dense_template_06 = getStructureLayer("inputStructureTemplate06", 45)
  
  input_structure_template_8 = keras.Input(shape=(45,), name='inputStructureTemplate08')
  layer_structure_template_8 = layers.Dense(24, kernel_regularizer=keras.regularizers.l2(l=0.001))(input_structure_template_8)
  layer_structure_template_8 = layers.LeakyReLU(alpha=leakyRelu_alpha)(layer_structure_template_8)
  layer_structure_template_8 = layers.Dropout(0.2)(layer_structure_template_8)

  layer_structure_dense = layers.concatenate([layer_structure_dense_basic_01,layer_structure_dense_basic_02,layer_structure_dense_basic_03,layer_structure_dense_basic_05,layer_structure_dense_template_01, layer_structure_dense_template_02, layer_structure_dense_template_03, layer_structure_dense_template_05, layer_structure_dense_template_06, layer_structure_template_8 ])
  layer_structure_dense = layers.Dense(128, kernel_regularizer=keras.regularizers.l2(l=0.001))(layer_structure_dense)
  layer_structure_dense = layers.LeakyReLU(alpha=leakyRelu_alpha)(layer_structure_dense)
  layer_structure_dense = layers.Dense(64, kernel_regularizer=keras.regularizers.l2(l=0.001))(layer_structure_dense)
  layer_structure_dense = layers.LeakyReLU(alpha=leakyRelu_alpha)(layer_structure_dense)
  layer_structure_dense = layers.Dropout(0.2)(layer_structure_dense)

  layer_structures = layer_structure_dense #layers.Dropout(0.2)(layer_structures)

  inputCategories = keras.Input(shape=(20,), name='categories')
  layersCategories = layers.Dense(8, kernel_regularizer=keras.regularizers.l2(l=0.001))(inputCategories)
  layersCategories = layers.LeakyReLU(alpha=leakyRelu_alpha)(layersCategories)

  inputTemplateCount = keras.Input(shape=(1,), name='templatecount')

  x = layers.concatenate([layersCategories, inputTemplateCount, layer_structures])
  x = layers.Dropout(0.2)(x)
  x = layers.Dense(32, kernel_regularizer=keras.regularizers.l2(l=0.001))(x)
  x = layers.LeakyReLU(alpha=leakyRelu_alpha)(x)
  output = layers.Dense(1, name='predictions')(x)

  inputs = [input_structure_basic_01, input_structure_basic_02, input_structure_basic_03, input_structure_basic_05,input_structure_template_01, input_structure_template_02, input_structure_template_03, input_structure_template_05, input_structure_template_06,input_structure_template_8, inputCategories, inputTemplateCount ]

  model = keras.Model(inputs=inputs, outputs=output)

  model.compile(optimizer=keras.optimizers.Adam(),
              loss='mean_squared_error',
              metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError()])
  return model

model = make_model()
model.save('/content/drive/My Drive/Data/tmp_ranking_model.h5')

trainingDataset = createDataTensor(data_tmp_training, training_data_length)
validationDataset = createDataTensor(data_tmp_validation, validation_data_length)

# for next_element in trainingDataset:
#     tf.print(next_element)
#     break

training_steps = training_data_length // batch_size
validation_steps = validation_data_length // batch_size

print(training_steps)
print(validation_steps)

metrics = model.metrics_names
checkpoint_filepath = '/content/drive/My Drive/Data/checkpoints/Rankingweights.{epoch:02d}-{' + metrics[1] + ':.2f}.h5'

callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath)]

print('# Fit model on training data')
history = model.fit(trainingDataset,
                    epochs=20,
                    steps_per_epoch=training_steps,
                    callbacks=callbacks,
                    validation_data=validationDataset,
                    validation_steps=validation_steps)

model_onnx_file_path = '/content/drive/My Drive/Data/Rankingmodel.onnx'
model_keras_file_path = '/content/drive/My Drive/Data/Rankingmodel.h5'
import keras2onnx
import onnx
onnx_model = keras2onnx.convert_keras(model)
onnx.save_model(onnx_model, model_onnx_file_path)

model.save(model_keras_file_path)
print('saved')

history_file_path = '/content/drive/My Drive/Data/Rankingmodel_history.txt'

print('\nhistory dict:', history.history)
write_text(history_file_path, history.history)


testDataset = createDataTensor(data_tmp_test, test_data_length)

history_file_path = '/content/drive/My Drive/Data/Rankingevaluation.txt'
test_steps = test_data_length // batch_size
results = model.evaluate(testDataset, steps= test_steps)
resultValuesSting = np.array2string(np.array(results))
resultMetrics_string = str (model.metrics_names)
resultString = resultValuesSting + '\n' + resultMetrics_string
write_text(history_file_path, resultString)