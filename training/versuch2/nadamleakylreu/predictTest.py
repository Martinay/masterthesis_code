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
    # dataset = dataset.repeat()
    dataset = dataset.shuffle(dataLength, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    print(dataset)
    return dataset

testDataset = createDataTensor(data_tmp_test, test_data_length)

print("load and predict")
model = tf.keras.models.load_model('/content/drive/My Drive/Data/Rankingmodel.h5')
test_steps = test_data_length // batch_size

results = []
count = 0
for features, label in testDataset:
    label = label.numpy()
    predictions = model.predict(features)
    predictions = [a[0] for a in predictions]
    for label, prediction in zip(label,predictions):
        combined = {"label":label.item(),"prediction":prediction.item()}
        results.append(combined)
    count += 1
    print (str(count) + "/" + str(test_steps))

print("Write Data")
import json
jsonstring = json.dumps(results)
result_file_path = '/content/drive/My Drive/Data/Rankingprediction.txt'
write_text(result_file_path, jsonstring)
