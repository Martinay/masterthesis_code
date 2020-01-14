from tensorflow.keras.models import load_model
from keras_preprocessing.text import tokenizer_from_json
from operator import itemgetter

import json
import sys
import pandas as pd
import numpy as np

data_path = sys.argv[1]
tokenizer_directory = sys.argv[2]
model_directory = sys.argv[3]
output_directory = sys.argv[4]
result_column_name = sys.argv[5]

model_keras_file_path = model_directory + "model.h5"

tokenizer_names = ["structure01",
                       "structure02",
                       "structure03",
                       "structure04",
                       "structure05",
                       "structure06",
                       "structure07",
                       "structure08"]
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
dataset = pd.read_csv(data_path, delimiter='\t', converters={i: str for i in range(0, 100)})

header_index_with_template_categories = [i for i, x in enumerate(dataset.columns.values) if x.startswith("Template_category")]

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

encoded_dataset = encode_features(dataset)

print("load x any y")
def get_x_y(dataset, encoded_features):
	matching = np.array(dataset[result_column_name])
	y = matching.astype(np.int)

	x = encoded_features
	x["Template_Count"] = np.array([[np.float32(x)] for x in dataset["Template_count"].values])
	x["Template_Categories"] = np.array(dataset.iloc[ : , header_index_with_template_categories ].values)
	x["Template_Categories"] = x["Template_Categories"].astype(np.int)
	return (x, y)
x, y = get_x_y(dataset, encoded_dataset)

print("loading model")
model = load_model(model_keras_file_path)
print("predict")
prediction = model.predict(x)
prediction = prediction.flatten()
print("writing data")
combined = [{"label":label.item(),"prediction":prediction.item()} for label, prediction in zip(y, prediction)]

write_text(output_directory + result_column_name + "predicted.txt", json.dumps(combined))