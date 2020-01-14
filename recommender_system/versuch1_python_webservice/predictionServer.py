from flask import Flask, request
app = Flask(__name__)

from tensorflow.keras.models import load_model
from keras_preprocessing.text import tokenizer_from_json

import json
import sys
import pandas as pd
import numpy as np
from io import StringIO

import time


tokenizer_directory='tokenizer/'
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

def load_tokenizers():
    jsons_characters = [(read_text("{0}tokenizer_{1}.json".format(tokenizer_directory, name)), name) for name in tokenizer_names]
    jsons_words = [(read_text("{0}tokenizer_{1}_words.json".format(tokenizer_directory, name)), "{0}_words".format(name)) for name in tokenizer_names]
    jsons = jsons_characters + jsons_words
    tokenizers = {name: tokenizer_from_json(json) for json, name in jsons}

    for key, tokenizer in tokenizers.items(): # fix string interpretation after loading tokenizer
        tokenizer.num_words = int(tokenizer.num_words)

    return tokenizers

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

def get_x(dataset, encoded_features, header_index_with_template_categories):
	x = encoded_features
	x["Template_Count"] = np.array([[float(x)] for x in dataset["Template_count"].values])
	x["Template_Categories"] = np.array(dataset.iloc[ : , header_index_with_template_categories ].values)
	x["Template_Categories"] = x["Template_Categories"].astype(np.int)
	return x

def predict(csv_data, model):
    start = time.time()    
    csv_data_wrapped = StringIO(str(csv_data,'utf-8'))
    end = time.time()    
    print("convert data to string", file=sys.stderr)
    print(end - start, file=sys.stderr)
    start = time.time()    
    dataset = pd.read_csv(csv_data_wrapped, delimiter='\t', converters={i: str for i in range(0, 100)})
    end = time.time()    
    print("read pandas", file=sys.stderr)
    print(end - start, file=sys.stderr)
    start = time.time()    
    header_index_with_template_categories = [i for i, x in enumerate(dataset.columns.values) if x.startswith("Template_category")]
    end = time.time()    
    print("header_index_with_template_categories", file=sys.stderr)
    print(end - start, file=sys.stderr)
    start = time.time()    

    encoded_dataset = encode_features(dataset)
    end = time.time()    
    print("encode_features", file=sys.stderr)
    print(end - start, file=sys.stderr)
    start = time.time() 

    x = get_x(dataset, encoded_dataset, header_index_with_template_categories)
    end = time.time()    
    print("get_x", file=sys.stderr)
    print(end - start, file=sys.stderr)
    start = time.time()

    prediction = model.predict(x)
    end = time.time()    
    print("predict", file=sys.stderr)
    print(end - start, file=sys.stderr)
    start = time.time()
    predictionFormatted = ["{0:.8f}".format(value) for value in prediction.flatten()]
    end = time.time()    
    print("predictionFormatted", file=sys.stderr)
    print(end - start, file=sys.stderr)
    start = time.time()
    
    return json.dumps(predictionFormatted)


print("loading tokenizer")
tokenizers = load_tokenizers()

print("loading models")
filter_model = load_model('models/filter.h5')
ranking_model = load_model('models/ranking.h5')


################################################################
@app.route('/filter', methods=['POST'])
def filter_request():
    print("predict filter")
    prediction = predict(request.data, filter_model)
    return prediction

@app.route('/rank', methods=['POST'])
def ranking_request():
    print("predict ranking")
    prediction = predict(request.data, ranking_model)
    return prediction

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)

