import ptvsd
ptvsd.enable_attach(address=('0.0.0.0', 5678))
#ptvsd.wait_for_attach()
#print("attached")


import sys
import pandas as pd
from keras.preprocessing.text import Tokenizer

training_data_path = sys.argv[1]
output_directory = sys.argv[2]
tokenizer_character_count = sys.argv[3]
tokenizer_word_count = sys.argv[4]

if not training_data_path or not output_directory or not tokenizer_character_count or not tokenizer_word_count:
    raise "wrong arguments %s, %s, %s, %s".format(training_data_path, output_directory, tokenizer_character_count, tokenizer_word_count)


def get_and_fit_tokenizer(structure_features, structure_item, onlyTemplate = False):
    tokenizer = Tokenizer(num_words=tokenizer_character_count, char_level=True, filters='')
    all_words = structure_features['Template_' + structure_item]
    if not onlyTemplate:
        all_words = all_words + structure_features['Basic_' + structure_item]
    tokenizer.fit_on_texts(all_words)
    return tokenizer

def get_and_fit_tokenizer_words(structure_features, structure_item, onlyTemplate = False):
    tokenizer = Tokenizer(num_words=tokenizer_word_count, char_level=False, filters='', split=' ')
    all_words = structure_features['Template_' + structure_item + '_words']
    if not onlyTemplate:
        all_words = all_words + structure_features['Basic_' + structure_item + '_words']
    tokenizer.fit_on_texts(all_words)
    return tokenizer

loaded_data = pd.read_csv(training_data_path, delimiter='\t', converters={i: str for i in range(0, 100)})
print(loaded_data.shape)
indices = loaded_data['matching'].values == '1'
loaded_data = loaded_data[indices]
loaded_data = loaded_data.reset_index()
print(loaded_data.shape)
structure01_tokenizer = get_and_fit_tokenizer(loaded_data, 'structure01')
structure02_tokenizer = get_and_fit_tokenizer(loaded_data, 'structure02')
structure03_tokenizer = get_and_fit_tokenizer(loaded_data, 'structure03')
structure04_tokenizer = get_and_fit_tokenizer(loaded_data, 'structure04')
structure05_tokenizer = get_and_fit_tokenizer(loaded_data, 'structure05')
structure06_tokenizer = get_and_fit_tokenizer(loaded_data, 'structure06')
structure07_tokenizer = get_and_fit_tokenizer(loaded_data, 'structure07')
structure08_tokenizer = get_and_fit_tokenizer(loaded_data, 'structure08', True)

structure01_tokenizer_words = get_and_fit_tokenizer_words(loaded_data, 'structure01')
structure02_tokenizer_words = get_and_fit_tokenizer_words(loaded_data, 'structure02')
structure03_tokenizer_words = get_and_fit_tokenizer_words(loaded_data, 'structure03')
structure04_tokenizer_words = get_and_fit_tokenizer_words(loaded_data, 'structure04')
structure05_tokenizer_words = get_and_fit_tokenizer_words(loaded_data, 'structure05')
structure06_tokenizer_words = get_and_fit_tokenizer_words(loaded_data, 'structure06')
structure07_tokenizer_words = get_and_fit_tokenizer_words(loaded_data, 'structure07')
structure08_tokenizer_words = get_and_fit_tokenizer_words(loaded_data, 'structure08', True)

tokenizer_json = structure01_tokenizer.to_json()
with open(output_directory + "tokenizer_structure01.json", "w", encoding="utf-8") as text_file:
    print(tokenizer_json, file=text_file)

tokenizer_json = structure02_tokenizer.to_json()
with open(output_directory + "tokenizer_structure02.json", "w", encoding="utf-8") as text_file:
    print(tokenizer_json, file=text_file)

tokenizer_json = structure03_tokenizer.to_json()
with open(output_directory + "tokenizer_structure03.json", "w", encoding="utf-8") as text_file:
    print(tokenizer_json, file=text_file)
	
tokenizer_json = structure04_tokenizer.to_json()
with open(output_directory + "tokenizer_structure04.json", "w", encoding="utf-8") as text_file:
    print(tokenizer_json, file=text_file)

tokenizer_json = structure05_tokenizer.to_json()
with open(output_directory + "tokenizer_structure05.json", "w", encoding="utf-8") as text_file:
    print(tokenizer_json, file=text_file)

tokenizer_json = structure06_tokenizer.to_json()
with open(output_directory + "tokenizer_structure06.json", "w", encoding="utf-8") as text_file:
    print(tokenizer_json, file=text_file)

tokenizer_json = structure07_tokenizer.to_json()
with open(output_directory + "tokenizer_structure07.json", "w", encoding="utf-8") as text_file:
    print(tokenizer_json, file=text_file)

tokenizer_json = structure08_tokenizer.to_json()
with open(output_directory + "tokenizer_structure08.json", "w", encoding="utf-8") as text_file:
    print(tokenizer_json, file=text_file)

	
tokenizer_json = structure01_tokenizer_words.to_json()
with open(output_directory + "tokenizer_structure01_words.json", "w", encoding="utf-8") as text_file:
    print(tokenizer_json, file=text_file)
	
tokenizer_json = structure02_tokenizer_words.to_json()
with open(output_directory + "tokenizer_structure02_words.json", "w", encoding="utf-8") as text_file:
    print(tokenizer_json, file=text_file)

tokenizer_json = structure03_tokenizer_words.to_json()
with open(output_directory + "tokenizer_structure03_words.json", "w", encoding="utf-8") as text_file:
    print(tokenizer_json, file=text_file)
	
tokenizer_json = structure04_tokenizer_words.to_json()
with open(output_directory + "tokenizer_structure04_words.json", "w", encoding="utf-8") as text_file:
    print(tokenizer_json, file=text_file)

tokenizer_json = structure05_tokenizer_words.to_json()
with open(output_directory + "tokenizer_structure05_words.json", "w", encoding="utf-8") as text_file:
    print(tokenizer_json, file=text_file)
	
tokenizer_json = structure06_tokenizer_words.to_json()
with open(output_directory + "tokenizer_structure06_words.json", "w", encoding="utf-8") as text_file:
    print(tokenizer_json, file=text_file)
	
tokenizer_json = structure07_tokenizer_words.to_json()
with open(output_directory + "tokenizer_structure07_words.json", "w", encoding="utf-8") as text_file:
    print(tokenizer_json, file=text_file)

tokenizer_json = structure08_tokenizer_words.to_json()
with open(output_directory + "tokenizer_structure08_words.json", "w", encoding="utf-8") as text_file:
    print(tokenizer_json, file=text_file)
