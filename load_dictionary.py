import json

import env


def load_dictionary():

    dataset = None
    with open(env.QIPEDC_DATASET, 'rb') as file:
        dataset = json.load(file)

    encoded_word = {}
    for word_metadata in dataset['data']:
        word = str(word_metadata['word']).lower()
        id = str(word_metadata['_id']).upper()
        if word not in encoded_word or not str(encoded_word[word]).endswith('B'):
            encoded_word[word] = id

    return encoded_word