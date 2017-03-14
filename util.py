# encoding: UTF-8
import re
import os
import collections

import tensorflow as tf
from tensorflow.python.ops import array_ops

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer

import numpy as np

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def build_fixed_size_dataset(origin_dataset, vocabulary_size):
    count = [['UNK', -1]]
    words = []
    for doc in origin_dataset:
        words.extend(doc)

    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    print('real vocab size is %d' % len(dictionary))
    dataset = list()
    unk_count = 0
    for doc in origin_dataset:
        data = []
        for word in doc:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            data.append(index)
        data = np.array(data).reshape(1, -1)
        dataset.append(data)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dataset, count, dictionary, reverse_dictionary


def build_test_set(origin_dataset, dictionary):
    dataset = list()
    for doc in origin_dataset:
        data = []
        for word in doc:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
            data.append(index)
        data = np.array(data).reshape(1, -1)
        dataset.append(data)
    return dataset


def format_data(file_prefix, lemmatize=True, stem=False):
    with open(file_prefix + '.txt') as txt:
        doc = txt.read()

    words = re.sub("[^a-zA-Z]", " ", doc).split()
    word_len = len(words)
    key_list = []

    with open(file_prefix + '.key') as key:
        for key in key.readlines():
            key_words = re.sub("[^a-zA-Z]", " ", key).split()
            if len(key_words) > 0:
                key_list.append(key_words)

    label = []
    index = 0
    while index < word_len:
        find = False
        for key_words in key_list:
            if words[index] == key_words[0]:
                if len(key_words) == 1:
                    label.append(1)
                    index += 1
                    find = True
                    break
                elif word_len - index >= len(key_words):
                    match = True
                    for i in range(1, len(key_words)):
                        if words[index + i] != key_words[i]:
                            match = False
                            break
                    if match:
                        for i in range(len(key_words)):
                            label.append(1)
                        index += len(key_words)
                        find = True
                        break
        if not find:
            label.append(0)
            index += 1

    data = []
    for word in words:
        data.append(word.lower())

    if lemmatize:
        data = [lemmatizer.lemmatize(w) for w in data]

    if stem:
        data = [stemmer.stem(w) for w in data]

    label_array = np.zeros((word_len, 2), dtype=float)
    label_array[np.arange(word_len), label] = 1.0
    return data, label_array


def generate_data_set(path):
    files = os.listdir(path)
    data_set = []
    label_set = []
    for f in files:
        if not os.path.isdir(f) and f.endswith('.txt') and ('justTitle' not in f):
            print f
            data, label = format_data(path + '/' + f[:-4])
            # print data
            # print label
            data_set.append(data)
            label_set.append(label)
    return data_set, label_set

def generate_key_word(test_data, test_label):
    data_len = len(test_data)
    index = 0
    key_words = []
    while index < data_len:
        if test_label[index][0] >= test_label[index][1]:
            index += 1
        else:
            key_word = []
            while index < data_len and test_label[index][0] < test_label[index][1]:
                key_word.append(test_data[index])
                index += 1
            key_words.append(key_word)
    return key_words


def bidirectional_rnn(cell_fw, cell_bw, inputs,
                      initial_state_fw=None, initial_state_bw=None,
                      dtype=None, sequence_length=None, scope=None):
    name = scope or "BiRNN"
    # Forward direction
    with tf.variable_scope(name + "_FW"):
        output_fw, _ = tf.nn.rnn(cell_fw, inputs, initial_state_fw, dtype,
                                 sequence_length)
    # Backward direction
    with tf.variable_scope(name + "_BW"):
        tmp, _ = tf.nn.rnn(cell_bw, list(reversed(inputs)),
                           initial_state_bw, dtype, sequence_length)
    output_bw = list(reversed(tmp))
    # Concat each of the forward/backward outputs
    outputs = [array_ops.concat(1, [fw, bw])
               for fw, bw in zip(output_fw, output_bw)]
    return outputs


#train_data_set, train_label_set = generate_data_set("./keyword/train")
#test_data_set, test_label_set = generate_data_set("./keyword/test")
#train_data_set, count, dictionary, reverse_dictionary = build_fixed_size_dataset(train_data_set, 50000)
#test_data_set = build_test_set(test_data_set, dictionary)
#print 'train-data-set'
#print train_data_set
#print 'test-data-set'
#print test_data_set
