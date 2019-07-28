# coding: utf8
import numpy as np
import tensorflow as tf

def read_file(filename, word_dict, sentences, labels):

    max_len = 0

    for line in file(filename):
        words = line.rstrip().split(" ")
        sentence = []
        for word in words:
            if word not in word_dict:
                word_dict[word] = len(word_dict)
            sentence.append(word_dict[word])
        if len(sentence) > max_len:
            max_len = len(sentence)

        sentences.append(np.array(sentence))
        if filename.endswith("pos"):
            labels.append(1)
        else:
            labels.append(0)

    return max_len

def load_data(data_dir):
    word_dict = {"<PAD>": 0}
    sentences = []
    labels = []
    max_len = []

    max_len.append(read_file(data_dir + "/rt-polarity.pos", word_dict, sentences, labels))
    max_len.append(read_file(data_dir + "/rt-polarity.neg", word_dict, sentences, labels))

    max_len = max(max_len)
    indices = np.random.permutation(np.arange(len(sentences)))

    sentences = np.array(sentences)
    sentences = tf.keras.preprocessing.sequence.pad_sequences(sentences,
                value=word_dict["<PAD>"], padding='post', maxlen=max_len)
    labels = np.asarray(labels)

    sentences = sentences[indices] 
    labels = labels[indices]

    return sentences, labels, word_dict, max_len


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1 

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

