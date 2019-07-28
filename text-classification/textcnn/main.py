# coding: utf8
import sklearn
from sklearn.model_selection import KFold
import tensorflow as tf

from preprocess import load_data, batch_iter
from model import TextCNN

num_epoch = 10
batch_size = 128

if __name__ == "__main__":
    sentences, labels, word_dict, max_len = load_data("./rt-polaritydata") 

    print "maxlen: %d, vocab_size: %d" % (max_len, len(word_dict)) 

    kFold = KFold(n_splits = 10) 

    model = TextCNN(len(word_dict), max_len, 2)

    val_accs = []

    for train_index, test_index in kFold.split(sentences):
        sentences_train, sentences_test = sentences[train_index], sentences[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

        session = tf.Session()
        session.run(tf.global_variables_initializer())

        best_acc = 0
        for epoch in range(num_epoch):
            for batch_idx, (sentences_batch, labels_batch) in enumerate(batch_iter(sentences_train, labels_train, batch_size)):
                feed_dict = model.feed_dict(sentences_batch, labels_batch)
                loss_train, _ = session.run([model.loss, model.optim], feed_dict=feed_dict)

                if batch_idx % 50 == 0: 
                    print "Epoch: %d, Batch: %d, Train Loss: %f" % (epoch, batch_idx, loss_train)

            feed_dict = model.feed_dict(sentences_test, labels_test)
            acc_test, = session.run([model.acc], feed_dict=feed_dict)
            if best_acc < acc_test:
                best_acc = acc_test

        val_accs.append(best_acc)

    print "Val Acc: %f" % (sum(val_accs) / len(val_accs))
