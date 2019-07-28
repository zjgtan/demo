# coding: utf8
import tensorflow as tf


class TextCNN(object):
    
    def __init__(self, vocab_size, seq_len, num_classes):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_classes = num_classes

        self.embedding_dim = 64
        self.num_filters = 100
        self.kernel_sizes = [3, 4, 5]
        self.hidden_dim = 100
        self.learning_rate = 1e-3

        self.build_graph()

    def feed_dict(self, sentences, labels):
        return {self.input_sentence: sentences, self.input_label: labels}

    def build_graph(self):
        self.input_sentence = tf.placeholder(tf.int32, [None, self.seq_len], name='input_sentence')
        self.input_label = tf.placeholder(tf.int64, [None], name='input_label')

        with tf.device("/cpu:0"):
            embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_dim])
            embedding_sentence = tf.nn.embedding_lookup(embedding, self.input_sentence)

        with tf.name_scope("cnn"):
            pools = []
            for kernel_size in self.kernel_sizes:
                conv = tf.layers.conv1d(embedding_sentence, self.num_filters, kernel_size, name='conv%d' % (kernel_size))
                pool = tf.reduce_max(conv, reduction_indices=[1], name='pool%d' % (kernel_size))
                pools.append(pool)

            pool = tf.concat(pools, axis=1)

        with tf.name_scope("fc"):
            fc = tf.layers.dense(pool, self.hidden_dim, name='fc1')
            #fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            self.logits = tf.layers.dense(fc, self.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope("optimize"):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_label)
            self.loss = tf.reduce_mean(cross_entropy)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(self.input_label, self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
