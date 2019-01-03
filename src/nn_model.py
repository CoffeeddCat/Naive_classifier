import tensorflow as tf

class Network:
    def __init__(self, fc_nums, activation_fn, input_num, learning_rate):
        self.fc_nums = fc_nums
        self.activation_fn = activation_fn
        self.input_num = input_num
        self.learning_rate = learning_rate

        self.init_network()

        self.sess = tf.Session()
        self.sess.run(tf.variables_initializer())

    def init_network(self):
        self.input_ph = tf.placeholder(tf.float32, [None, self.input_num])
        self.standard_y_ph = tf.placeholder(tf.float32, [None, 1])
        with tf.variable_scope('fc', reuse=False):
            out = self.input_ph
            for output_num in self.fc_nums:
                layer = tf.layers.dense(
                    inputs=out,
                    units=output_num,
                    activation=self.activation_fn,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003)
                )
                out = layer
            self.logits = tf.nn.softmax(logits=out)

        self.out = tf.argmax(input=self.logits, axis=1)
        self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.standard_y_ph, logits=self.logits)
        self.loss = tf.reduce_mean(self.cross_entropy)

        self.trainer = tf.train.Optimizer.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def train(self, input_x):
        self.sess.run([self.loss, self.trainer], feed_dict={
            self.input_ph: input_x
        })
        return self.loss

    def output(self, input_x, input_y):
        return self.sess.run(self.out, feed_dict={
            self.input_ph: input_x,
            self.out: input_y
        })