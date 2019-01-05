import tensorflow as tf

class Network:
    def __init__(self, fc_nums, activation_fn, input_num, learning_rate):
        self.fc_nums = fc_nums
        self.activation_fn = activation_fn
        self.input_num = input_num
        self.learning_rate = learning_rate
        self.train_step = 0
        self.init_network()

        conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        conf.gpu_options.allow_growth=True

        self.sess = tf.Session(config=conf)
        self.sess.run(tf.initialize_all_variables())

    def init_network(self):
        self.input_ph = tf.placeholder(tf.float32, [None, self.input_num], name="input_ph")
        self.standard_y_ph = tf.placeholder(tf.int32, [None], name="y_ph")
        self.y = tf.one_hot(self.standard_y_ph, self.fc_nums[-1], 1, 0)
        # print(self.y)
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
            self.logits = out
            print(self.logits)

        self.out = tf.argmax(input=self.logits, axis=1)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
        self.loss = tf.reduce_mean(self.cross_entropy)

        self.trainer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def train(self, input_x, input_y):
        # print(input_x.shape)
        # print(input_y.shape)
        self.train_step += 1
        loss, _ = self.sess.run([self.loss, self.trainer], feed_dict={
            self.input_ph: input_x,
            self.standard_y_ph: input_y
        })
        if self.train_step % 100 == 0:
            print("now train step: %d"%self.train_step, "now loss % f"%loss)
        return self.loss

    def output(self, input_x):
        return self.sess.run(self.out, feed_dict={
            self.input_ph: input_x
        })
