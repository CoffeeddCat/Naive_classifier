import tensorflow as tf

class Autoencoder:
    def __init__(self, fc_nums, activation_fn, input_num, learning_rate, model_load_path):
        self.fc_nums = fc_nums
        # self.fc_nums = [2048, 1024, 256]
        self.activation_fn = activation_fn
        self.input_num = input_num
        self.learning_rate = learning_rate
        self.model_load_path = model_load_path
        self.train_step = 0

        self.init_network()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.variables_initializer())

    def init_network(self):
        self.input_ph = tf.placeholder(tf.float32, [None, self.input_num])
        with tf.variable_scope('fc', reuse=False):
            out = self.input_ph
            for output_num in self.fc_nums:
                layer = tf.layers.dense(
                    inputs=out,
                    units=output_num,
                    activation=self.activation_fn
                )
                out = layer

            self.auto_encoder_out = out

            for output_num in self.fc_nums.reverse():
                layer = tf.layers.dense(
                    inputs=out,
                    units=output_num,
                    activation=self.activation_fn
                )
                out = layer
            self.output_y = tf.layers.dense(
                inputs=out,
                units=self.input_num,
                activation=None
            )
        self.loss = tf.reduce_mean(tf.square(self.input_ph - self.output_y))
        self.trainer = tf.train.Optimizer.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def train(self, input_x):
        self.train_step += 1
        self.sess.run([self.loss, self.trainer], feed_dict={
            self.input_ph: input_x
        })
        if self.train_step % 100 == 1 :
            self.model_save()
        return self.loss

    def output(self, input_x):
        return self.sess.run(self.auto_encoder_out, feed_dict={
            self.input_ph: input_x
        })

    def model_save(self):
        self.saver.save(self.sess, "../saved_model/auto_encoder/", global_step=self.train_step)

    def model_load(self):
        self.saver.restore(self.sess, self.model_load_path)


