import tensorflow as tf

from .. import xavier_init, model_session

class Weight():
    def __init__(self, in_size, out_size, activation, activation_size):
        """
        in_size -> int, size of the input dimension
        out_size -> int, size of the output dimension
        activation -> function used to activate the weight
        activation_size -> size of the intermediary activated tensor
        """
        self.in_size = in_size
        self.out_size = out_size
        self.activation = activation
        self.activation_size = activation_size

        self.weight1 = tf.Variable(xavier_init([in_size, activation_size]), name='weight1')
        self.bias1 = tf.Variable(tf.zeros(shape=[activation_size]), name='bias1')
        self.weight2 = tf.Variable(xavier_init([activation_size, out_size]), name='weight2')
        self.bias2 = tf.Variable(tf.zeros(shape=[out_size]), name='bias2')

    def apply(self, t_in):
        assert(t_in.shape[1] == self.in_size)

        in_rank = model_session.run(tf.rank(t_in))
        assert(in_rank == 2)

        t_1 = tf.matmul(t_in, self.weight1) + self.bias1
        t_2 = self.activation(t_1)
        return tf.matmul(t_2, self.weight2) + self.bias2

