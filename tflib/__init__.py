import tensorflow as tf

model_session = tf.Session()

def xavier_init(size):
    """
    Xavier Init is used to assign initial values to our weights
    you can read more here: https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/
    """
    with tf.name_scope('xavier_init'):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random.normal(shape=size, stddev=xavier_stddev)