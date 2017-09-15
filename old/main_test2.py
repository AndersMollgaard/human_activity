import tensorflow as tf
from data import prepare_data
import numpy as np

########################################

BATCH_SIZE = 128
LEARNING_RATE = 0.01

#############

def get_batch(X, Y, BATCH_SIZE):
    indices_batch = np.random.choice(Y.shape[0], BATCH_SIZE)
    X_batch = X[indices_batch].astype('float32')
    Y_batch = Y[indices_batch].astype('float32')
    return X_batch, Y_batch

#X, Y = prepare_data.get_data(8,0)
#Y = Y[:, 2]
with tf.Session() as sess:
    data = prepare_data.data_build_fn()
    X_, Y_ = data['X'], data['Y']
#    X_ = tf.placeholder(tf.float32, (BATCH_SIZE, 40))
#    Y_ = tf.placeholder(tf.float32, (BATCH_SIZE,))
    input_shape = tf.TensorShape([X_.get_shape()[1]])
    weights = tf.Variable(tf.random_normal(input_shape, 0.0, 0.1),
                              name='weights')
    bias = tf.Variable(0., dtype=tf.float32)
    # model
    logits = tf.tensordot(X_, weights, [[1], [0]]) + bias
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                       labels=Y_))
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    tf.global_variables_initializer().run()
    loss_sum = 0
    for i in range(10000):
        _, loss_ = sess.run([optimizer, loss])
#        X_batch, Y_batch = get_batch(X, Y, BATCH_SIZE)
#        feed_dict = {X_: X_batch, Y_: Y_batch}
#        _, loss_ = sess.run([optimizer, loss], feed_dict=feed_dict)
        loss_sum += loss_
        if i % 100 == 0:
            print(loss_sum/100)
            loss_sum = 0