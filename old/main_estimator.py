import tensorflow as tf
from sklearn import model_selection
from data import prepare_data
import sys

tf.logging.set_verbosity(tf.logging.INFO)

############# Variables ################################

DATA_DIR = '/home/amollgaard/Data/Telefon'
BATCH_SIZE = 128
STEPS = 5 * 10**4
LEARNING_RATE = 0.01
STEPS_PRINT = 500
MODEL_DIR = 'results/linear_estimator'

############## The model ########################################
    
def linear_model(features, labels, mode):
    '''Linear model to be fed the estimator class.'''
    X = features['X']
    # variables
    input_length = tf.cast(X.get_shape()[1], dtype=tf.int32)
    weights = tf.Variable(tf.random_normal((input_length,), 0.0, 0.1),
                          name='weights', dtype=tf.float32)
    bias = tf.Variable(0., dtype=tf.float32, name='bias')
    # model
    logits = tf.add(tf.tensordot(X, weights, [[1], [0]]), bias, 'logits')
    # predictions
    classes = tf.greater(logits, tf.zeros_like(logits), name='classes')
    predictions = {
            "classes": classes,
            'probabilities': tf.nn.sigmoid(logits, name="sigmoid")
            }
    # predict mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    # calculate the loss for train and eval mode
    loss = tf.losses.sigmoid_cross_entropy(labels, logits)
    # summaries
    tf.summary.scalar('loss', loss)
    tf.summary.histogram('weights', weights)
    # train mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, 
                                          train_op=train_op)
    # eval mode
    eval_metric_ops = {
            'auc': tf.metrics.auc(labels=labels, predictions=predictions['probabilities']),
            "accuracy": tf.metrics.accuracy(
                    labels=labels, predictions=predictions["classes"])
            }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          eval_metric_ops=eval_metric_ops)

############ Main #########################
        
def main(history_length=8, future=0, test_size=0.05):
    # check if data is loaded and load if not
    global X, Y, X_train, X_test, Y_train, Y_test
    try:
        X, Y
    except:
        X, Y = prepare_data.get_data(history_length, future, True)
        Y[Y==-1] = 0
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size)
    # create estimator
    for datatype in range(5):
        model_dir = '%s/type=%d_hist=%d_fut=%d' %(MODEL_DIR, datatype, history_length, future)
        linear_estimator = tf.estimator.Estimator(model_fn=linear_model, 
                                                  model_dir=model_dir)
        # create hooks for logging
        tensors_to_log = {'probabilities': 'sigmoid'}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=STEPS_PRINT)
        # train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x = {'X': X_train},
                y = Y_train[:, datatype],
                batch_size = BATCH_SIZE,
                num_epochs=None,
                shuffle=True)
        linear_estimator.train(train_input_fn,
                               steps=STEPS,
                               hooks=[logging_hook])
        # evaluate the model
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x = {'X': X_test},
                y = Y_test[:, datatype],
                num_epochs = 1,
                shuffle = False)
        eval_results = linear_estimator.evaluate(input_fn=eval_input_fn)
        print(eval_results)

if __name__ == '__main__':
    main(8, int(sys.argv[1]))

########## CRAP CODE ############################
    

#def main():
#    # create estimator
#    linear_estimator = tf.estimator.Estimator(model_fn=linear_model, 
#                                              model_dir='%s' %(MODEL_DIR))
#    # create hooks for logging
#    tensors_to_log = {'probabilities': 'sigmoid'}
#    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=STEPS_PRINT)
#    # train the model
#    X, Y = prepare_data.get_XY(BATCH_SIZE, 'train')
#    linear_estimator.train(lambda: (X, Y),
#                           steps=STEPS,
#                           hooks=[logging_hook])
    