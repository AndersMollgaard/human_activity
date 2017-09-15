import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics
from sklearn import model_selection

############# Variables ################################

DATA_DIR = '/home/amollgaard/Data/Telefon'
FILE = 'activity_hist=8_fut=0.p'
BATCH_SIZE = 128
STEPS = 10**4
LEARNING_RATE = 0.01
STEPS_PRINT = 200

############# Loads data #################################

def get_data():
    with open('%s/%s' %(DATA_DIR, FILE), 'rb') as f:
        X, Y = pickle.load(f)
    return X, Y[:,2]


def get_batch(X, Y, BATCH_SIZE):
    indices_batch = np.random.choice(Y.shape[0], BATCH_SIZE)
    X_batch = X[indices_batch].astype('float32')
    Y_batch = Y[indices_batch].astype('float32')
    return X_batch, Y_batch

############## Helper functions ###############################
    
def calc_auc(Y, Y_pred):
    # Calculate the roc curve
    fpr, tpr, thresholds = metrics.roc_curve(Y, Y_pred)
    auc = metrics.auc(fpr, tpr)
    # Plot the roc curve
#    plt.figure()
#    plt.plot(fpr, tpr, color='darkorange', lw=2, 
#             label='ROC curve (area = %0.2f)' % auc)
#    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#    plt.xlim([0.0, 1.0])
#    plt.ylim([0.0, 1.05])
#    plt.xlabel('False Positive Rate')
#    plt.ylabel('True Positive Rate')
#    plt.legend(loc="lower right")
#    plt.show()
    # 
    return auc

############## The model ########################################
    
class LinearModel():
    
    def __init__(self, input_length, sess):
        self.input_length = input_length
        self.sess = sess
    
    def build_graph(self):
        # placeholders
        X_ = tf.placeholder(tf.float32, (None, self.input_length), 'input')
        Y_ = tf.placeholder(tf.float32, (None,), 'target')
        # variables
        weights = tf.Variable(tf.random_normal((self.input_length,), 0.0, 0.1),
                              name='weights')
        bias = tf.Variable(0., dtype=tf.float32)
        # model
        logits = tf.tensordot(X_, weights, [[1], [0]]) + bias
        Y_pred = tf.nn.sigmoid(logits)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                       labels=Y_))
        # summaries
        with tf.name_scope('summaries'):
            tf.summary.scalar("loss", loss)
            tf.summary.histogram("loss", loss)
            summary = tf.summary.merge_all()
        # optimizer
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
        # evaluation (crashes when called)
        auc = tf.metrics.auc(labels=Y_, predictions=Y_pred)
        # initialize graph
        tf.global_variables_initializer().run()
        # add operations to model
        self.X_ = X_
        self.Y_ = Y_
        self.logits = logits
        self.Y_pred = Y_pred
        self.loss = loss
        self.summary = summary
        self.optimizer = optimizer
        self.auc = auc
    
    def train(self, X, Y, steps=STEPS, batch_size=BATCH_SIZE, steps_print=STEPS_PRINT):
        '''Train the model on random batches of data points for a given number
        of steps.'''
        # initialize a writer for summaries
        writer = tf.summary.FileWriter('results/linear_model')
        # loss_sum is used to calc a moving average
        loss_sum = 0
        for step in range(steps):
            # get a batch and make an update
            X_batch, Y_batch = get_batch(X, Y, batch_size)
            feed_dict = {self.X_:X_batch, self.Y_:Y_batch}
            _, loss = self.sess.run([self.optimizer, self.loss], feed_dict)
            # add loss to moving average
            loss_sum += loss
            # every steps_print
            if step % steps_print == 0:
                # print moving average
                loss_avg = loss_sum / steps_print
                print('Avg loss: %s' %loss_avg)
                loss_sum = 0.
                # write summary
                summary = self.sess.run(self.summary, feed_dict)
                writer.add_summary(summary, global_step=step)
        # close the writer
        writer.close()
    
    def evaluate(self, X, Y):
        '''Evaluate the loss and auc on the input (typically test data).'''
        feed_dict = {self.X_:X, self.Y_:Y}
        Y_pred, loss = self.sess.run([self.Y_pred, self.loss], feed_dict)
        print('Test loss: %s' %loss)
        auc = calc_auc(Y, Y_pred)
        print('Test auc: %.3f' %auc)
        corr = np.corrcoef(Y, Y_pred)[0,1]
        print('Test corr: %.3f' %corr)
    
    def predict(self, X):
        '''Return probability predictions, Y_pred, given input data, X.'''
        feed_dict = {self.X_:X}
        Y_pred = self.sess.run(self.Y_pred, feed_dict)
        return Y_pred

############# Main function ############################

def main(train_split=0.95):
    # check if data is loaded and load if not
    try:
        global X, Y, X_train, X_test, Y_train, Y_test
        X, Y
    except:
        X, Y = get_data()
        Y[Y==-1] = 0
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.05)
    # start session
    with tf.Graph().as_default(), tf.Session() as sess:
        # build graph
        model = LinearModel(X.shape[1], sess=sess)
        model.build_graph()
        # train model and evaluate
        model.train(X_train, Y_train, STEPS, BATCH_SIZE, STEPS_PRINT)
        model.evaluate(X_test, Y_test)