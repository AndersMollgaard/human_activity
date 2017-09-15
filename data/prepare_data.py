import numpy as np
import pickle
import tensorflow as tf
import sys

##########################################################

DATA_DIR = '/home/amollgaard/Data/Telefon'

########## Load data #########################

def get_sigdic():
    '''Load sigdic - a dictionary with users as keys and time series 
    arrays as values'''
    with open('%s/signal_filtered.p' %DATA_DIR, 'rb') as f:
        sigdic = pickle.load(f)
    return sigdic

def make_entries(length=5,future=0):
    '''Return xentries and yentries that together define the indices for
    a vector in a signal: vector = signal[offset + xentries, yentries].'''
    entries = []
    for i in range(length):
        for j in range(5):
            if i == length-1:
                entries.append((i+future,j))
            else:
                entries.append((i,j))
    xentries = np.array([ elem[0] for elem in entries ])
    yentries = np.array([ elem[1] for elem in entries ])
    return xentries,yentries

################ sigdic to arrays #################################
# this can become a problem for memory

#def sigdic_to_arrays(sigdic, xentries=np.array([0,1,2,3]), yentries=np.array([0,0,0,0])):
#    X = []
#    Y = []
#    for ii,(user,signal) in enumerate(sigdic.items()):
#        print(ii, user)
#        signal_length = signal.shape[0]
#        entries_length = max(xentries)
#        for offset in range(signal_length-entries_length-1):
#            vec = signal[offset + xentries, yentries]
#            if -1 not in vec:
#                X.append(np.copy(vec[:-5]))
#                Y.append(np.copy(vec[-5:]))
#            del vec
#    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
#
#def get_data(history_length=8, future=0, save=True):
#    try:
#        with open('%s/activity_hist=%d_fut=%d.p' %(DATA_DIR, history_length, future), 'rb') as f:
#            X, Y = pickle.load(f)
#        print('Data loaded.')
#        return X.astype(np.float32), Y.astype(np.float32)
#    except:
#        global sigdic
#        try:
#            sigdic
#        except:
#            sigdic = get_sigdic()
#        xentries, yentries = make_entries(history_length+1, future)
#        X, Y = sigdic_to_arrays(sigdic, xentries, yentries)
#        if save:
#            with open('%s/activity_hist=%d_fut=%d.p' %(DATA_DIR, history_length, future), 'wb') as f:
#                pickle.dump((X, Y), f)
#        print('Data prepared.')
#        return X, Y

################ sigdic to csv #########################################

def sigdic_to_csv(history_length=8, future=0, test_size=0.05):
    # no need to load sigdic if it already is loaded
    global sigdic
    try:
        sigdic
    except:
        sigdic = get_sigdic()
    # create the entries that define vector samples from the signals
    xentries, yentries = make_entries(history_length+1, future)
    # open the csv files to be written to
    with open('%s/activity_train.csv' %DATA_DIR, 'w') as f_train, \
            open('%s/activity_test.csv' %DATA_DIR, 'w') as f_test:
        # iterate over the users and signals in sigdic
        for ii,(user,signal) in enumerate(sigdic.items()):
            print(ii, user)
            signal_length = signal.shape[0]
            entries_length = max(xentries)
            # iterate over the signal
            for offset in range(signal_length-entries_length-1):
                # create a vector with each iteration
                vec = signal[offset + xentries, yentries]
                # check that the vector does not contain missing values
                if -1 not in vec:
                    # add the vector with some probability to either
                    # test or training set
                    if np.random.random() < test_size: 
                        f = f_test
                    else:
                        f = f_train
                    # create a string from the vector
                    for elem in vec[:-1]:
                        f.write('%d ' %elem)
                    # write the vector to file
                    f.write('%d\n' %vec[-1])
                        
#################### dataset API ###############################

def data_build_fn(history=9, channel=0, mode='train', batch_size=128, reshape=False):
    '''Build the ops for reading patches of data from csv files.'''
    filenames = ['%s/activity_%s.csv' %(DATA_DIR, mode)]
    # create a dataset object from the filename
    dataset = tf.contrib.data.TextLineDataset(filenames)
    if mode == 'train':
        # buffer should allow datapoints to be sufficiently mixed
        # for vectors of length 2, there are 13 * 10**6 data points
        dataset = dataset.shuffle(buffer_size=5*10**6)
        # iterating the dataset should return batches
        dataset = dataset.batch(batch_size)
        # repeat will allow us to iterate over several epochs
        dataset = dataset.repeat()
    if mode == 'test':
        dataset = dataset.batch(batch_size)
        # one should only iterate over one epoch of test data
        dataset.repeat(1)
    # an iterator object is created from the dataset object
    iterator = dataset.make_one_shot_iterator()
    # the iterator returns a list of string tensors
    next_elem = iterator.get_next()
    # a list of float tensors is created from the string tensors
    # note that each tensor is a column and not a row (i.e. tensor != data point)
    next_elem = tf.decode_csv(next_elem, (history+1)*5*[tf.zeros(1, dtype=tf.float32)], 
                              field_delim=' ')
    # we stack the list of tensors into one batch tensor (only history values)
    X = tf.transpose(tf.stack(next_elem[:-5]))
    # in case of LSTM we want the time steps to be ordered into columns
    if reshape:
        X = tf.reshape(X, [batch_size, 5, history])
    # the labels are created
    Y = next_elem[-5 + channel]
    data = {'X': X, 'Y': Y}
    return data

#def npdata_build_fn(batch_size=128, mode='train'):
#    X, Y = get_data(2, 0)
#    Y = Y[:,-3]
#    dataset = tf.contrib.data.Dataset.from_tensor_slices((X, Y))
#    if mode == 'train':
#        dataset = dataset.shuffle(buffer_size=10000)
#        dataset = dataset.batch(batch_size)
#        dataset = dataset.repeat()
#    if mode == 'test':
#        dataset = dataset.batch(batch_size)
#        dataset.repeat(1)
#    iterator = dataset.make_one_shot_iterator()
#    X, Y = iterator.get_next()
#    data = {'X': X, 'Y': Y}
#    return data

################# Main ###########################################

if __name__ == '__main__':
    history = int(sys.argv[1])
    future = int(sys.argv[2])
    print('History:', history)
    print('Future:', future)
    sigdic_to_csv(history, future)