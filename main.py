import tensorflow as tf
from mytensorflow.models.LinearClassifier import LinearClassifier
from mytensorflow.models.DNNPlusClassifier import DNNPlusClassifier
from mytensorflow.models.LSTMClassifier import LSTMClassifier
from data import prepare_data
import sys

##################################################################

def main(model_class='LinearClassifier', BATCH_SIZE=128, LEARNING_RATE=0.002, STEPS=10**4, 
         STEPS_LOG=100, STEPS_SAVE=1000, history=8, future=0, channel=0):
    # initialize model class
    if model_class == 'LinearClassifier':
        hparams = {'learning_rate': LEARNING_RATE,
                   'history': history,
                   'future': future,
                   'channel': channel}
        model = LinearClassifier('results/LinearClassifier', hparams)
    if model_class == 'DNNPlusClassifier':
        hparams = {'learning_rate': LEARNING_RATE,
                   'history': history,
                   'future': future,
                   'channel': channel,
                   'n_hiddens': [5 * history, 5 * history]}
        model = DNNPlusClassifier('results/DNNPlusClassifier', hparams)
    if model_class == 'LSTMClassifier':
        hparams = {'learning_rate': LEARNING_RATE,
                   'history': history,
                   'future': future,
                   'channel': channel,
                   'n_hiddens': [5 * history, 5 * history]}
        model = LSTMClassifier('results/LSTMClassifier', hparams)
    # get function to build data ops and define data args
    data_build_fn = prepare_data.data_build_fn
    data_build_args_train = {'batch_size': BATCH_SIZE, 'mode': 'train', 
                             'history': history, 'channel': channel,
                             'reshape': model_class == 'LSTMClassifier'}
    data_build_args_eval = {'batch_size': BATCH_SIZE, 'mode': 'test', 
                            'history': history, 'channel': channel,
                            'reshape': model_class == 'LSTMClassifier'}
    # train the model
    model.train(data_build_fn, data_build_args_train, steps=STEPS, steps_log=STEPS_LOG, 
                steps_save=STEPS_SAVE)
    # choose eval metrics
    model.evaluate(data_build_fn, data_build_args_eval)
    

if __name__ == '__main__':
    main(model_class=sys.argv[1], history=int(sys.argv[2]), future=int(sys.argv[3]), 
         channel=int(sys.argv[4]))