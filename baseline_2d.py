from dqn_baseline import main
import tensorflow as tf
from envs import ENVS_DICTIONARY

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 1e-3
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           10000, 1e-4, staircase=True)

params_dictionary = {}
params_dictionary["discount_rate"] = 1.0
params_dictionary["mem_size"] = 50000
params_dictionary["sample_size"] = 32
params_dictionary["n_hidden_layers"] = 1
params_dictionary["n_hidden_units"] = 32
params_dictionary["activation"] = tf.nn.relu
params_dictionary["optimizer"] = tf.train.MomentumOptimizer
params_dictionary["opt_kws"] = {'learning_rate':0.01,'momentum':0.2}
params_dictionary["n_episodes"] = 200
params_dictionary["n_epochs"] = 5
params_dictionary["retrain_period"] = 1
params_dictionary["epsilon"] = 0.1
params_dictionary["epsilon_decay"] = 1
params_dictionary["ini_steps_retrain"] = 1000

main(ENVS_DICTIONARY['2DMountainCar'],'2DMC',2,3,params_dictionary)
