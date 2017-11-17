from dqn_baseline import main
import tensorflow as tf
from envs import ENVS_DICTIONARY

params_dictionary = {}
params_dictionary["discount_rate"] = 0.9
params_dictionary["mem_size"] = 100
params_dictionary["sample_size"] = 50
params_dictionary["n_hidden_layers"] = 2
params_dictionary["n_hidden_units"] = 16
params_dictionary["activation"] = tf.nn.relu
params_dictionary["optimizer"] = tf.train.MomentumOptimizer
params_dictionary["opt_kws"] = {'learning_rate':0.01,'momentum':0.2}
params_dictionary["n_episodes"] = 500
params_dictionary["n_epochs"] = 5
params_dictionary["retrain_period"] = 1
params_dictionary["epsilon"] = 0.5
params_dictionary["epsilon_decay"] = 0.999
params_dictionary["ini_steps_retrain"] = 50

main(ENVS_DICTIONARY['2DMountainCar'],'3DMC',2,3,params_dictionary)
