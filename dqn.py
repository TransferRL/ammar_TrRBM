import tensorflow as tf
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from collections import deque

class q_network(object):
    
    """
    This class builds and runs a neural network for deep q learning
    """
    
    def __init__(self
                 ,discount_rate = 0.9
                 ,mem_size = 5000
                 ,sample_size = 1000
                 ,n_hidden_layers = 2
                 ,activation = tf.nn.relu
                 ,opt = tf.train.MomentumOptimizer
                 ,opt_kws = {'learning_rate':0.001,'momentum':0.5}
                 ):
        self.discount_rate = discount_rate
        self.mem_size = mem_size
        self.sample_size = sample_size
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.opt = opt(**opt_kws)
        self.memory = deque([],mem_size)
        self.sess = None
        self.losses = []
        
    def initialize_graph(self):
        
        def weight_matrix(n_from,n_to):
            return tf.Variable(tf.truncated_normal(
                shape=(n_from,n_to)
                ,mean=0.0
                ,stddev=1/np.sqrt(n_from+n_to)
                ,dtype=tf.float32))

        def bias_matrix(n_to):
            return tf.Variable(tf.zeros(shape=(1,n_to),dtype=tf.float32))
        
        hidden_dict = {}
        
        input_layer = tf.placeholder(name='q_input', shape=(None,n_input),dtype=tf.float32)

        for n in range(1,self.n_hidden_layers+1):
            hidden_dict[n] = {}
            if n == 1:
                hidden_dict[n]['weights'] = weight_matrix(n_input,n_hidden)
                hidden_dict[n]['bias'] = bias_matrix(n_hidden)
                hidden_dict[n]['layer'] = self.activation(tf.matmul(input_layer,hidden_dict[n]['weights']) + hidden_dict[n]['bias'])
            else:
                hidden_dict[n]['weights'] = weight_matrix(n_hidden,n_hidden)
                hidden_dict[n]['bias'] = bias_matrix(n_hidden)
                hidden_dict[n]['layer'] = self.activation(tf.matmul(hidden_dict[n-1]['layer'],hidden_dict[n]['weights']) + hidden_dict[n]['bias'])


        output_weights = weight_matrix(n_hidden,n_output)
        output_bias = bias_matrix(n_output)

        output_pred = tf.matmul(hidden_dict[self.n_hidden_layers]['layer'],output_weights) + output_bias
        output_truth = tf.placeholder(shape=(None,n_output),dtype=tf.float32)

        loss = tf.reduce_mean(tf.nn.l2_loss(output_truth - output_pred))
        
        all_variables = [hidden_dict[n]['weights'] for n in range(1,self.n_hidden_layers+1)] + [output_weights] + [hidden_dict[n]['bias'] for n in range(1,self.n_hidden_layers+1)] + [output_bias]
        opt_op = self.opt.minimize(loss, var_list = all_variables)
        
        self.input_layer = input_layer
        self.output_pred = output_pred
        self.output_truth = output_truth
        self.loss = loss
        self.all_variables = all_variables
        self.opt_op = opt_op
    
    def open_session(self):
        self.sess = tf.Session()
        
    def close_session(self):
        self.sess.close()
        
    def initialize_new_variables(self):
        self.sess.run(tf.global_variables_initializer())
        
    def run_training(self,n_epochs, samples):
        """
        'samples' is a tuple with all states, coded_actions, transitions, rewards.
        'states' are the vectors of s as row vectors in matrix of samples.
        'coded_actions' are a single 1-D vector of actions, encoded as index of
            one-hot vector of actions, i.e. if one-hot = [0,1,0,0] then coded_output = 1.
            This hack allows for easy generation of ground_truth samples. 
        'transitions' are vectors of s' as row vectors in matrix of samples.
        'rewards' are a single 1-D vector of rewards.
        """
        
        states,coded_actions,transitions,rewards = samples
        
        pbar = tqdm(range(n_epochs))
        _losses = []
        for _ in pbar:

            bellman_trans_q = np.max(self.sess.run(self.output_pred, feed_dict= {self.input_layer:transitions}) ,axis=1)

            ground_truth = self.sess.run(self.output_pred, feed_dict={self.input_layer:states})
            ground_truth[list(range(len(states))),coded_actions] = rewards + (self.discount_rate * bellman_trans_q) #this is the bellman update for fitted q iteration, with all non-action outputs as the Q value that will be predicted by the current network - this negates any back-prop through these output nodes. 

            _, _loss = self.sess.run([self.opt_op,self.loss],feed_dict={self.input_layer:states, self.output_truth:ground_truth})
            self.losses.append(_loss)
            pbar.set_description('loss: {}'.format(_loss))
            
    def plot_loss(self):
        plt.plot(list(range(len(self.losses))),self.losses)
        plt.show()
        
    def add_new_obvs(self,obvs):
        for state,action,transition,reward in obvs:
            self.memory.append([state,action,transition,reward])

            
    def get_memory_sample(self, size = None):
        if size:
            _size = size
        else:
            _size = self.sample_size
        idx = np.random.choice(range(_size),_size)
        sample = np.array(self.memory)[idx]
        return sample[:,0], sample[:,1], sample[:,2], sample[:,3]
        
        
if __name__ == '__main__':
    
    n_samples = 10000
    n_input = 10
    n_output = 5
    n_hidden = 15

    actions = np.random.binomial(n_output-1,[1/n_output],n_samples)
    states = np.random.randn(n_input*n_samples).reshape(n_samples,n_input).astype(float)
    transitions = np.random.randn(n_input*n_samples).reshape(n_samples,n_input).astype(float)
    rewards = np.random.binomial(1,[1/5],n_samples)

    dqn = q_network(discount_rate = 0.9
                 ,mem_size = 5000
                 ,sample_size = 1000
                 ,n_hidden_layers = 2
                 ,activation = tf.nn.relu
                 ,opt = tf.train.MomentumOptimizer
                 ,opt_kws = {'learning_rate':0.0001,'momentum':0.2}
                 )
   
    
    dqn.initialize_graph()
    dqn.open_session()
    dqn.initialize_new_variables()
    
    dqn.add_new_obvs(zip(states,actions,transitions,rewards))
    dqn.run_training(500, dqn.get_memory_sample())
    dqn.plot_loss()
    
    dqn.add_new_obvs(zip(states,actions,transitions,rewards))
    dqn.run_training(500, dqn.get_memory_sample())
    dqn.plot_loss()