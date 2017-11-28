import pickle
from matplotlib import pyplot as plt
from TrRBM_2d_3d_plots import plot_with_gp


if __name__ == '__main__':

    with open('exp_data/mountainCarNoTransfer.pkl', 'rb') as f:
        x, rewards, steps = pickle.load(f)

    with open('exp_data/mountainCarTransfer.pkl', 'rb') as f:
        x_t, rewards_t, steps_t = pickle.load(f)


    plt.plot(rewards, 'b')
    plt.plot(rewards_t, 'r')
    plt.show()
    plot_with_gp(x, rewards, title='reward per episode', xlabel='episode', ylabel='reward', plt_lable='no transfer',
                 color='r')
    plot_with_gp(x_t, rewards_t, title='reward per episode', xlabel='episode', ylabel='reward',
                 plt_lable='with transfer', color='b')
    plt.legend(loc='upper right')
    plt.show()

    plot_with_gp(x, steps, title='steps per episode', xlabel='episode', ylabel='steps', plt_lable='no transfer',
                 color='r')
    plot_with_gp(x_t, steps_t, title='steps per episode', xlabel='episode', ylabel='steps',
                 plt_lable='with transfer', color='b')
    plt.legend(loc='upper right')
    plt.show()
