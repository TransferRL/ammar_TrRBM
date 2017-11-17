import sys
sys.path.append('../taylor_master/lib/env/')
from threedmountain_car import ThreeDMountainCarEnv

from gym import envs

ENVS_DICTIONARY = {
    '3DMountainCar':ThreeDMountainCarEnv
    ,'2DMountainCar':envs.make('MountainCar-v0')
    ,'CartPole':envs.make('CartPole-v1')
#    ,'InvertedPendulum':envs.make('InvertedPendulum-v1')
#    ,'DoublePendulum':envs.make('InvertedDoublePendulum-v1')
    
    }


