import pdb
import cv2
import sys
import os

from torch._C import device
sys.path.append("game/")
from math import *
import game.wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque # deque 是一个双端队列，可以从两头append数据
import torch
from model.net import *
from torch.autograd import Variable
import torch.nn as nn

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 1000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH_SIZE = 32 # size of minibatch
FRAME_PER_ACTION = 1
UPDATE_TIME = 100
width = 80
height = 80

def preprocess(observation):
    # 输入图像预处理：裁剪为80x80，并二值化处理(像素大于1的值全部置为255)
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    # print(observation.shape)
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    # print(observation.shape)
    # 返回一个灰度图
    return np.reshape(observation, (1,80,80))

def Rotation(img, angle):

    height,width=img.shape[:2]
    degree=angle

    #旋转后的尺寸

    heightNew=int(width*fabs(sin(radians(degree)))+height*fabs(cos(radians(degree))))
    widthNew=int(height*fabs(sin(radians(degree)))+width*fabs(cos(radians(degree))))
    matRotation=cv2.getRotationMatrix2D((width/2,height/2),degree,1)
    matRotation[0,2] +=(widthNew-width)/2 #重点在这步，目前不懂为什么加这步
    matRotation[1,2] +=(heightNew-height)/2 #重点在这步
    imgRotation=cv2.warpAffine(img,matRotation,(widthNew,heightNew),borderValue=(255,255,255))

    return imgRotation
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# DQN主体
class BrainDQNMain(object):
    # 保存模型参数
    def save(self):
        print("save model param")
        torch.save(self.Q_net.state_dict(), 'params3.pth')

    # 载入模型权重
    def load(self):
        if os.path.exists("params3.pth"):
            print("load model param")
            self.Q_net.load_state_dict(torch.load('params3.pth'))
            self.Q_netT.load_state_dict(torch.load('params3.pth'))

    # 参数定义及初始化
    def __init__(self,actions, net):
        self.replayMemory = deque() # 经验池
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON # 初始贪心值
        self.actions = actions
        self.Q_net=net # 实时更新网络
        self.Q_netT=net # 稳定学习网络
        self.load()
        self.loss_func=nn.MSELoss()
        LR=1e-6
        self.optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=LR)


    def train(self): # Step 1: obtain random minibatch from replay memory
        # 经验池随机抽样
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        # 分配参数 s,a,r,s'
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch] # Step 2: calculate y
        # 初始化 target
        y_batch = np.zeros([BATCH_SIZE,1])
        # 初始化下一状态(转为numpy矩阵)
        nextState_batch=np.array(nextState_batch) #print("train next state shape")
        #print(nextState_batch.shape)
        # 转化为tensor
        nextState_batch=torch.Tensor(nextState_batch)
        # 初始化动作(转为numpy矩阵)
        action_batch=np.array(action_batch)
        # 取当前最大动作值的索引
        index=action_batch.argmax(axis=1)
        # 32个动作值
        print("action "+str(index))
        # 矩阵转置
        index=np.reshape(index,[BATCH_SIZE,1])
        action_batch_tensor=torch.LongTensor(index)
        # Q_netT网络计算下一状态对应Q值
        QValue_batch = self.Q_netT(nextState_batch)
        QValue_batch=QValue_batch.detach().numpy()

        for i in range(0, BATCH_SIZE):
            # 游戏终止标志
            terminal = minibatch[i][4]
            # 如果终止直接将游戏定义的r返回
            if terminal:
                y_batch[i][0]=reward_batch[i]
            else:
                # 这里的QValue_batch[i]为数组，大小为所有动作集合大小，QValue_batch[i],代表
                # 做所有动作的Q值数组，y计算为如果游戏停止，y=rewaerd[i],如果没停止，则y=reward[i]+gamma*np.max(Qvalue[i])
                # 代表当前y值为当前reward+未来预期最大值*gamma(gamma:经验系数)
                y_batch[i][0]=reward_batch[i] + GAMMA * np.max(QValue_batch[i])
        
        # 每帧图像输入的Q值
        y_batch=np.array(y_batch)
        y_batch=np.reshape(y_batch,[BATCH_SIZE,1])
        # 状态输入转化为tensor
        state_batch_tensor=Variable(torch.Tensor(state_batch))
        # print(state_batch_tensor.shape)
        # Q_netT网络预测Q值转化为tensor
        y_batch_tensor=Variable(torch.Tensor(y_batch))
        # 实时预测Q_net网络，对输入计算Q值
        y_predict=self.Q_net(state_batch_tensor).gather(1,action_batch_tensor)
        # 计算loss
        loss=self.loss_func(y_predict,y_batch_tensor)
        print("loss is "+str(loss))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 满足训练轮次更新一次Q_netT网络权重
        if self.timeStep % UPDATE_TIME == 0:
            self.Q_netT.load_state_dict(self.Q_net.state_dict())
            self.save()

    # 定义
    def setPerception(self,nextObservation,action,reward,terminal): #print(nextObservation.shape)
        # s'状态
        newState = np.append(self.currentState[1:,:,:],nextObservation,axis = 0) # newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
        # 经验池更新，5元素元组
        self.replayMemory.append((self.currentState,action,reward,newState,terminal))
        # 经验池满即开始以旧换新
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        # 初始采集1000轮数据，再开始训练，不一定填充满经验池
        if self.timeStep > OBSERVE: # Train the network
            self.train()

        # print info
        state = ""
        # 不满足OBSERVE时为算法为观察状态
        if self.timeStep <= OBSERVE:
            state = "observe"
        # 探索期
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        # 探索完成后是训练期
        else:
            state = "train"
        print ("TIMESTEP", self.timeStep, "/ STATE", state, "/ EPSILON", self.epsilon)
        # 更新state
        self.currentState = newState
        # 更新步长
        self.timeStep += 1

    def getAction(self):
        currentState = torch.Tensor([self.currentState])
        QValue = self.Q_net(currentState)[0]
        action = np.zeros(self.actions)
        # 每一步都要进行贪心或随机决策，FRAME_PER_ACTION控制决策频率
        if self.timeStep % FRAME_PER_ACTION == 0:
            # 随即动作
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                print("choose random action " + str(action_index))
                action[action_index] = 1
            # 贪心动作，由最新Q网络预测得出
            else:
                action_index = np.argmax(QValue.detach().numpy())
                print("choose qnet value action " + str(action_index))
                action[action_index] = 1
        # 没有动作时，小鸟自然下落
        else:
            action[0] = 1  # do nothing

        # change episilon
        # 贪心值逐渐下降(动态调整)
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        return action

    # 一个state由四次连续observation组成，这是初始的输入，用的是四张同样的图片
    def setInitState(self, observation):
        self.currentState = np.stack((observation, observation, observation, observation),axis=0)
        # print(self.currentState.shape)

if __name__ == '__main__': 
    # Step 1: init BrainDQN
    actions = 2
    net = DP2() # 选择网络
    brain = BrainDQNMain(actions, net) # Step 2: init Flappy Bird Game
    flappyBird = game.GameState() # Step 3: play game
    # Step 3.1: obtain init state
    action0 = np.array([1,0]) # do nothing
    observation0, reward0, terminal = flappyBird.frame_step(action0)
    # cv2.imshow('img',observation0)
    observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
    # cv2.imshow('img',observation0)
    brain.setInitState(observation0)
    # print(brain.currentState.shape) # Step 3.2: run the game

    while True:
        action = brain.getAction()
        nextObservation,reward,terminal = flappyBird.frame_step(action)
        # print(nextObservation.shape)
        nextObservation = cv2.cvtColor(nextObservation, cv2.COLOR_BGR2RGB)
        nextObservation = cv2.flip(nextObservation,1)
        nextObservation = Rotation(nextObservation, 90)
        # cv2.imshow('game',nextObservation)
        # cv2.imwrite('game.jpg',nextObservation)
        nextObservation = preprocess(nextObservation)
        nextObservations = np.reshape(nextObservation, (80,80))
        # print(nextObservations.shape)
        # cv2.imshow('image',nextObservations)
        # cv2.imwrite('input.jpg',nextObservations)
        # print(nextObservation.shape)
        brain.setPerception(nextObservation,action,reward,terminal)
