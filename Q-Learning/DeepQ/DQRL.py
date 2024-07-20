import torch
from torch import nn, optim

import random
import gymnasium as gym
from collections import deque

# enable GPU acceleration if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReplayMemory():
    """
    initialises empty experience buffers for agent transitions
    :param capacity: maximum transitions buffer can store
    :type capacity: int
    """
    def __init__(self, capacity:int) -> None:
        self.bufferS  = deque([], maxlen=capacity)
        self.bufferA  = deque([], maxlen=capacity)
        self.bufferR  = deque([], maxlen=capacity)
        self.bufferS_ = deque([], maxlen=capacity)
        self.bufferT  = deque([], maxlen=capacity)
    
    def __len__(self) -> int:
        """returns the number of transitions stored"""
        return len(self.bufferR)
    
    def store(self, s, a, r, s_, t) -> None:
        """
        pushes the passed transition to the top of respective buffers
        :param s:  start state
        :param a:  action taken
        :param r:  reward value
        :param s_: next state
        :param t:  terminal?
        """
        self.bufferS.append(s)
        self.bufferA.append(a)
        self.bufferR.append(r)
        self.bufferS_.append(s_)
        self.bufferT.append(t)
    
    def sampleBatch(self, batchSize:int):
        """
        returns a random uniformly sampled batch of transitions
        :param batchSize: size of batch
        :type batchSize: int
        """
        indices = random.sample(population=range(len(self)), k = batchSize)
        batchS  = torch.FloatTensor([self.bufferS[idx]  for idx in indices]).to(DEVICE)
        batchA  = torch.LongTensor ([self.bufferA[idx]  for idx in indices]).to(DEVICE).view(-1, 1)
        batchR  = torch.FloatTensor([self.bufferR[idx]  for idx in indices]).to(DEVICE).view(-1, 1)
        batchS_ = torch.FloatTensor([self.bufferS_[idx] for idx in indices]).to(DEVICE)
        batchT  = torch.IntTensor  ([self.bufferT[idx]  for idx in indices]).to(DEVICE).view(-1, 1)
        return batchS, batchA, batchR, batchS_, batchT

class FCQNetwork(nn.Module):
    """
    :param numInputs: dimension of observation space
    :type numInputs: int
    :param numActions: cardinality of action space
    :type numActions: int
    :param nodes: sizes of hidden fully connected layers
    :type nodes: tuple
    """
    def __init__(self, numInputs:int, numActions:int, nodes:tuple) -> None:
        super(FCQNetwork, self).__init__()
        
        # creating fully connected neural network with the given layer sizes
        self.model = nn.Sequential()
        self.model.append(nn.Linear(numInputs, nodes[0]))
        self.model.append(nn.ReLU())
        for k in range(len(nodes)-1):
            self.model.append(nn.Linear(nodes[k], nodes[k+1]))
            self.model.append(nn.ReLU())
        self.model.append(nn.Linear(nodes[-1], numActions))
        self.model.to(device=DEVICE)
    
    def predict(self, state) -> torch.Tensor:
        """
        performs forward pass through FCQNN and returns all state-action values 
        :param state: state
        """
        state = torch.tensor(state).to(device=DEVICE)
        return self.model.forward(state)

class NFQ():
    """
    initialises Neural Fitted Q-value RL agent
    :param learningRate: ALPHA
    :type learningRate: float
    :param discountRate: GAMMA
    :type discountRate: float
    :param epsMax: initial epsilon
    :type epsMax: float
    :param epsMin: final epsilon
    :type epsMin: float
    :param epsDec: episodes to linearly decay epsilon
    :type epsDec: int
    :param numInputs: dimension of observation space
    :type numInputs: int
    :param numActions: cardinality of action space
    :type numActions: int
    :param nodes: sizes of hidden fully connected layers
    :type nodes: tuple
    :param capacity: maximum transitions to keep in replay memory
    :type capacity: int
    :param batchSize: size of batch of transitions to train FCQNN
    :type batchSize: int
    """
    def __init__(self,
                 learningRate:float, discountRate:float,
                 epsMax:float, epsMin:float, epsDec:int,
                 numInputs:int, numActions:int, nodes:tuple,
                 capacity:int, batchSize:int) -> None:
        # hyperparameters
        self.ALPHA      = learningRate
        self.GAMMA      = discountRate
        self.EPS_MAX    = epsMax
        self.EPS_MIN    = epsMin
        self.EPS_DEC    = epsDec
        self.epsilon    = epsMax
        self.numInputs  = numInputs
        self.numActions = numActions

        # instantiating FCQNN with L2 loss and RMSProp optimiser
        self.FCQNN      = FCQNetwork(numInputs=numInputs, numActions=numActions, nodes=nodes)
        self.loss_fn    = nn.MSELoss()
        self.optimiser  = optim.RMSprop(self.FCQNN.parameters(), lr=self.ALPHA)

        # instantiating replay memory
        self.expMemory  = ReplayMemory(capacity=capacity)
        self.batchSize  = batchSize
    
    def step(self, state, action, reward, state_, terminated) -> None:
        """
        pushes the passed transition to the top of respective buffers
        :param s:  start state
        :param a:  action taken
        :param r:  reward value
        :param s_: next state
        :param t:  terminal?
        """
        self.expMemory.store(s=state, a=action, r=reward, s_=state_, t=terminated)
    
    def selectEpsilonGreedyAction(self, state) -> int:
        """
        selects and returns an epsilon-greedy action
        :param state: state
        """
        if random.random() < self.epsilon:
            action = random.randrange(start=0, stop=self.numActions)
        else:
            with torch.no_grad():
                action = self.FCQNN.predict(state).argmax().item()
        return action
    
    def selectGreedyAction(self, state) -> int:
        """
        selects and returns greediest action
        :param state: state
        """
        with torch.no_grad():
            return self.FCQNN.predict(state).argmax().item()
    
    def update(self) -> None:
        """trains and updates model over a batch of transitions"""
        # update epsilon with linear decay
        self.epsilon = max(self.epsilon - (self.EPS_MAX-self.EPS_MIN)/(self.EPS_DEC), self.EPS_MIN)

        # skip training if minimal unique batch cannot be sampled
        if len(self.expMemory) < self.batchSize:
            return
        
        # sample batch
        states, actions, rewards, states_, terminateds = self.expMemory.sampleBatch(batchSize=self.batchSize)
        
        # calculate predicted and target Q-values according to off-policy 1-step TD(0) target values
        # policyQ => [Q(s,a;θ)] with attachment to gradient chain maintained to perform gradient ascent
        policyQ = self.FCQNN.predict(states).gather(1, actions)
        # targetQ => [r + γ*max_{a'}Q(s',a';θ)] detached from the gradient chain, made a pseudo-constant estimate of true Q-value
        targetQ = rewards + self.GAMMA*self.FCQNN.predict(states_).detach().max(1)[0].unsqueeze(1)*(1-terminateds)
        
        # calculate batch loss and back-propagate
        loss = self.loss_fn(policyQ, targetQ)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

class DQN():
    """
    initialises Neural Fitted Q-value RL agent
    :param learningRate: ALPHA
    :type learningRate: float
    :param discountRate: GAMMA
    :type discountRate: float
    :param epsMax: initial epsilon
    :type epsMax: float
    :param epsMin: final epsilon
    :type epsMin: float
    :param epsDec: episodes to linearly decay epsilon
    :type epsDec: int
    :param numInputs: dimension of observation space
    :type numInputs: int
    :param numActions: cardinality of action space
    :type numActions: int
    :param nodes: sizes of hidden fully connected layers
    :type nodes: tuple
    :param sync: number of environment steps to sync policy and target networks
    :type sync: int
    :param capacity: maximum transitions to keep in replay memory
    :type capacity: int
    :param batchSize: size of batch of transitions to train FCQNN
    :type batchSize: int
    """
    def __init__(self,
                 learningRate:float, discountRate:float,
                 epsMax:float, epsMin:float, epsDec:int,
                 numInputs:int, numActions:int, nodes:tuple, sync:int,
                 capacity:int, batchSize:int) -> None:
        # hyperparameters
        self.ALPHA      = learningRate
        self.GAMMA      = discountRate
        self.EPS_MAX    = epsMax
        self.EPS_MIN    = epsMin
        self.EPS_DEC    = epsDec
        self.epsilon    = epsMax
        self.numInputs  = numInputs
        self.numActions = numActions
        self.nodes      = nodes
        self.sync       = sync
        self.steps2sync = 0

        # instantiating policy FCQNN with L2 loss and RMSProp optimiser
        self.policyFCQN = FCQNetwork(numInputs=numInputs, numActions=numActions, nodes=nodes)
        self.loss_fn    = nn.MSELoss()
        self.optimiser  = optim.RMSprop(self.policyFCQN.parameters(), lr=self.ALPHA)

        # instantiating target FCQNN as a copy of the policy FCQNN
        self.targetFCQN = FCQNetwork(numInputs=numInputs, numActions=numActions, nodes=nodes)
        self.targetFCQN.load_state_dict(self.policyFCQN.state_dict())

        # instantiating replay memory
        self.expMemory  = ReplayMemory(capacity=capacity)
        self.batchSize  = batchSize
    
    def step(self, state, action, reward, state_, terminated) -> None:
        """
        pushes transition to the top of replay memory and handles syncing of policy and target networks 
        :param s:  start state
        :param a:  action taken
        :param r:  reward value
        :param s_: next state
        :param t:  terminal?
        """
        self.expMemory.store(s=state, a=action, r=reward, s_=state_, t=terminated)
        self.steps2sync += 1
        if self.steps2sync == self.sync:
            self.steps2sync = 0
            self.targetFCQN.load_state_dict(self.policyFCQN.state_dict())
    
    def selectEpsilonGreedyAction(self, state) -> int:
        """
        selects and returns an epsilon-greedy action
        :param state: state
        """
        if random.random() < self.epsilon:
            action = random.randrange(start=0, stop=self.numActions)
        else:
            with torch.no_grad():
                action = self.policyFCQN.predict(state).argmax().item()
        return action
    
    def selectGreedyAction(self, state) -> int:
        """
        selects and returns greediest action
        :param state: state
        """
        with torch.no_grad():
            return self.policyFCQN.predict(state).argmax().item()
    
    def update(self) -> None:
        """trains and updates model over a batch of transitions"""
        # update epsilon with linear decay
        self.epsilon = max(self.epsilon - (self.EPS_MAX-self.EPS_MIN)/(self.EPS_DEC), self.EPS_MIN)

        # skip training if minimal unique batch cannot be sampled
        if len(self.expMemory) < self.batchSize:
            return
        
        # sample batch
        states, actions, rewards, states_, terminateds = self.expMemory.sampleBatch(batchSize=self.batchSize)
        
        # calculate predicted and target Q-values according to off-policy 1-step TD(0) target values
        # policyQ => [Q(s,a;θ)] with attachment to gradient chain maintained to perform gradient ascent
        policyQ = self.policyFCQN.predict(states).gather(1, actions)
        # targetQ => [r + γ*max_{a'}Q(s',a';θ)] detached from the gradient chain, made a pseudo-constant estimate of true Q-value
        targetQ = rewards + self.GAMMA*self.targetFCQN.predict(states_).detach().max(1)[0].unsqueeze(1)*(1-terminateds)
        
        # calculate batch loss and back-propagate
        loss = self.loss_fn(policyQ, targetQ)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

class DoubleDQN():
    pass

class DuellingDoubleDQN():
    pass