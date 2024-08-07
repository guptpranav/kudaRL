import torch
from torch import nn, optim

import random
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

# TODO: implement Prioritized Experience Replay (paper/grokking)

class FCQNet(nn.Module):
    """
    initialises a sequential Fully Connected Q Network with the state [s] as input and all action-values [Q(s,a)] as output
    :param numInputs: dimension of observation space
    :type numInputs: int
    :param numActions: cardinality of action space
    :type numActions: int
    :param nodes: sizes of hidden fully connected layers
    :type nodes: tuple
    """
    def __init__(self, numInputs:int, numActions:int, nodes:tuple) -> None:
        super(FCQNet, self).__init__()
        
        # creating fully connected neural network with the given layer sizes
        self.model = nn.Sequential()
        self.model.append(nn.Linear(numInputs, nodes[0]))
        self.model.append(nn.ReLU())
        for k in range(len(nodes)-1):
            self.model.append(nn.Linear(nodes[k], nodes[k+1]))
            self.model.append(nn.ReLU())
        self.model.append(nn.Linear(nodes[-1], numActions))
        self.model.to(device=DEVICE)
    
    def forward(self, state) -> torch.Tensor:
        """
        performs forward pass through FCQNN and returns all state-action values 
        :param state: state
        """
        state = torch.tensor(state, device=DEVICE, dtype=torch.float32)
        return self.model.forward(state)

class FCDuelingQNet(nn.Module):
    """
    initialises a sequential Fully Connected Dueling Q Network with state [s] as input and action-values [Q(s,a)] as output\n
    penultimate dueling layer has a node for state-value [V(s)] and the rest for action-advantages [A(s,a)]
    :param numInputs: dimension of observation space
    :type numInputs: int
    :param numActions: cardinality of action space
    :type numActions: int
    :param nodes: sizes of hidden fully connected layers
    :type nodes: tuple
    """
    def __init__(self, numInputs:int, numActions:int, nodes:tuple) -> None:
        super(FCDuelingQNet, self).__init__()
        
        # creating fully connected neural network with the given layer sizes
        self.model = nn.Sequential()
        self.model.append(nn.Linear(numInputs, nodes[0]))
        self.model.append(nn.ReLU())
        for k in range(len(nodes)-1):
            self.model.append(nn.Linear(nodes[k], nodes[k+1]))
            self.model.append(nn.ReLU())
        self.model.to(device=DEVICE)
        self.V = nn.Linear(nodes[-1], 1).to(DEVICE)
        self.A = nn.Linear(nodes[-1], numActions).to(DEVICE)
    
    def forward(self, state) -> torch.Tensor:
        """
        performs forward pass through FCDuelingQNN and returns all state-action values 
        :param state: state
        """
        state = torch.tensor(state, device=DEVICE, dtype=torch.float32)
        x = self.model.forward(state)
        a = self.A.forward(x)
        v = self.V.forward(x).expand_as(a)
        q = v + a - a.mean(-1, keepdim=True).expand_as(a)
        return q

class LoadAgent():
    """
    load trained agent model from file
    :param modelPATH: path to model file
    :type modelPATH: string
    """
    def __init__(self, PATH) -> None:
        self.model = torch.load(PATH)
    
    def selectAction(self, state) -> int:
        """
        selects and returns greediest action
        :param state: state
        """
        with torch.no_grad():
            return self.model.forward(state).argmax().item()

class DeepQAgent():
    """
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
    :param tau: fraction of policy network to supercompose onto target network
    :type tau: float
    :param sync: number of environment steps to sync policy and target networks
    :type sync: int
    :param capacity: maximum transitions to keep in replay memory
    :type capacity: int
    :param batchSize: size of batch of transitions to train FCQNN
    :type batchSize: int
    :param updates: number of update cycles per episode completed
    :type updates: int
    """
    def __init__(self,
                 learningRate:float, discountRate:float,
                 epsMax:float, epsMin:float, epsDec:int,
                 numInputs:int, numActions:int, nodes:tuple, tau:float, sync:int,
                 capacity:int, batchSize:int, updates:int) -> None:
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
        self.tau        = tau
        self.steps2sync = 0

        # instantiating replay memory
        self.expMemory  = ReplayMemory(capacity=capacity)
        self.batchSize  = batchSize
        self.updates    = updates
    
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
        for target, policy in zip(self.targetNet.parameters(), self.policyNet.parameters()):
            polyakAvg = (1-self.tau)*target.data + self.tau*policy.data
            target.data.copy_(polyakAvg)
        if self.steps2sync == self.sync:
            self.steps2sync = 0
            self.targetNet.load_state_dict(self.policyNet.state_dict())

    def selectEpsilonGreedyAction(self, state) -> int:
        """
        selects and returns an action epsilon-greedy to action-value
        :param state: state
        """
        if random.random() < self.epsilon:
            action = random.randrange(start=0, stop=self.numActions)
        else:
            with torch.no_grad():
                action = self.policyNet.forward(state).argmax().item()
        return action

    def selectGreedyAction(self, state) -> int:
        """
        selects and returns action greediest to action-value
        :param state: state
        """
        with torch.no_grad():
            return self.policyNet.forward(state).argmax().item()
    
    def save(self, PATH) -> None:
        """
        saves model parameters to PATH
        :param PATH: path to save file
        :type PATH: str
        """
        torch.save(self.policyNet, PATH)

class NFQ(DeepQAgent):
    """
    initialises Neural Fitted Q-value RL agent: action selection and model evaluation done using the same FCQNN
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
    :param updates: number of update cycles per episode completed
    :type updates: int
    """
    def __init__(self,
                 learningRate:float, discountRate:float,
                 epsMax:float, epsMin:float, epsDec:int,
                 numInputs:int, numActions:int, nodes:tuple,
                 capacity:int, batchSize:int, updates:int) -> None:
        super().__init__(learningRate=learningRate, discountRate=discountRate,
                         epsMax=epsMax, epsMin=epsMin, epsDec=epsDec,
                         numInputs=numInputs, numActions=numActions, nodes=nodes, tau=0, sync=0,
                         capacity=capacity, batchSize=batchSize, updates=updates)

        # instantiating FCQNN with L2 loss and RMSProp optimiser
        self.policyNet = FCQNet(numInputs=numInputs, numActions=numActions, nodes=nodes)
        self.loss_fn   = nn.MSELoss()
        self.optimiser = optim.RMSprop(self.policyNet.parameters(), lr=self.ALPHA)
    
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

    def update(self) -> None:
        """
        trains and updates model over a batch of transitions
        target Q values bootstrapped to 1-step TD(0) update
        """
        # update epsilon with linear decay
        self.epsilon = max(self.epsilon - (self.EPS_MAX-self.EPS_MIN)/(self.EPS_DEC), self.EPS_MIN)

        # skip training if minimal unique batch cannot be sampled
        if len(self.expMemory) < self.batchSize:
            return
        
        for _ in range(self.updates):
            # sample batch
            states, actions, rewards, states_, terminateds = self.expMemory.sampleBatch(batchSize=self.batchSize)
            
            # policyQ => [Q(s,a;θ)]
            # results are kept attached to gradient chain to perform gradient ascent
            policyQ = self.policyNet.forward(states).gather(1, actions)
            # targetQ => [r + γ*max_{a'}Q(s',a';θ)]
            # results are detached from the gradient chain and treated like a pseudo-constant estimate of true Q-value
            targetQ = rewards + self.GAMMA*self.policyNet.forward(states_).detach().max(1)[0].unsqueeze(1)*(1-terminateds)
            
            # calculate batch loss and back-propagate
            loss = self.loss_fn(policyQ, targetQ)
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

class DQN(DeepQAgent):
    """
    initialises Deep Q Network RL agent: action selection and model evaluation performed with seperate FCQNNs, synced periodically\n
    learning stabilised by temporary freezing of target Q network
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
    :param tau: fraction of policy network to supercompose onto target network
    :type tau: float
    :param sync: number of environment steps to sync policy and target networks
    :type sync: int
    :param capacity: maximum transitions to keep in replay memory
    :type capacity: int
    :param batchSize: size of batch of transitions to train FCQNN
    :type batchSize: int
    :param updates: number of update cycles per episode completed
    :type updates: int
    """
    def __init__(self,
                 learningRate:float, discountRate:float,
                 epsMax:float, epsMin:float, epsDec:int,
                 numInputs:int, numActions:int, nodes:tuple, tau:int, sync:int,
                 capacity:int, batchSize:int, updates:int) -> None:
        super().__init__(learningRate=learningRate, discountRate=discountRate,
                         epsMax=epsMax, epsMin=epsMin, epsDec=epsDec,
                         numInputs=numInputs, numActions=numActions, nodes=nodes, tau=tau, sync=sync,
                         capacity=capacity, batchSize=batchSize, updates=updates)
        
        # instantiating policy FCQNN with L2 loss and RMSProp optimiser
        self.policyNet = FCQNet(numInputs=numInputs, numActions=numActions, nodes=nodes)
        self.loss_fn   = nn.MSELoss()
        self.optimiser = optim.RMSprop(self.policyNet.parameters(), lr=self.ALPHA)

        # instantiating target FCQNN as a copy of the policy FCQNN
        self.targetNet = FCQNet(numInputs=numInputs, numActions=numActions, nodes=nodes)
        self.targetNet.load_state_dict(self.policyNet.state_dict())
    
    def update(self) -> None:
        """
        trains and updates model over a batch of transitions
        target Q values bootstrapped to 1-step TD(0) update
        """
        # update epsilon with linear decay
        self.epsilon = max(self.epsilon - (self.EPS_MAX-self.EPS_MIN)/(self.EPS_DEC), self.EPS_MIN)

        # skip training if minimal unique batch cannot be sampled
        if len(self.expMemory) < self.batchSize:
            return
        
        # sample batch
        states, actions, rewards, states_, terminateds = self.expMemory.sampleBatch(batchSize=self.batchSize)
        
        # policyQ => [Q(s,a;θ)]
        policyQ = self.policyNet.forward(states).gather(1, actions)
        # targetQ => [r + γ*max_{a'}Q(s',a';θ')]
        targetQ = rewards + self.GAMMA*self.targetNet.forward(states_).detach().max(1)[0].unsqueeze(1)*(1-terminateds)
        
        # calculate batch loss and back-propagate
        loss = self.loss_fn(policyQ, targetQ)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

class DuelingDQN(DeepQAgent):
    """
    initialises Deep Q Network RL agent: action selection and model evaluation performed with seperate FCQNNs, synced periodically\n
    learning stabilised by temporary freezing of target Q network
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
    :param tau: fraction of policy network to supercompose onto target network
    :type tau: float
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
                 numInputs:int, numActions:int, nodes:tuple, tau:int, sync:int,
                 capacity:int, batchSize:int, updates:int) -> None:
        super().__init__(learningRate=learningRate, discountRate=discountRate,
                         epsMax=epsMax, epsMin=epsMin, epsDec=epsDec,
                         numInputs=numInputs, numActions=numActions, nodes=nodes, tau=tau, sync=sync,
                         capacity=capacity, batchSize=batchSize, updates=updates)
        
        # instantiating policy FCQNN with L2 loss and RMSProp optimiser
        self.policyNet = FCDuelingQNet(numInputs=numInputs, numActions=numActions, nodes=nodes)
        self.loss_fn   = nn.MSELoss()
        self.optimiser = optim.RMSprop(self.policyNet.parameters(), lr=self.ALPHA)

        # instantiating target FCQNN as a copy of the policy FCQNN
        self.targetNet = FCDuelingQNet(numInputs=numInputs, numActions=numActions, nodes=nodes)
        self.targetNet.load_state_dict(self.policyNet.state_dict())
    
    def update(self) -> None:
        """
        trains and updates model over a batch of transitions
        target Q values bootstrapped to 1-step TD(0) update
        """
        # update epsilon with linear decay
        self.epsilon = max(self.epsilon - (self.EPS_MAX-self.EPS_MIN)/(self.EPS_DEC), self.EPS_MIN)

        # skip training if minimal unique batch cannot be sampled
        if len(self.expMemory) < self.batchSize:
            return
        
        # sample batch
        states, actions, rewards, states_, terminateds = self.expMemory.sampleBatch(batchSize=self.batchSize)
        
        # policyQ => [Q(s,a;θ)]
        policyQ = self.policyNet.forward(states).gather(1, actions)
        # targetQ => [r + γ*max_{a'}Q(s',a';θ')]
        targetQ = rewards + self.GAMMA*self.targetNet.forward(states_).detach().max(1)[0].unsqueeze(1)*(1-terminateds)
        
        # calculate batch loss and back-propagate
        loss = self.loss_fn(policyQ, targetQ)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

class DoubleDQN(DeepQAgent):
    """
    initialises Deep Q Network RL agent: action selection and model evaluation performed with seperate FCQNNs, synced periodically\n
    stabler learning: temporary freezing of target Q network\n
    lower maximisation bias: while boostrapping, select best next actions using policy network and evaluate them with target network 
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
    :param tau: fraction of policy network to supercompose onto target network
    :type tau: float
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
                 numInputs:int, numActions:int, nodes:tuple, tau:int, sync:int,
                 capacity:int, batchSize:int, updates:int) -> None:
        super().__init__(learningRate=learningRate, discountRate=discountRate,
                         epsMax=epsMax, epsMin=epsMin, epsDec=epsDec,
                         numInputs=numInputs, numActions=numActions, nodes=nodes, tau=tau, sync=sync,
                         capacity=capacity, batchSize=batchSize, updates=updates)
        
        # instantiating policy FCQNN with L2 loss and RMSProp optimiser
        self.policyNet = FCQNet(numInputs=numInputs, numActions=numActions, nodes=nodes)
        self.loss_fn   = nn.MSELoss()
        self.optimiser = optim.RMSprop(self.policyNet.parameters(), lr=self.ALPHA)

        # instantiating target FCQNN as a copy of the policy FCQNN
        self.targetNet = FCQNet(numInputs=numInputs, numActions=numActions, nodes=nodes)
        self.targetNet.load_state_dict(self.policyNet.state_dict())

    def update(self) -> None:
        """
        trains and updates model over a batch of transitions
        target Q values bootstrapped to 1-step TD(0) update
        """
        # update epsilon with linear decay
        self.epsilon = max(self.epsilon - (self.EPS_MAX-self.EPS_MIN)/(self.EPS_DEC), self.EPS_MIN)

        # skip training if minimal unique batch cannot be sampled
        if len(self.expMemory) < self.batchSize:
            return
        
        # sample batch
        states, actions, rewards, states_, terminateds = self.expMemory.sampleBatch(batchSize=self.batchSize)
        
        # policyQ => [Q(s,a;θ)]
        policyQ = self.policyNet.forward(states).gather(1, actions)
        # targetQ => [r + γ*Q(s',argmax_{a'}Q(s',a';θ);θ')]
        actions_ = self.policyNet.forward(states_).argmax(1).unsqueeze(1) # argmax_{a'}Q(s',a';θ)
        targetQ = rewards + self.GAMMA*self.targetNet.forward(states_).detach().gather(1, actions_)*(1-terminateds)
        
        # calculate batch loss and back-propagate
        loss = self.loss_fn(policyQ, targetQ)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

class DuelingDoubleDQN(DeepQAgent):
    """
    initialises Deep Q Network RL agent: action selection and model evaluation performed with seperate FCQNNs, synced periodically\n
    stabler learning: temporary freezing of target Q network\n
    lower maximisation bias: while boostrapping, select best next actions using policy network and evaluate them with target network 
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
    :param tau: fraction of policy network to supercompose onto target network
    :type tau: float
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
                 numInputs:int, numActions:int, nodes:tuple, tau:int, sync:int,
                 capacity:int, batchSize:int, updates:int) -> None:
        super().__init__(learningRate=learningRate, discountRate=discountRate,
                         epsMax=epsMax, epsMin=epsMin, epsDec=epsDec,
                         numInputs=numInputs, numActions=numActions, nodes=nodes, tau=tau, sync=sync,
                         capacity=capacity, batchSize=batchSize, updates=updates)
        
        # instantiating policy FCQNN with L2 loss and RMSProp optimiser
        self.policyNet = FCDuelingQNet(numInputs=numInputs, numActions=numActions, nodes=nodes)
        self.loss_fn   = nn.MSELoss()
        self.optimiser = optim.RMSprop(self.policyNet.parameters(), lr=self.ALPHA)

        # instantiating target FCQNN as a copy of the policy FCQNN
        self.targetNet = FCDuelingQNet(numInputs=numInputs, numActions=numActions, nodes=nodes)
        self.targetNet.load_state_dict(self.policyNet.state_dict())
    
    def update(self) -> None:
        """
        trains and updates model over a batch of transitions
        target Q values bootstrapped to 1-step TD(0) update
        """
        # update epsilon with linear decay
        self.epsilon = max(self.epsilon - (self.EPS_MAX-self.EPS_MIN)/(self.EPS_DEC), self.EPS_MIN)

        # skip training if minimal unique batch cannot be sampled
        if len(self.expMemory) < self.batchSize:
            return
        
        # sample batch
        states, actions, rewards, states_, terminateds = self.expMemory.sampleBatch(batchSize=self.batchSize)
        
        # policyQ => [Q(s,a;θ)]
        policyQ = self.policyNet.forward(states).gather(1, actions)
        # targetQ => [r + γ*Q(s',argmax_{a'}Q(s',a';θ);θ')]
        actions_ = self.policyNet.forward(states_).argmax(1).unsqueeze(1) # argmax_{a'}Q(s',a';θ)
        targetQ = rewards + self.GAMMA*self.targetNet.forward(states_).detach().gather(1, actions_)*(1-terminateds)
        
        # calculate batch loss and back-propagate
        loss = self.loss_fn(policyQ, targetQ)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()