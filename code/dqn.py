import gym
import math
import random
random.seed(498)
import numpy as np
np.random.seed(498)
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
torch.manual_seed(498)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, hidden_size, num_inputs, action_space):
        super(DQN, self).__init__()
    # TODO: design the architecture, fc-relu-fc might be good enough
        self.action_space = action_space
        num_outputs = action_space.n
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
    # Called with either one element to determine next action, or a batch
    # during optimization.
    # Given batch of states, predict the Q value for each action
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        return output

def select_action(state, policy_net, eps_end, eps_start, eps_decay, steps_done, device):
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * \
        math.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    #TODO: epsilon-greedy action selection
    #with probability eps_threshold, take random action
    #with probability 1-eps_threshold, take the greedy action
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1), steps_done
    else:
        n_actions = policy_net.action_space.n
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), steps_done
    



def optimize_model(policy_net, target_net, optimizer, memory, batch_size, gamma, device):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.

    Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # TODO: Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net\
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # TODO: Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.

    # TODO: Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # TODO: Compute Huber loss. In practice, Huber loss might be better than L2 loss.
    # smooth_l1_loss in pytorch might be useful
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def plot_durations(episode_durations, save_path):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.savefig(save_path)
    plt.show()
    plt.close()



def main():
    env = gym.make('CartPole-v0').unwrapped
    env.seed(498)

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.5
    EPS_END = 0.02
    EPS_DECAY = 200
    TARGET_UPDATE = 10
    
    hidden_size = 128
    num_inputs = env.observation_space.shape[0]
    action_space = env.action_space
    policy_net = DQN(hidden_size, num_inputs, action_space).to(device)
    target_net = DQN(hidden_size, num_inputs, action_space).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=0.002)
    memory = ReplayMemory(10000)

    steps_done = 0
    episode_durations = []

    num_episodes = 400
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        state = env.reset()
        state = torch.tensor([state], device=device, dtype=torch.float32)
        for t in count():
            # Select and perform an action
            action, steps_done = select_action(state, policy_net, eps_end=EPS_END, eps_start=EPS_START, eps_decay=EPS_DECAY, steps_done=steps_done, device=device)            
            next_state, reward, done, _ = env.step(action.item())
            next_state = torch.tensor([next_state], device=device, dtype=torch.float32)
            reward = torch.tensor([reward], device=device)
            if done:
                next_state=None
    
            # TODO: Store the transition in memory
            memory.push(state, action, next_state, reward)
        
            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model(policy_net=policy_net, target_net=target_net, optimizer=optimizer, memory=memory, batch_size=BATCH_SIZE, gamma=GAMMA, device=device)
            if done:
                episode_durations.append(t + 1)
                if i_episode %10==0:
                    print('episode', i_episode, 'duration', episode_durations[-1])
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')
    env.render()
    env.close()
    plt.ioff()
    plt.show()
    #plot_durations(episode_durations, 'dqn_reward.png')
    return episode_durations, target_net, policy_net

if __name__ == '__main__':
    episode_durations, target_net, policy_net = main()
