import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        ##### TODO ######
        ### Complete definition 
        self.l1 = nn.Linear(4, 128, bias=False)
        self.l2 = nn.Linear(128, 2, bias=False)
        self.l3 = nn.Linear(128, 1, bias=False)
        
        self.eposide_reward = []

    def forward(self, x):
        ##### TODO ######
        ### Complete definition 
        x = F.relu(self.l1(x))
        actor = self.l2(x)
        actor = F.softmax(actor)
        critic = self.l3(x)
        return actor, critic

model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()

    log_prob = m.log_prob(action)
    return action, log_prob, state_value

def sample_episode():

    state, ep_reward = env.reset(), 0
    episode = []

    for t in range(1, 10000):  # Run for a max of 10k steps

        action, log_prob, state_value = select_action(state)

        # Perform action
        next_state, reward, done, _ = env.step(action.item())

        episode.append((state_value, log_prob, reward))
        state = next_state

        ep_reward += reward

        if args.render:
            env.render()

        if done:
            break

    return episode, ep_reward

def compute_losses(episode):

    ####### TODO #######
    #### Compute the actor and critic losses
    actor_loss, critic_loss = 0, 0
    
    R = 0
    rewards = []
    for i in reversed(range(len(episode))):
        reward = episode[i][-1]
        R = reward + args.gamma * R #+ args.gamma**len(episode)*model.state_value[i]
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    
    for i in range(len(rewards)):    
        advantage = rewards[i] - episode[i][0]
        actor_loss -= episode[i][1]*advantage
        critic_loss += advantage**2
    
    #actor_loss = 
    #critic_loss = 

    return actor_loss/len(rewards), critic_loss/len(rewards)

ave_reward = []
def main():
    running_reward = 10
    
    for i_episode in count(1):

        episode, episode_reward = sample_episode()
        
        model.eposide_reward.append(episode_reward)

        optimizer.zero_grad()

        actor_loss, critic_loss = compute_losses(episode)

        loss = actor_loss + critic_loss
        
        loss.backward()

        optimizer.step()

        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        ave_reward.append(running_reward)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, episode_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, len(episode)))
            break
        
        if i_episode>=1000:
            break


if __name__ == '__main__':
    main()
    plt.plot(model.eposide_reward)
    plt.plot(ave_reward)
    plt.xlabel('episode')
    plt.ylabel('reward')
