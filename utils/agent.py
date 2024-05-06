import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from utils.QNetwork import QNet
from collections import deque
import random

batch_size = 128
init_epsilon = 0.99
Gamma = 0.99
LR = 1e-4
TAU = 0.005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(object):

    def __init__(self, env, n_states, n_actions, mem_cap):

        super().__init__()

        self.env =env
        self.n_states = n_states
        self.n_actions = n_actions
        self.q_net = QNet(n_states, n_actions)
        self.q_net.to(device)
        self.target_net = QNet(n_states, n_actions)
        self.target_net.to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr = LR, amsgrad=True)

        self.curr_mem = 0
        self.mem_length = 0
        self.memory = deque([], maxlen=mem_cap)
        self.min_mem = False
        self.step = 0
        self.epsilon = init_epsilon
        self.exp_replay_started = 0
    
    def set_epsilon(self, epsilon):

        self.epsilon = epsilon
    
    def get_epsilon(self):

        return self.epsilon


    def add_mem(self, s, a, r, s_, done):

        self.memory.append((s, a, r, s_, done))
        self.mem_length = len(self.memory)

    def choose_action(self, states):

        self.step += 1

        seed = np.random.rand(1)

        states = torch.tensor(states, dtype= torch.float32, device=device)

        if seed >= self.epsilon:

            with torch.no_grad():

              return torch.argmax(self.q_net(states)).item()

        action = self.env.action_space.sample()

        return action

    def learn(self):

        if self.mem_length < batch_size:

            return

        else:

            if self.exp_replay_started == 0:

                print("Experience replay started")

                self.exp_replay_started = 1

            batch = random.sample(self.memory, batch_size)

            #perform training using state batches

            batch = list(zip(*batch))

            states = torch.cat(batch[0], dim = 0)

            actions = torch.cat(batch[1], dim = 0)

            rewards = torch.cat(batch[2], dim = 0)

            next_states = torch.cat([s for s in batch[3] if s is not None], dim = 0)

            state_action_values = self.q_net(states).gather(1, actions.unsqueeze(1))

            mask = torch.tensor(tuple(map(lambda s: s is not None, batch[3])))

            next_state_action_values = torch.zeros(batch_size, dtype= torch.float32, device=device)

            with torch.no_grad():

                next_state_action_values[mask] = self.target_net(next_states).max(1).values

            expected_state_action_values = next_state_action_values*Gamma + rewards

            criterion = nn.MSELoss()

            loss = criterion(state_action_values.float(), expected_state_action_values.unsqueeze(1).float())

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 1) #maybe use a lower gradient, e.g. 0.5 (100 used for cartpole)
            self.optimizer.step()


            q_net_dict = self.q_net.state_dict()
            target_net_dict = self.target_net.state_dict()

            for key in q_net_dict:
                target_net_dict[key] = q_net_dict[key]*TAU + target_net_dict[key]*(1-TAU)
            self.target_net.load_state_dict(target_net_dict)
            return
    
    def load_weights(self):
        print("Weights loaded")
        self.q_net.load_state_dict(torch.load("saved_weights/Q_net.pth"))
        self.target_net.load_state_dict(torch.load("saved_weights/target_net.pth"))
    
    def save_weights(self):
        torch.save(self.q_net.state_dict(), "saved_weights/Q_net.pth")
        torch.save(self.target_net.state_dict(), "saved_weights/target_net.pth")