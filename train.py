from utils.agent import Agent
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def main():

    env = gym.make("LunarLander-v2", render_mode="human")

    episodes = 800
    epsilon_decay = 0.996

    current_state, info = env.reset()
    n_state = len(current_state)
    n_action = env.action_space.n

    agent = Agent(env, n_state, n_action, 500000)

    reward_list = []

    passed = 0

    for e in range(episodes):

        current_state, info = env.reset()
        done = 0

        episode_reward = 0

        for count in range(1500):

            action = agent.choose_action(current_state)

            new_state, reward, terminated, truncated, info = env.step(action)

            passed += 1

            if terminated or truncated:

                next_state = None

                done = 1
            else:

                next_state = torch.tensor([new_state], dtype= torch.float32, device='cuda')

            agent.add_mem(torch.tensor([current_state], dtype= torch.float32, device='cuda'), torch.tensor([action], dtype= torch.long, device='cuda'), 
                          torch.tensor([reward], dtype= torch.float32, device='cuda'), next_state, torch.tensor([done], dtype= torch.float32, device='cuda'))

            episode_reward += reward

            agent.learn()

            current_state = new_state

            if done:

                break

        print(f"Current episode : {e} finished with reward {episode_reward} and memory length is {passed}")

        reward_list.append(episode_reward)

        #agent.epsilon *= epsilon_decay
        agent.set_epsilon(agent.get_epsilon()*epsilon_decay)

    env.close()
    agent.save_weights()

    new_list = np.array(reward_list)
    new_list = np.convolve(reward_list, np.ones(50))
    new_list = new_list[49:-49]/50
    plt.plot(list(range(episodes)), reward_list)
    new_list_episode = list(range(24, episodes - 25))
    plt.plot(new_list_episode, new_list)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend(['Reward', 'Average Reward'])
    plt.show()

    return

if __name__== "__main__":

    main()