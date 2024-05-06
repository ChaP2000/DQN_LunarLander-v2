from utils.agent import Agent
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def main():

    env = gym.make("LunarLander-v2", render_mode="human")

    episodes = 100

    current_state, info = env.reset()
    n_state = len(current_state)
    n_action = env.action_space.n

    agent = Agent(env, n_state, n_action, 50000)

    agent.load_weights()

    reward_list = []

    passed = 0

    agent.set_epsilon(0)

    for episode in range(episodes):

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

            episode_reward += reward


            if done:

                break

            current_state = new_state

        print(f"Current episode : {episode} finished with reward {episode_reward}")

        reward_list.append(episode_reward)

    env.close()

    new_list = np.array(reward_list)
    new_list = np.convolve(reward_list, np.ones(10))
    new_list = new_list[9:-9]/10
    plt.plot(list(range(episodes)), reward_list)
    new_list_episode = list(range(4, episodes - 5))
    plt.plot(new_list_episode, new_list)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend(['Reward', 'Average Reward'])
    plt.show()

    return

if __name__== "__main__":

    main()