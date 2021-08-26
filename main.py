from dataLoader import DataLoaderZipf, MakePopularity
from environment import Environment
from config import *
from agent import *
import numpy as np

## main code

def run(env):
  steps = []
  all_costs = []
  for episode in range(num_episodes):
    observation = env.reset() #환경 초기화
    print("Episode: {} / {}".format(episode+1, num_episodes))
    step = 0
    cost = 0

    while True:
      # print("episode:", episode, "step:", step)
      s = env.states.index(observation['state'])
      # print("State: ", observation['state'])

      action = agent._act(s)
      # print("action =", action)
      observation_next, reward, done = env.step(action)

      s_next = env.states.index(observation_next['state'])
      cost += agent._update_q_value(s, action, reward, s_next, eta)

      # Swap observation
      observation = observation_next

      step += 1

      if done:
        steps += [step]
        all_costs += [cost]
        break

  env.display()
  agent.plot_results(steps, all_costs)

if __name__ == "__main__":

    # make popularity
    p_maker = MakePopularity(num_files, zipf_param)
    popularity = p_maker.get_popularity(num_servers)
    env = Environment(popularity, cache_size, num_servers, num_files, reward_params, env_params, queue_length)

    Q = np.zeros([env.n_states, env.n_actions])
    agent = Agent(env, gamma, Q)

    # for i in range(num_servers):
    #     agent.set_cooperNet(MECServer(i, cache_size))

    # print(env.add_servers(agent.get_cooperNet()))

    run(env)