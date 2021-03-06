import random

gamma = 0.99
eta = 0.5
max_steps = 200
num_episodes = 1000
num_files = 30
num_servers = 5
cache_size = 5
zipf_param = 1.0
env_params = dict(r_iu=1, r_ij=10, r_ci=100, #r_iu: transmission rate between user and MEC
                  file_size=[random.randrange(1, 30) for _ in range(num_files)]
                  )
queue_length = [random.randrange(0, 100) for _ in range(num_servers)]   # sum of the size of files in each queue


reward_params = dict(alpha=0.5, psi=10, mu=1, beta=0.3)