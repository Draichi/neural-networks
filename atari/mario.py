# sudo apt-get install fceux

mport gym_super_mario_bros
env = gym_super_mario_bros.make('SuperMarioBros-v0')

observation = env.reset()
done = False
t = 0
while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    t += 1
    if not t % 100:
        print(t, info)