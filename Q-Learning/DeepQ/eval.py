import DQRL
import gymnasium as gym

testCP = trainCP = gym.make(
    "LunarLander-v2",
    continuous = False,
    gravity = -9.81,
    enable_wind = False,
    wind_power = 10.0,
    turbulence_power = 1.0,
    render_mode = 'human'
)

bestAgent = DQRL.LoadAgent('/home/pranav/Workspace/RL/src/Q-Learning/DeepQ/LLD3QN.pt')

for episode in range(5):
    done = False
    state, info = testCP.reset(seed = 42)

    while not done:
        action = bestAgent.selectAction(state=state)
        new_state, reward, terminated, truncated, info = testCP.step(action=action)
        done = terminated or truncated
        state = new_state

testCP.close()