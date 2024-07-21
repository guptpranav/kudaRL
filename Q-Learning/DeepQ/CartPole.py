import DQRL
import gymnasium as gym

agent = DQRL.LoadAgent(modelPATH="./CPDDQN.pt")
testCP = gym.make('CartPole-v1', render_mode = 'human')

N = 5
for episode in range(N):
    done = False
    state, info = testCP.reset(seed = 42)

    while not done:
        action = agent.selectGreedyAction(state=state)
        new_state, reward, terminated, truncated, info = testCP.step(action=action)
        done = terminated or truncated
        state = new_state

testCP.close()