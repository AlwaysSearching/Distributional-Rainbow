import logging
import gymnasium as gym

LOG = logging.getLogger(__name__)

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()

def main():
    for _ in range(1000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()
