import gym
import numpy as np
from rich.progress import track

from src.Agent import Agent


def train(
    episode_per_epoch,
    gamma,
    lr,
    actor_checkpoint,
    critic_checkpoint,
):
    # Set everything up.
    env = gym.make("LunarLander-v2")
    agent = Agent(
        lr=lr,
        input_dim=8,
        output_dim=2,
    )
    agent.load_models(
        actor_checkpoint=actor_checkpoint,
        critic_checkpoint=critic_checkpoint,
    )

    running = True
    epoch = 0
    best_average_score = None
    checkpoint = 0
    while running:

        # Run an epoch.
        for episode in track(range(episode_per_epoch), description=f"Epoch {epoch:3}"):
            reward_records = []

            # Run an episode.
            done = False
            states = []
            actions = []
            rewards = []
            state, info = env.reset()
            while not done:
                states.append(state)
                action = agent.choose_action(state)
                actions.append(action)
                state, reward, terminate, truncated, info = env.step(
                    action)
                rewards.append(reward)
                done = terminate or truncated

            # Compute cumulative rewards.
            cum_rewards = np.zeros_like(rewards)
            reward_len = len(rewards)
            for i in reversed(range(reward_len)):
                cum_rewards[i] = rewards[i] + \
                    (cum_rewards[i+1]*gamma if i+1 < reward_len else 0)

            # Train the networks.
            agent.train_critic(states, cum_rewards)
            agent.train_actor(states, actions, cum_rewards)

            # Record the rewards.
            final_reward = sum(rewards)
            reward_records.append(final_reward)

        # Finalize the epoch.
        average_reward = np.average(reward_records)
        print(f"Score \33[1;34m{average_reward:.01f}\33[0m.")
        saved = best_average_score is None or average_reward > best_average_score
        if saved:
            agent.save_models(
                actor_checkpoint=f"checkpoints/actor/actor{checkpoint}.pth",
                critic_checkpoint=f"checkpoints/critic/critic{checkpoint}.pth",
            )
            best_average_score = average_reward
            checkpoint += 1
            print(f"Checkpoint \33[1;34m{checkpoint}\33[0m saved.")

        epoch += 1

    env.close()


if __name__ == "__main__":
    raise SystemExit("This is not a script.")
