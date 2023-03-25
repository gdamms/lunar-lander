import gym

from src.Agent import Agent


def evaluate(
    actor_checkpoint,
):

    # Set everything up.
    env = gym.make("LunarLander-v2", render_mode="human")
    agent = Agent(
        lr=0.0,
        input_dim=8,
        output_dim=4,
    )
    agent.load_models(
        actor_checkpoint=actor_checkpoint,
    )

    while True:
        # Run an episode.
        running = True
        state, info = env.reset()
        score = 0
        while running:
            action = agent.choose_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            score += reward
            running = not terminated and not truncated
        print(f"Score: \33[1;34m{score:.01f}\33[0m")
    env.close()


if __name__ == "__main__":
    raise SystemExit("This is not a script.")
