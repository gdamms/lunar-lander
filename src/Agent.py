import torch
import torch.nn.functional as F

from src.Critic import Critic
from src.Actor import Actor


class Agent():
    def __init__(
        self,
        lr,
        input_dim,
        output_dim,
    ):
        self.lr = lr  # Learning parameters.

        # Network parameters.
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Create the networks.
        self.actor = Actor(lr=lr, input_dim=input_dim, output_dim=output_dim)
        self.critic = Critic(lr=lr, input_dim=input_dim)

    def load_models(
        self,
        actor_checkpoint="",
        critic_checkpoint="",
    ):
        # Load the models.
        if actor_checkpoint:
            self.actor.load_checkpoint(actor_checkpoint)
        if critic_checkpoint:
            self.critic.load_checkpoint(critic_checkpoint)

    def save_models(
        self,
        actor_checkpoint="checkpoints/actor/actor.pth",
        critic_checkpoint="checkpoints/critic/critic.pth",
    ):
        # Save the models.
        self.actor.save_checkpoint(actor_checkpoint)
        self.critic.save_checkpoint(critic_checkpoint)

    def train_critic(
            self,
            states,
            cum_rewards,
    ):
        # Train the critic.
        self.critic.optimizer.zero_grad()
        states = torch.tensor(states, dtype=torch.float)
        states = states.to(self.critic.device)
        cum_rewards = torch.tensor(cum_rewards, dtype=torch.float)
        cum_rewards = cum_rewards.to(self.critic.device)
        values = self.critic(states)
        values = values.squeeze(dim=1)
        vf_loss = F.mse_loss(
            values,
            cum_rewards,
            reduction="none")
        vf_loss.sum().backward()
        self.critic.optimizer.step()

    def train_actor(
            self,
            states,
            actions,
            cum_rewards,
    ):
        # Train the actor.
        states = torch.tensor(states, dtype=torch.float)
        cum_rewards = torch.tensor(cum_rewards, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.int64)
        with torch.no_grad():
            values = self.critic(states.to(self.critic.device))
        self.actor.optimizer.zero_grad()
        actions = torch.tensor(actions, dtype=torch.int64)
        actions = actions.to(self.actor.device)
        advantages = cum_rewards - values
        logits = self.actor(states.to(self.actor.device))
        log_probs = -F.cross_entropy(logits, actions, reduction="none")
        pi_loss = -log_probs * advantages
        pi_loss.sum().backward()
        self.actor.optimizer.step()

    def choose_action(self, state):
        # Take the best action.
        state = torch.tensor(
            [state],
            dtype=torch.float)
        state = state.to(self.actor.device)
        actions = self.actor.forward(state)
        action = torch.argmax(actions).item()
        return action


if __name__ == "__main__":
    raise SystemExit("This is not a script.")
