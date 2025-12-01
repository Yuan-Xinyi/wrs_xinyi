import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical 
import numpy as np

# --- 1. Task-Specific Placeholders (Environment Interaction) ---

def task_initialize_environment():
    """
    Placeholder: Initialize the environment and return state/action dimensions.
    Returns: state_dim (int), action_dim (int)
    """
    print("Placeholder: Initializing environment, returning state/action dims...")
    # TODO: Replace with actual environment initialization and dimension extraction
    return 4, 2 # Example values for a discrete action space problem (e.g., CartPole)

def task_run_one_step(env, action):
    """
    Placeholder: Execute one step in the environment.
    env: Environment object (TODO: Actual environment instance)
    action: Action taken (int or np.array)
    Returns: next_state (np.array), reward (float), done (bool), info (dict)
    """
    # TODO: Replace with env.step(action)
    return np.zeros(4), 0.0, False, None 

def task_reset_environment(env):
    """
    Placeholder: Reset the environment and return the initial state.
    env: Environment object (TODO: Actual environment instance)
    Returns: initial_state (np.array)
    """
    # TODO: Replace with env.reset()
    return np.zeros(4) 

# --- 2. PPO Core Model Components (Actor-Critic Networks) ---

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Policy Network (Actor): Input -> Action probability distribution
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1) # For discrete actions
        )
        
        # Value Network (Critic): Input -> State Value V(s)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Outputs a single scalar value estimate
        )

    def forward(self, state):
        """Simultaneous forward pass for policy and value"""
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

# --- 3. PPO Agent Class ---

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 ppo_epochs=10, mini_batch_size=64, clip_epsilon=0.2, 
                 gae_lambda=0.95, vf_coeff=0.5, ent_coeff=0.01):
        
        # PPO Hyperparameters
        self.gamma = gamma                # Discount factor
        self.ppo_epochs = ppo_epochs      # Optimization epochs on collected data
        self.mini_batch_size = mini_batch_size # Mini-batch size
        self.clip_epsilon = clip_epsilon  # Clipping parameter epsilon
        self.gae_lambda = gae_lambda      # GAE parameter lambda
        self.vf_coeff = vf_coeff          # Value function loss coefficient (c1)
        self.ent_coeff = ent_coeff        # Entropy loss coefficient (c2)

        # Actor-Critic Networks
        self.policy = ActorCritic(state_dim, action_dim)
        # Old Policy Network (used to calculate probability ratio r_t)
        self.policy_old = ActorCritic(state_dim, action_dim) 
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Optimizer (Optimizes both Actor and Critic parameters)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Loss function for Value network
        self.MseLoss = nn.MSELoss()
        
        # TODO: Device management (CPU/GPU)

    def select_action(self, state):
        """
        Selects an action from the policy_old network for data collection.
        state: np.array of shape [state_dim]
        Returns: action (int), action_log_prob (float), value (float)
        """
        # Convert state (numpy array) to tensor, add batch dimension
        state = torch.from_numpy(state).float().unsqueeze(0)
        
        # Get action probabilities and state value from the old policy
        with torch.no_grad():
            action_probs, value = self.policy_old(state)
        
        # Sample action from the distribution
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # Calculate log probability of the sampled action
        action_log_prob = dist.log_prob(action)
        
        return action.item(), action_log_prob.item(), value.item()

    def calculate_gae_and_returns(self, rewards, values, dones, next_value):
        """
        Core Function: Calculates GAE Advantages and Monte Carlo Returns.
        values list must include the next_value at the end.
        """
        returns = []
        advantages = []
        
        # Initialize GAE advantage accumulator
        last_gae = 0.0
        
        # The 'values' list should be of size len(rewards) + 1
        # Iterate backwards through the experience buffer
        for i in reversed(range(len(rewards))):
            # V(s_{t+1}) is the value estimate of the next state
            next_value_t = values[i+1] 
            # Multiplier for non-terminal state (0 if done, 1 otherwise)
            not_done = 1.0 - dones[i]

            # TD Error (Delta)
            delta = rewards[i] + self.gamma * next_value_t * not_done - values[i]
            
            # GAE Advantage A_t = delta_t + gamma * lambda * A_{t+1}
            last_gae = delta + self.gamma * self.gae_lambda * not_done * last_gae
            advantages.insert(0, last_gae)
            
            # Monte Carlo Return (Target value for Critic) R_t = A_t + V(s_t)
            returns.insert(0, last_gae + values[i])

        # Convert to Tensor
        advantages = torch.tensor(advantages, dtype=torch.float)
        returns = torch.tensor(returns, dtype=torch.float)
        
        # Advantage normalization (highly recommended for stable training)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns

    def update(self, states, actions, old_log_probs, returns, advantages):
        """
        Core Function: PPO Policy and Value Network update using Mini-Batches.
        """
        
        # Convert all data to Tensor format
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float)
        
        data_size = states.size(0)
        
        # PPO Optimization Loop (multiple epochs over the same data)
        for _ in range(self.ppo_epochs):
            # Randomly shuffle indices for mini-batching
            batch_indices = torch.randperm(data_size)
            
            # Iterate through Mini-Batches
            for start in range(0, data_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                minibatch_indices = batch_indices[start:end]
                
                # Extract Mini-Batch Data
                mb_states = states[minibatch_indices]
                mb_actions = actions[minibatch_indices]
                mb_old_log_probs = old_log_probs[minibatch_indices]
                mb_returns = returns[minibatch_indices]
                mb_advantages = advantages[minibatch_indices]

                # 1. Forward Pass: Get current policy's action probabilities and state values
                action_probs, state_values = self.policy(mb_states)
                
                # Get log probabilities for the taken actions
                dist = Categorical(action_probs)
                current_log_probs = dist.log_prob(mb_actions)
                
                # 2. Policy Loss (PPO Clipped Surrogate Loss)
                
                # Probability ratio: r_t(theta) = pi_new / pi_old
                ratios = torch.exp(current_log_probs - mb_old_log_probs)
                
                # Clipped loss term
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 
                                    1 - self.clip_epsilon, 
                                    1 + self.clip_epsilon) * mb_advantages
                
                # PPO Policy Loss: Negative sign because we maximize the objective
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 3. Value Loss (Value Function Loss)
                
                # Value network output is a column vector, squeeze to 1D
                state_values = state_values.squeeze(-1) 
                
                # Mean Squared Error Loss: (V(s) - R_t)^2
                value_loss = self.MseLoss(state_values, mb_returns)
                
                # TODO: Optional: Implement clipped value function loss (PPO2 style)
                
                # 4. Entropy Loss (Regularization for exploration)
                entropy = dist.entropy().mean()
                entropy_loss = -entropy # Maximize entropy -> minimize negative entropy

                # 5. Total PPO Loss
                total_loss = (policy_loss 
                              + self.vf_coeff * value_loss 
                              + self.ent_coeff * entropy_loss)
                
                # 6. Optimization Step
                self.optimizer.zero_grad()
                total_loss.backward()
                # Gradient Clipping (recommended for stability)
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5) 
                self.optimizer.step()
        
        # 7. Update Old Policy Network (after all PPO Epochs are complete)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # TODO: Return relevant metrics (e.g., loss, kl divergence)
        return total_loss.item() 

# --- 4. Main Training Loop ---

def main_ppo_loop():
    
    # 1. Task Initialization
    state_dim, action_dim = task_initialize_environment()
    # TODO: Initialize the actual environment object
    env = None 
    
    # 2. Initialize PPO Agent
    agent = PPOAgent(state_dim, action_dim)
    
    # 3. Training Parameters
    max_steps = 100000 
    # Batch size N: Number of steps to collect before performing an update
    steps_per_batch = 2048 
    
    # Experience Buffer (Trajectory Buffer)
    all_states, all_actions, all_log_probs, all_rewards, all_dones, all_values = [], [], [], [], [], []
    
    step_count = 0
    # TODO: Handle initial state from environment reset
    current_state = task_reset_environment(env) 
    
    while step_count < max_steps:
        
        # --- Collect Trajectory (Rollout) ---
        
        for t in range(steps_per_batch):
            
            # 4. Select Action
            action, log_prob, value = agent.select_action(current_state)
            
            # 5. Interact with Environment
            next_state, reward, done, _ = task_run_one_step(env, action)

            # 6. Store Experience
            all_states.append(current_state)
            all_actions.append(action)
            all_log_probs.append(log_prob)
            all_rewards.append(reward)
            all_dones.append(done)
            all_values.append(value) # Value estimate V(s_t)
            
            # Update state and step counter
            current_state = next_state
            step_count += 1
            
            if done:
                # Environment episode finished
                # TODO: Record episode length and total reward
                current_state = task_reset_environment(env)
                
        # --- Calculate Next State Value V(s_T) ---
        
        if not done: # If the rollout ended because steps_per_batch reached, not done
             # Get the value estimate V(s_T) for the final state in the buffer
             with torch.no_grad():
                current_state_tensor = torch.from_numpy(current_state).float().unsqueeze(0)
                next_value = agent.policy.critic(current_state_tensor).item()
        else:
             next_value = 0.0 # Terminal state value is 0
        
        # Append V(s_T) to the values list for GAE calculation
        all_values.append(next_value) 
        
        # 7. Calculate GAE Advantages and Returns
        advantages, returns = agent.calculate_gae_and_returns(
            all_rewards, all_values, all_dones, next_value
        )

        # 8. PPO Optimization Update
        loss = agent.update(
            all_states, all_actions, all_log_probs, returns, advantages
        )
        
        # 9. Clean up Buffer for next rollout
        all_states, all_actions, all_log_probs, all_rewards, all_dones, all_values = [], [], [], [], [], []
        
        # TODO: Implement logging and checkpoint saving logic

if __name__ == '__main__':
    main_ppo_loop()
    
# TODO: Add utility functions for loading/saving models