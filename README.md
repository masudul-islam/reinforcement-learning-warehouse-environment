# Reinforcement Learning Warehouse Environment
**CS687 Final Report**  
*Authors: Md Masudul Islam, Zekai Zhang · December 2024*

This project explores three RL algorithms (One-Step Actor-Critic, REINFORCE with Baseline, and Semi-Gradient n-Step SARSA) in a simulated warehouse environment.

- **One-Step Actor-Critic**
- **REINFORCE with Baseline**
- **Episodic Semi-Gradient n-Step SARSA**

These algorithms are applied to a **simulated warehouse environment** featuring stochastic transitions, negative rewards for shelves, and terminal states for item pickups/deliveries.

## Motivation
Warehouse management demands efficient, adaptive policies to handle:
- Unpredictable demand
- Complex workflows
- Multiple constraints (obstacles, limited resources)

RL algorithms can learn robust strategies by interacting with the environment, outperforming manual or rule-based methods in dynamic settings.

## Environment Description
- **States**: Agent’s (x, y) position on a grid plus special shelf/terminal cells  
- **Actions**: Up, Down, Left, Right (with probabilistic drift)  
- **Rewards**:
  - +100 for reaching a terminal state
  - -9 for traversing a shelf
  - -1 for regular grid cells  
- **Transition**: 70% to intended action, 12% each side, 6% no movement

## Algorithms Implemented
1. **One-Step Actor-Critic**  
   Combines a parameterized policy (Actor) with a value function (Critic).  
   - Uses TD error for simultaneous policy and value updates.  
2. **REINFORCE with Baseline**  
   - Policy gradient approach with a baseline to reduce variance.  
   - Calculates returns and updates the policy at episode’s end.  
3. **Episodic Semi-Gradient n-Step SARSA**  
   - Q-learning-like approach with multi-step returns.  
   - Neural network for Q-value approximation.  
   - Transfer learning strategies for larger environments.

## Key Results
- **One-Step Actor-Critic** typically yields stable performance and high rewards (e.g., ~100 on 3×3 grids).  
- **REINFORCE with Baseline** excels in smaller grids, reaching 98–100 reward with careful hyperparameter tuning.  
- **n-Step SARSA** scales effectively but is more sensitive to initialization and hyperparameters.

## Repository Structure
