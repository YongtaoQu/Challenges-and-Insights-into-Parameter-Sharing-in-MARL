# README

This code is primarily forked from [MAPPO Repository](https://github.com/marlbenchmark/on-policy/tree/de66d7a4b23fac2513f56f96f73b3f5cb96695ac). It includes enhancements and additional custom environments for Multi-Agent Reinforcement Learning (MARL).

## Installation

To run this code, follow the original MAPPO installation steps in the [MAPPO repository](https://github.com/marlbenchmark/on-policy). 

Ensure that you also install the required environments according to the original setup instructions.

## Custom Environments

This repository includes several manually designed environments located in `~/mappo/onpolicy/envs/mpe/scenarios`. These environments are:

- **multi-color-spread**
- **4vs5-spread**
- **simple-spread-discrete**
- **golden-point**
- **large-spread**
- **simple-speaker-listener-padded**
- **sacrifice**

Some of these environments are introduced in the project paper, while others are designed specifically for testing purposes.

### Adding Custom Environments

If you wish to add your own environments, use the existing scenarios in `~/mappo/onpolicy/envs/mpe/scenarios` as examples. Ensure that your custom environment adheres to the required format and integrates seamlessly with the MAPPO framework.

## Running the Code

1. **Write a Training Script**
   To train models with the new environments, create a script in `mappo/onpolicy/scripts/train_mpe_scripts` similar to the format of existing scripts.
2. **Configure the Script**
   Update the following parameters in the script:
   - Scenario name (e.g., `multi-color-spread`, `4vs5-spread`, etc.)
   - Number of agents
   - Number of landmarks
   - WandB (Weights & Biases) account settings (if applicable)
3. **Parameter Sharing**
   - By default, parameter sharing is enabled.
   - To disable parameter sharing, add the flag `--share_policy` to the script.

### Observation and Action Space Padding

If you want to add padding to the observation and action spaces, refer to the implementation in `simple-speaker-listener-padded.py`. Follow the instructions provided there to extend the spaces as needed.

### Adding Agent Indication

To include agent indication in an environment, modify the observation function in the corresponding `.py` file for your scenario. Replace the last few lines in observation function with the following code:

```python
for i, other in enumerate(world.agents):
    if other is agent:
        agent_index = np.eye(len(world.agents))[i]
        continue
    comm.append(other.state.c)
    other_pos.append(other.state.p_pos - agent.state.p_pos)
return np.concatenate([agent_index, agent.state.p_vel, agent.state.p_pos] + entity_pos + other_pos + comm)
```

This code appends an agent-specific one-hot encoded indicator to the observation space, allowing the policy to distinguish between agents.
