# Reinforcement_learning_pendulum

This repository focuses on training a reinforcement learning (RL) agent to swing up and stabilize a pendulum from various starting positions. It includes a custom OpenAI Gym environment for simulation and provides both Python and MATLAB implementations.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Customization](#customization)
- [MATLAB Implementation](#matlab-implementation)
- [Requirements](#requirements)
- [Notes](#notes)

## Features
- **Custom Gym Environment**: Simulates the dynamics of a pendulum with adjustable start positions.
- **Modifiable Dynamics and Rewards**: Easily change pendulum parameters and reward functions.
- **Jupyter Notebook**: `Agent.ipynb` for training and executing the RL agent.
- **Automatic Weight Saving**: Trained actor weights are saved in a MATLAB file.
- **MATLAB Integration**: Train an agent directly in MATLAB with pre-configured settings.
- **Simulink Models**:
  - `Env_Agent_Simulation.slx` for agent program execution.
  - `Deployment_Env.slx` for testing on real hardware (Quanser IP02).

## Installation

### Clone the Repository

```bash
git clone git@github.com:samebanimb/Reinforcement_learning_pendulum.git
```
### Install Dependencies
```bash
cd Reinforcement_learning_pendulum/
pip install -e .
```
## Usage

### Python Implementation

**Modify Pendulum Dynamics (Optional):**
- Edit parameters in `utils/model` to change pendulum dynamics.

**Modify Reward Function (Optional):**
- Edit `env/pendulum` to customize the reward function.

**Train the Agent:**
- Open `Agent.ipynb` in Jupyter Notebook.
- Run the notebook to train the agent.
- After training, the actor's weights are automatically saved in a MATLAB file.

**Execution Environment:**
- Operating System: Linux OS or Google Colab is required to access certain packages.

## Customization

### Pendulum Dynamics:
- Adjust parameters in `utils/model.py` for Python.
- Modify `DynamicPendulum.mat` for MATLAB.

### Reward Function:
- Edit `env/pendulum.py` for Python.
- Change `RewardPendulum.mat` for MATLAB.

## MATLAB Implementation

**Requirements:**
- MATLAB version 2022 or later is necessary for using the agent and policy blocks in Simulink.

**Files and Models:**
- **Agent Configuration:** `RL_Agent_simulation.mat`Located in the MATLAB directory.
- **Simulink Models:**
  - `Env_Agent_Simulation.slx`: Open this when executing the agent program.
  - `Deployment_Env.slx`: Use this model for testing on the Quanser IP02 hardware.

**Customization:**
- **Dynamics:** Modify `DynamicPendulum.mat`.
- **Reward Function:** Adjust `RewardPendulum.mat`.

## Requirements

### Python Packages:
- `tensorflow`
- `gym`
- `pygame`
- `tf_agents`

### MATLAB:
- Version 2022 or later.

## Notes
- The Python code should be executed on a Linux operating system or Google Colab due to package dependencies.
- For hardware testing, ensure compatibility with the Quanser IP02 setup.
- The repository provides flexibility for users to experiment with different pendulum dynamics and reward strategies.

For any questions or suggestions, feel free to contact the project maintainer at [samebanimb15@hotmail.com](mailto:samebanimb15@hotmail.com).
