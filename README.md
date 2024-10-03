# latent_learning_successor_representation

This repository contains the source code and data to reproduce the simulations and the figures described in the paper **Revealing the mechanisms underlying latent learning with successor representations** (Menezes, M.; Zeng, X.; Cheng, S.).

## Description

The available code files allow for running latent and direct learning simulations and plotting the agent's data.

The folder `application` contains the code of the RL agent, the demo files, the memory module, and the interfaces between the environment and the agents.

- The `agents` folder contains distinct agents for the gridworld and Tolman simulations
- The `demo` folder contains two demo files for the gridworld and Tolman simulations
- The `interface` folder contains the code that interfaces the actions in the environment with the RL agent
- The `memory_modules` folder contains the memory structure used by the RL agent
- `rl_monitoring` contains a file to show the RL agent learning progress

The folder `data` contains the saved data from previous simulations.

The `plot.py` code allows you to plot data on the escape latency, successor representation, Q-values, and more from the RL agent.

## Getting Started

### Installing

- Install `python3.8`
- Create a venv with `python3.8 -m venv VENV_NAME` and activate the venv with `source VENV_NAME/bin/activate`
- Upgrade setuptools with `pip install pip==24.2 wheel==0.44.0 setuptools==75.1.0`
- Install the requirements with `pip install -r requirements.txt`
- Get the CoBeL-RL 1.1 package (https://github.com/sencheng/CoBeL-RL/releases/tag/1.1)
- Install CoBeL-RL with `pip install /path/to/CoBeL-RL`
- Add an extra python path: `export PYTHONPATH=/your_directory/latent_learning_successor_representation`

### Running Simulations

To quickly get started and test the system, you can run one of the two available demos:

- `application/demo/latent_learning_gridworld.py` to run the gridworld simulation.
- `application/demo/latent_learning_tolman.py` to run the Tolman maze simulation

### Authors

Matheus Menezes (menezes.matheus@lacmor.ufma.br)
