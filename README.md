# latent_learning_successor_representation

This repository contains the source code and data to reproduce the simulations and the figures described in the paper **Revealing the mechanisms underlying latent learning with successor representations** (Menezes, M.; Zeng, X.; Cheng, S.).

## Description

The available code files allow for running latent and direct learning simulations and plotting the agent's data.

The folder `application` contains the code of the RL agent, the demo files, the memory module, and the interfaces between the environment and the agents.

- The `agents` folder contains distinct agents for the gridworld and Tolman simulations
- The `demo` folder contains two demo files for the gridworld and Tolman simulations
- The `environments` folder contains the Blender 3D environment models and inteface scripts
- The `frontends` folder contains control and communicate scripts with the Blender-based simulations
- The `interface` folder contains the code that interfaces the actions in the environment with the RL agent
- The `memory_modules` folder contains the memory structure used by the RL agent
- `rl_monitoring` contains a file to show the RL agent learning progress
- The `topology_graphs` folder contains the manually defined topology graph (no-rotation) implementation that builds nodes/edges from the world

The folder `data/one-hot` contains the saved data from previous one-hot encoding simulations.

The `plot-oh.py` code allows allows you to plot data related to one-hot encoding simulations, including escape latency, successor representation, Q-values, and more from the RL agent.

The `plot-image.py` code allows allows you to plot data related to image-based simulations, including escape latency, successor representation, Q-values, and more from the RL agent.

## Getting Started

### Installing

- Install `python3.8`
- Create a venv with `python3.8 -m venv VENV_NAME` and activate the venv with `source VENV_NAME/bin/activate`
- Upgrade setuptools with `pip install pip==24.2 wheel==0.44.0 setuptools==75.1.0`
- Image-based simulations depend on Blender2.79b. Download and install Blender2.79b at: `https://download.blender.org/release/Blender2.79/` Note : Only v2.79b is supported. Newer versions of Blender will not work with the system.  
- Install the requirements with `pip install -r requirements.txt`
- Get the CoBeL-RL 1.1 package (https://github.com/sencheng/CoBeL-RL/releases/tag/1.1).
- Install CoBeL-RL with `pip install /path/to/CoBeL-RL`
- Add extra python paths: `export PYTHONPATH=/your_directory_cobel/CoBeL-RL-1.1/:/your_directory_application/latent_learning_successor_representation`
- If you wish to run image-based simulations, add an Blender path: `export BLENDER_EXECUTABLE_PATH=/your_blender_path/blender-2.79-linux-glibc219-x86_64/blender`

### Running Simulations

To quickly get started and test the system, you can run one of the two available demos:

- `application/demo/latent_learning_gridworld.py` to run the deep successor representation one-hot encoding gridworld simulation.
- `application/demo/latent_learning_tolman.py` to run the deep successor representation one-hot encoding Tolman maze simulation
- `application/demo/latent_learning_dsr.py` to run the deep successor representation image-based simulation. Change the `demo_scene` variable to run either the gridworld or Tolman maze simulations
- `application/demo/latent_learning_dqn.py` to run the Dyna-DQN image-based simulation. Change the `demo_scene` variable to run either the gridworld or Tolman maze simulations

### Authors

Matheus Menezes (menezes.matheus@lacmor.ufma.br)
