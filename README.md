# SS26 Drone control for various trajectory tracking tasks

[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.1.0-black.svg)](https://isaac-sim.github.io/IsaacLab/v2.1.0/index.html)

## Overview

This project facilitates training policies for various trajectory tracking tasks. 
We provide the following trajectories that are parametrizable in your training.
 - Hover
 - Line
 - Circle
 - Lemniscate
 - Slalom


## Demo Videos

### Circle trajectory

<video controls width="720">
  <source src="videos/circle.mp4" type="video/mp4">
  Your browser does not support HTML video. Download it here: <a href="videos/circle.mp4">circle.mp4</a>
</video>


### Hover trajectory

<video controls width="720">
  <source src="videos/hover.mp4" type="video/mp4">
  Your browser does not support HTML video. Download it here: <a href="videos/hover.mp4">hover.mp4</a>
</video>




## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/pip_installation.html).
  We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/QuadControl

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Quadcontrol-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>

        # for example
        python scripts/skrl/train.py --task Quadcontrol-v0 

        # to train a specific trajectory
        python scripts/skrl/train_custom.py --task Quadcontrol-Meta-v0 --traj_type circle
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.


### To cite us
```
@misc{Chidananda_2025,
  author = {Chidananda, Vikas},
  title = {Trajectory Tracking for Quadrotors in IsaacLab},
  year = {2025},
}
```
