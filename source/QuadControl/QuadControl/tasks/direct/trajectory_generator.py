import numpy as np
import torch

# trajectory parameters-limits
DEAFULT_CONFIG = {
    "circle": {"radius": 2, "height": 0.0, "speed": 0.3},
    "lemniscate": {"a": 1.5, "height": 2.0, "speed": 0.2},
    # "slalom": {"length": 3.0, "width": 1.0, "height": 2.0, "speed": 0.2},
    # "helix": {"radius": 1.0, "height_speed": 0.3, "speed": 0.5},
    "straight": {"speed": 0.2, "direction": 1},
    "hover": {"x": 1.2, "y": 1.0, "height": 1.0},
}

# limits for random sampling of trajectory parameters
config = {
    "straight": {
        "speed": [0.1, 0.5],
        "x": [-1, 1],  # x and y direction can be -1 or 1
        "y": [-1, 1],
    },
    "circle": {
        "radius": [2, 3],
        # "height": [0.5, 1.5],
        "height": [0.0, 0.0],
        "speed": [0.1, 0.3],
    },
    "lemniscate": {"a": [1, 2], "height": [0.1, 1], "speed": [0.1, 0.3]},
    # "slalom": {
    #     "length": [2, 5],
    #     "width": [0.5, 2],
    #     "height": [0.5, 2],
    #     "speed": [0.05, 0.2],
    # },
    # "helix": {"radius": [0.5, 2], "height_speed": [0.1, 0.5], "speed": [0.1, 0.3]},
    "hover": {"x": [-4.0, 4.0], "y": [-4.0, 4.0], "height": [0.5, 1.0]},
}


class TrajectoryGenerator:
    """
    Generates different types of trajectories as goal point for the quadcopter to follow.
    Supported trajectories:
        - circle
        - lemniscate
        - slalom
        - helix
        - straight (hover)
    Each trajectory has its own set of parameters that can be randomized at the start of each episode.
    The trajectory is generated in the world frame with Z as up-axis.
    Args:
        randomise (bool): Whether to randomize trajectory parameters at the start of each episode.
        device (str): Device to run the computations on.
    Returns:
        pos (torch.Tensor): Desired position at the current time step.
    """

    def __init__(
        self,
        randomise_family=False,
        randomise_task=False,
        trajectory_type=None,
        device="cuda:0",
        dt=0.01,
        num_lookahead=3,
    ):
        self.device = device
        self.time = 0.0
        self.params = DEAFULT_CONFIG.copy()
        self.randomise_family = randomise_family
        self.randomise_task = randomise_task
        self.task_families = list(config.keys())
        self.trajectory_type = trajectory_type
        self.dt = dt  # timestep
        self.num_lookahead = num_lookahead  # number of future goals to return

    def reset(self, time):

        if (self.trajectory_type is None) or (self.randomise_family):
            self.trajectory_type = np.random.choice(self.task_families)

        print(f"Selected trajectory type: {self.trajectory_type}")

        if self.randomise_task:
            self._get_params(time)
        else:
            print("Using default trajectory parameters.")
            param_dict = DEAFULT_CONFIG[self.trajectory_type]
            for param in param_dict.keys():
                self.params[self.trajectory_type][param] = torch.full_like(
                    time, fill_value=param_dict[param], device=self.device
                )

    def _get_params(self, time, partial=False):
        param_dict = config[self.trajectory_type]
        if partial:
            # print("Partial randomization of trajectory parameters.")
            indices = (time == 0.0).nonzero(as_tuple=True)[0]
            t_subset = time[indices]

            for param, (low, high) in param_dict.items():
                self.params[self.trajectory_type][param][indices] = torch.zeros_like(t_subset).uniform_(low, high)
                # print(
                # f"Randomized {self.trajectory_type} param {param} to {self.params[self.trajectory_type][param]}"
                # )
        else:
            print("Full randomization of trajectory parameters.")
            for param, (low, high) in param_dict.items():
                self.params[self.trajectory_type][param] = torch.zeros_like(time).uniform_(low, high)

    def update(self, dt):
        self.time += dt

    def circle(self, t, center=(0.0, 0.0, 0.5)):
        if (t == 0.0).any():
            self._get_params(t, partial=True)

        r = self.params["circle"]["radius"]
        h = self.params["circle"]["height"]
        freq = self.params["circle"]["speed"]

        # x = center[0] + r * torch.cos(2 * torch.pi * freq * t)
        x = -r + r * torch.cos(2 * torch.pi * freq * t)
        y = center[1] + r * torch.sin(2 * torch.pi * freq * t)
        z = center[2] + h * torch.sin(0.5 * torch.pi * freq * t)
        return torch.stack((x, y, z), dim=1).squeeze()

    def lemniscate(self, t, center=(0, 0, 0.5)):
        if (t == 0.0).any():
            self._get_params(t, partial=True)

        a = self.params["lemniscate"]["a"]
        freq = self.params["lemniscate"]["speed"]
        # scale = 1 / (1 + torch.sin(2 * torch.pi * freq * t) ** 2)

        # x = center[0] + a * torch.cos(2 * torch.pi * freq * t) * scale
        # y = center[1] + a * torch.sin(4 * torch.pi * freq * t) * scale
        # z = center[2] + 0.0 * torch.sin(3 * torch.pi * freq * t)

        # trying with lemniscate of Gerono
        x = center[0] + a * torch.sin(2 * torch.pi * freq * t)
        y = center[1] + a * torch.sin(2 * torch.pi * freq * t) * torch.cos(2 * torch.pi * freq * t)
        z = center[2] + 0.0 * torch.cos(4 * torch.pi * freq * t)

        return torch.stack((x, y, z), dim=1).squeeze()

    def slalom(self, t, center=(0, 0, 0.5)):
        L = self.params["slalom"]["length"]
        w = self.params["slalom"]["width"]
        freq = self.params["slalom"]["speed"]

        x = center[0] + w * torch.sin(2 * torch.pi * freq * t)
        y = -L + L * torch.cos(2 * torch.pi * freq * t)
        z = center[2] + 0.0 * torch.cos(4 * torch.pi * freq * t)

        return torch.stack((x, y, z), dim=-1).squeeze()

    def helix(self, t, center=(0, 0, 1)):
        r = self.params["helix"]["radius"]
        freq = self.params["helix"]["speed"]
        h_speed = self.params["helix"]["height_speed"]
        x = center[0] + r * torch.cos(2 * torch.pi * freq * t)
        y = center[1] + r * torch.sin(2 * torch.pi * freq * t)
        z = center[2] + h_speed * t
        return torch.stack((x, y, z), dim=-1).squeeze()

    def straight(self, t):
        if (t == 0.0).any():
            self._get_params(t, partial=True)

        x, y = self.params["straight"]["x"], self.params["straight"]["y"]
        speed = self.params["straight"]["speed"]

        next_waypoint = torch.concat((
            x * speed,
            y * speed,
            torch.zeros_like(speed),
        )).reshape(3, -1)
        time_step = torch.floor(t / 0.034).reshape(1, -1)
        pos = next_waypoint * time_step
        return pos.T

    def hover(self, t):
        # """
        # rand is static, static is rand
        # """
        if (t == 0.0).any():
            self._get_params(t, partial=True)
            # print("Resetting hover trajectory parameters.")
            # reset hover positions for those time steps
            # indices = (t == 0.0).nonzero(as_tuple=True)[0]
            # t_subset = t[indices]
            # self.params["hover"]["x"][indices] = torch.zeros_like(t_subset).uniform_(-2.0, 2.0)
            # self.params["hover"]["y"][indices] = torch.zeros_like(t_subset).uniform_(-2.0, 2.0)
            # self.params["hover"]["height"][indices] = torch.zeros_like(t_subset).uniform_(0.5, 1.5)

        # print("Here in hover trajectory generation.")
        x = self.params["hover"]["x"]
        y = self.params["hover"]["y"]
        z = self.params["hover"]["height"]
        return torch.stack((x, y, z), dim=-1).squeeze()

    def generate_trajectory(self, time):
        trajectory_type = self.trajectory_type
        self.time = time
        # print(f"Generating trajectory of type: {trajectory_type}")
        if trajectory_type == "circle":
            pos = self.circle(self.time)
        elif trajectory_type == "lemniscate":
            pos = self.lemniscate(self.time)
        elif trajectory_type == "slalom":
            pos = self.slalom(self.time)
        elif trajectory_type == "helix":
            pos = self.helix(self.time)
        elif trajectory_type == "straight":  # Default to hover
            pos = self.straight(self.time)
        else:  # hover
            # pos = torch.zeros_like(self.time).uniform_(0.0, 2.0)  # hover at (0,0,1)
            pos = self.hover(self.time)

        return pos.to(device=self.device)

    def generate_multiple_goals(self, time, num_goals=None):
        """
        Generate current goal + N future lookahead goals.

        Returns:
            torch.Tensor: shape [num_envs, num_goals, 3] containing goal positions
        """
        if num_goals is None:
            num_goals = self.num_lookahead

        trajectory_type = self.trajectory_type
        goals = []

        for i in range(num_goals):
            t_future = time + i * self.dt

            if trajectory_type == "circle":
                pos = self.circle(t_future)
            elif trajectory_type == "lemniscate":
                pos = self.lemniscate(t_future)
            elif trajectory_type == "slalom":
                pos = self.slalom(t_future)
            elif trajectory_type == "helix":
                pos = self.helix(t_future)
            elif trajectory_type == "straight":
                pos = self.straight(t_future)
            else:
                pos = self.hover(t_future)

            goals.append(pos)

        # Stack into [num_envs, num_goals, 3]
        if len(goals) > 1:
            goals = torch.stack(goals, dim=1)
        else:
            goals = goals[0].squeeze(1)  # add lookahead dimension

        return goals.to(device=self.device)
