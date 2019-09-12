"""Environment with Bees."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard imports.
import math
import random
import itertools
from typing import Tuple, Dict, Any, List, Set

# Third-party imports.
import numpy as np

# Package imports.
import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# Bees imports.
from agent import Agent
from utils import convert_obs_to_tuple, get_logs

# Set global log file
REPR_LOG, REW_LOG = get_logs()

# Settings for ``__repr__()``.
PRINT_AGENT_STATS = True
PRINT_DONES = True


class Env(MultiAgentEnv):
    """ Environment with bees in it. """

    def __init__(
        self,
        width: int,
        height: int,
        sight_len: int,
        obj_types: int,
        num_agents: int,
        aging_rate: float,
        food_density: float,
        food_size_mean: float,
        food_size_stddev: float,
        n_layers: int,
        hidden_dim: int,
        reward_weight_mean: float,
        reward_weight_stddev: float,
        consts: Dict[str, Any],
    ) -> None:

        self.width = width
        self.height = height
        self.sight_len = sight_len
        self.obj_types = obj_types
        self.num_agents = num_agents
        self.aging_rate = aging_rate
        self.food_density = food_density
        self.food_size_mean = food_size_mean
        self.food_size_stddev = food_size_stddev

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.reward_weight_mean = reward_weight_mean
        self.reward_weight_stddev = reward_weight_stddev

        # pylint: disable=invalid-name
        # Get constants.
        self.consts = consts
        self.LEFT = consts["LEFT"]
        self.RIGHT = consts["RIGHT"]
        self.UP = consts["UP"]
        self.DOWN = consts["DOWN"]
        self.STAY = consts["STAY"]
        self.EAT = consts["EAT"]
        self.MATE = consts["MATE"]
        self.NO_MATE = consts["NO_MATE"]
        self.HEAVEN: Tuple[int, int] = tuple(consts["BEE_HEAVEN"])  # type: ignore

        # Construct object identifier dictionary.
        self.obj_type_id = {"agent": 0, "food": 1}
        self.obj_type_name = {0: "agent", 1: "food"}

        # Compute number of foods.
        num_squares = self.width * self.height
        self.initial_num_foods = math.floor(self.food_density * num_squares)
        self.num_foods = 0

        # Construct ``self.grid``.
        self.grid = np.zeros((self.width, self.height, self.obj_types))

        # Construct observation and action spaces.
        # HARDCODE
        self.action_space = gym.spaces.Tuple(
            (gym.spaces.Discrete(5), gym.spaces.Discrete(2), gym.spaces.Discrete(2))
        )
        num_actions = 5 + 2 + 2
        self.num_actions = num_actions

        # Each observation is a k * k matrix with values from a discrete
        # space of size self.obj_types, where k = 2 * self.sight_len + 1
        outer_list = []
        for _x in range(-self.sight_len, self.sight_len + 1):
            inner_list = []
            for _y in range(-self.sight_len, self.sight_len + 1):
                agent_space = gym.spaces.Discrete(2)
                food_space = gym.spaces.Discrete(2)
                inner_list.append(gym.spaces.Tuple((agent_space, food_space)))
            inner_space = gym.spaces.Tuple(tuple(inner_list))
            outer_list.append(inner_space)
        self.observation_space = gym.spaces.Tuple(tuple(outer_list))

        self.agents: Dict[int, Agent] = {}

        # Misc settings.
        self.dones: Dict[int, bool] = {}
        self.resetted = False
        self.iteration = 0

    def fill(self) -> None:
        """Populate the environment with food and agents."""
        # Reset ``self.grid``.
        self.grid = np.zeros((self.width, self.height, self.obj_types))
        # MOD
        self.num_foods = 0

        # Set unique agent positions.
        grid_positions = list(itertools.product(range(self.height), range(self.width)))
        agent_positions = random.sample(grid_positions, self.num_agents)
        for i, (j, agent) in enumerate(self.agents.items()):
            agent_pos = agent_positions[i]
            REPR_LOG.write("Initializing agent '%d' at '%s'.\n" % (j, str(agent_pos)))
            self._place(self.obj_type_id["agent"], agent_pos)
            agent.pos = agent_pos

        # Set unique food positions.
        assert self.num_foods == 0
        food_positions = random.sample(grid_positions, self.initial_num_foods)
        for food_pos in food_positions:
            self._place(self.obj_type_id["food"], food_pos)
        self.num_foods = self.initial_num_foods

    def reset(self) -> Dict[int, Tuple[Tuple[Tuple[int, ...], ...], ...]]:
        """ Reset the entire environment. """

        # Get average rewards for agents from previous episode.
        if len(self.agents) > 0:
            avg_reward = np.mean([agent.total_reward for _, agent in self.agents.items()])
            REPR_LOG.write("{:.10f}".format(avg_reward) + "\n")

        # MOD
        # Reconstruct agents.
        self.agents = {}
        self.agent_ids_created = 0
        for _ in range(self.num_agents):
            self.agents[self._new_agent_id()] = Agent(
                sight_len=self.sight_len,
                obj_types=self.obj_types,
                consts=self.consts,
                n_layers=self.n_layers,
                hidden_dim=self.hidden_dim,
                num_actions=self.num_actions,
                reward_weight_mean=self.reward_weight_mean,
                reward_weight_stddev=self.reward_weight_stddev,
            )

        self.iteration = 0
        self.resetted = True
        self.dones = {}
        self.fill()

        # Set initial agent observations
        for _, agent in self.agents.items():
            agent.observation = self._get_obs(agent.pos)

        return {i: agent.reset() for i, agent in self.agents.items()}

    def _update_pos(self, pos: Tuple[int, int], move: int) -> Tuple[int, int]:
        """Compute new position from a given move."""
        new_pos = tuple([0, 0])
        if move == self.UP:
            new_pos = tuple([pos[0], pos[1] + 1])
        elif move == self.DOWN:
            new_pos = tuple([pos[0], pos[1] - 1])
        elif move == self.LEFT:
            new_pos = tuple([pos[0] - 1, pos[1]])
        elif move == self.RIGHT:
            new_pos = tuple([pos[0] + 1, pos[1]])
        elif move == self.STAY:
            new_pos = pos
        else:
            REPR_LOG.close()
            raise ValueError("'%s' is not a valid action.")
        return new_pos  # type: ignore

    def _remove(self, obj_type_id: int, pos: Tuple[int, int]) -> None:

        grid_idx = pos + (obj_type_id,)
        if self.grid[grid_idx] != 1:
            REPR_LOG.close()
            raise ValueError(
                "Object '%s' does not exist at grid position '(%d, %d)'."
                % (self.obj_type_name[obj_type_id], pos[0], pos[1])
            )
        self.grid[grid_idx] = 0

    def _place(self, obj_type_id: int, pos: Tuple[int, int]) -> None:

        grid_idx = pos + (obj_type_id,)
        if obj_type_id == self.obj_type_id["agent"] and self.grid[grid_idx] == 1:
            REPR_LOG.close()
            raise ValueError(
                "An agent already exists at grid position '(%d, %d)'."
                % (pos[0], pos[1])
            )
        self.grid[grid_idx] = 1

    def _obj_exists(self, obj_type_id: int, pos: Tuple[int, int]) -> bool:

        grid_idx = pos + (obj_type_id,)
        return self.grid[grid_idx] == 1

    def _move(
        self, action_dict: Dict[int, Tuple[int, int, int]]
    ) -> Dict[int, Tuple[int, int, int]]:
        """ Identify collisions and update ``action_dict``,
            ``self.grid``, and ``agent.pos``.
        """
        # Shuffle the keys.
        shuffled_items = list(action_dict.items())
        random.shuffle(shuffled_items)
        for agent_id, action in shuffled_items:
            agent = self.agents[agent_id]
            pos = agent.pos
            move, consume, mate = action
            assert pos is not None

            new_pos = self._update_pos(pos, move)

            # Validate new position.
            out_of_bounds = False
            if new_pos[0] < 0 or new_pos[0] >= self.width:
                out_of_bounds = True
            if new_pos[1] < 0 or new_pos[1] >= self.height:
                out_of_bounds = True

            if out_of_bounds or self._obj_exists(self.obj_type_id["agent"], new_pos):
                action_dict[agent_id] = (self.STAY, consume, mate)
            else:
                self._remove(self.obj_type_id["agent"], pos)
                self._place(self.obj_type_id["agent"], new_pos)
                agent.pos = new_pos

        return action_dict

    def _consume(self, action_dict: Dict[int, Tuple[int, int, int]]) -> None:
        """ Takes as input a collision-free ``action_dict`` and
            executes the ``consume`` action for all agents.
        """
        for agent_id, action in action_dict.items():

            agent = self.agents[agent_id]
            pos = agent.pos

            # If the agent is dead, don't do anything
            if agent.health <= 0.0:
                continue

            # If they try to eat when there's nothing there, do nothing.
            _, consume, _ = action
            if self._obj_exists(self.obj_type_id["food"], pos) and consume == self.EAT:
                self._remove(self.obj_type_id["food"], pos)
                self.num_foods -= 1
                food_size = np.random.normal(self.food_size_mean, self.food_size_stddev)
                REPR_LOG.write("Num foods: '%d'.\n" % self.num_foods)
                REPR_LOG.write(
                    "Updating agent '%d' health from '%f' to '%f'.\n"
                    % (agent_id, agent.health, min(1, agent.health + food_size))
                )
                agent.health = min(1, agent.health + food_size)

    def _mate(self, action_dict: Dict[int, Tuple[int, int, int]]) -> Set[int]:
        """
        Takes as input a collision-free ``action_dict`` and
        executes the ``mate`` action for all agents.
        Returns a set of the ids of the newly created children.
        """
        child_ids = set()
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            pos = agent.pos

            # If the agent is dead, don't do anything
            if agent.health <= 0.0:
                continue

            # Grab action, do nothing if the agent chose not to mate.
            _, _, mate = action
            if mate == self.NO_MATE:
                continue

            # Search adjacent positions for possible mates
            adj_positions = self._get_adj_positions(pos)
            next_to_agent = False
            for adj_pos in adj_positions:
                if self._obj_exists(self.obj_type_id["agent"], adj_pos):
                    next_to_agent = True
                    mate_pos = adj_pos
                    break

            # If there is another agent in an adjacent position, spawn child.
            if next_to_agent:

                # Choose child location
                open_positions = self._get_adj_positions(pos)
                open_positions += self._get_adj_positions(mate_pos)
                open_positions = list(set(open_positions))
                open_positions = [
                    open_pos
                    for open_pos in open_positions
                    if not self._obj_exists(self.obj_type_id["agent"], open_pos)
                ]

                # Only create a new child if there are valid open positions.
                if open_positions != []:
                    child_pos = random.choice(open_positions)

                    # Place child and add to ``self.grid``
                    child = Agent(
                        sight_len=self.sight_len,
                        obj_types=self.obj_types,
                        consts=self.consts,
                        n_layers=self.n_layers,
                        hidden_dim=self.hidden_dim,
                        num_actions=self.num_actions,
                        pos=child_pos,
                        reward_weight_mean=self.reward_weight_mean,
                        reward_weight_stddev=self.reward_weight_stddev,
                    )
                    child_id = self._new_agent_id()
                    self.agents[child_id] = child
                    REPR_LOG.write("Adding child with id '%d'.\n" % child_id)
                    child_ids.add(child_id)

                    REPR_LOG.write(
                        "Placing child '%d' at '%s'.\n" % (child_id, str(child_pos))
                    )
                    self._place(self.obj_type_id["agent"], child_pos)

        return child_ids

    def _get_obs(self, pos: Tuple[int, int]) -> Tuple[Tuple[Tuple[int, ...]]]:
        """ Returns a ``np.ndarray`` of observations given an agent ``pos``. """

        # Calculate bounds of field of vision.
        x = pos[0]
        y = pos[1]
        sight_left = x - self.sight_len
        sight_right = x + self.sight_len
        sight_bottom = y - self.sight_len
        sight_top = y + self.sight_len

        # Calculate length of zero-padding in case sight goes out of bounds.
        pad_left = max(-sight_left, 0)
        pad_right = max(sight_right - self.width + 1, 0)
        pad_bottom = max(-sight_bottom, 0)
        pad_top = max(sight_top - self.height + 1, 0)

        # Constrain field of vision within grid bounds.
        sight_left = max(sight_left, 0)
        sight_right = min(sight_right, self.width - 1)
        sight_bottom = max(sight_bottom, 0)
        sight_top = min(sight_top, self.height - 1)

        # Construct observation.
        obs_len = 2 * self.sight_len + 1
        obs = np.zeros((obs_len, obs_len, self.obj_types))
        pad_x_len = obs_len - pad_left - pad_right
        pad_y_len = obs_len - pad_top - pad_bottom
        obs[
            pad_left : pad_left + pad_x_len, pad_bottom : pad_bottom + pad_y_len
        ] = self.grid[sight_left : sight_right + 1, sight_bottom : sight_top + 1]
        obs = convert_obs_to_tuple(obs)

        return obs

    def get_action_dict(self) -> Dict[int, Tuple[int, int, int]]:
        """
        Constructs ``action_dict`` by querying individual agents for
        their actions based on their observations.
        """
        action_dict = {}

        for agent_id, agent in self.agents.items():
            action_dict[agent_id] = agent.get_action()

        return action_dict

    def step(
        self, action_dict: Dict[int, Tuple[int, int, int]]
    ) -> Tuple[
        Dict[int, np.ndarray], Dict[int, float], Dict[Any, bool], Dict[int, Any]
    ]:
        """
        ``action_dict`` has agent indices as keys and a dict of the form
        ``{"move": <move>, "consume": <consume>)`` where the dict values
        are strings from the sets
            ``movements = set(["up", "down", "left", "right", "stay"])``
            ``consumptions = set(["eat", "noeat"])``.
        """
        REPR_LOG.write("===STEP===\n")
        REPR_LOG.flush()

        # Execute move, consume, and mate actions, and calculate reward
        obs: Dict[int, np.ndarray] = {}
        rew: Dict[int, float] = {}
        done: Dict[Any, bool] = {}
        info: Dict[int, Any] = {}

        # Execute actions
        prev_health = {
            agent_id: agent.health for agent_id, agent in self.agents.items()
        }
        action_dict = self._move(action_dict)
        self._consume(action_dict)
        child_ids = self._mate(action_dict)

        # Compute reward.
        for agent_id, agent in self.agents.items():
            if agent.health > 0.0 and agent_id not in child_ids:
                rew[agent_id] = agent.compute_reward(
                    prev_health[agent_id], action_dict[agent_id]
                )
            # First reward for children is zero.
            if agent_id in child_ids:
                rew[agent_id] = 0

        # Decrease agent health, compute observations and dones.
        killed_agent_ids = []
        for agent_id, agent in self.agents.items():
            REPR_LOG.write("Agent '%d' health: '%f'.\n" % (agent_id, agent.health))
            if agent.health > 0.0:
                agent.health -= self.aging_rate
                obs[agent_id] = self._get_obs(agent.pos)
                agent.observation = obs[agent_id]
                # MOD: num_foods == 0 -> num_foods <= 0
                done[agent_id] = self.num_foods <= 0 or agent.health <= 0.0

                # Kill agent if ``done[agent_id]`` and remove from ``self.grid``.
                if done[agent_id]:

                    REPR_LOG.write("Killing agent '%d'.\n" % agent_id)
                    self._remove(self.obj_type_id["agent"], agent.pos)
                    agent.pos = self.HEAVEN
                    killed_agent_ids.append(agent_id)

        # Remove killed agents from self.agents
        for killed_agent_id in killed_agent_ids:
            self.agents.pop(killed_agent_id)

        done["__all__"] = all(done.values())
        self.dones = dict(done)

        # Write environment representation to log
        self._log_state()

        self.iteration += 1
        return obs, rew, done, info

    def _get_adj_positions(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Returns the positions adjacent (left, right, up, and down) to a given
        position.
        """

        x, y = pos
        adj_positions = [
            (x + dX, y + dY) for dX, dY in itertools.product([-1, 1], repeat=2)
        ]
        random.shuffle(adj_positions)

        # Validate new positions.
        validated_adj_positions = []
        for new_pos in adj_positions:
            out_of_bounds = False
            if new_pos[0] < 0 or new_pos[0] >= self.width:
                out_of_bounds = True
            if new_pos[1] < 0 or new_pos[1] >= self.height:
                out_of_bounds = True
            if not out_of_bounds:
                validated_adj_positions.append(new_pos)

        return validated_adj_positions

    def __repr__(self) -> str:
        """
        Returns a representation of the environment state.
        """
        output = "\n"

        # Print grid.
        for y in range(self.height):
            for x in range(self.width):

                pos = (x, y)
                object_id = "_"

                # Check if there is an agent in ``pos``.
                if self._obj_exists(self.obj_type_id["agent"], pos):
                    object_id = "B"
                # NOTE: ``B`` currently overwrites ``*``.
                # Check if there is a food in ``pos``.
                elif self._obj_exists(self.obj_type_id["food"], pos):
                    object_id = "*"

                output += object_id + " "

            output += "\n"

        # Print agent stats.
        if PRINT_AGENT_STATS:
            for agent_id, agent in self.agents.items():
                if agent.health > 0.0:
                    output += "Agent %d: " % agent_id
                    output += agent.__repr__()
            output += "\n"

        # Print dones.
        if PRINT_DONES:
            output += "Dones: " + str(self.dones) + "\n"

        REPR_LOG.flush()

        return output

    def _log_state(self) -> None:
        """
        Logs the state of the environment as a string to a
        prespecified log file path.
        """

        REPR_LOG.write("Iteration %d:\n" % self.iteration)
        REPR_LOG.write(self.__repr__())
        REPR_LOG.write(",\n")

    def _new_agent_id(self) -> int:
        self.agent_ids_created += 1
        return self.agent_ids_created - 1
