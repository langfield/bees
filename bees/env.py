"""Environment with Bees."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard imports.
import os
import math
import random
import functools
import itertools
from pprint import pformat
from typing import Tuple, Dict, Any, List, Set, TextIO, Optional
import pickle

# Third-party imports.
import torch
import torch.nn.functional as F
import numpy as np

# Package imports.
import gym

# Bees imports.
from bees.agent import Agent
from bees.genetics import get_child_reward_network
from bees.config import Config
from bees.utils import DEBUG, flat_action_to_tuple

# Settings for ``__repr__()``.
PRINT_AGENT_STATS = True
PRINT_FAST = True

# Parameter for exponential moving average
ALPHA = 0.9
NORMALIZER = 1000

# pylint: disable=bad-continuation


class Env(Config):
    """
    Environment with bees in it. Note that all of the parameters are contained in the
    ``config`` argument.

    Parameters
    ----------
    width : ``int``.
        Width of the environment grid.
    height : ``int``.
        Height of the environment grid.
    sight_len : ``int``.
        How far agents are able to see in each cardinal direction.
    num_obj_types : ``int``.
        The number of distinct entity classes in the environment. Note that
        we currently have only two (agents, food).
    num_agents : ``int``.
        Initial number of agents in the environment.
    aging_rate : ``float``.
        The amount of health agents lose on each timestep.
    initial_food_density : ``float``.
        The initial proportion of food in the grid.
    initial_food_regen_prob : ``float``.
        The initial probability of food regeneration per square per timestep.
    food_size_mean : ``float``.
        The mean of the Gaussian from which food size is sampled.
    food_size_stddev : ``float``.
        The standard deviation of the Gaussian from which food size is sampled.
    mating_cooldown_len : ``int``.
        How long agents must wait in between mate actions.
    target_agent_density: ``float``.
        The target agent density for adaptive food regeneration rate.
    print_repr: ``bool``.
        Whether or not to print environment repr at each iteration.
    n_layers : ``int``.
        Number of layers in the reward network.
    hidden_dim : ``int``.
        Hidden dimension of the reward network.
    reward_weight_mean : ``float``.
        Mean for weight initialization distribution.
    reward_weight_stddev : ``float``.
        Standard deviation for weight initialization distribution.
    mut_sigma : ``float``.
        Standard deviation of mutation operation on reward vectors.
    mut_p : ``float``.
        Probability of mutation on reward vectors.
    consts : ``Dict[str, Any]``.
        Dictionary of various constants.
    """

    def __init__(self, config: Config) -> None:

        # TODO: Consider replacing all ``env.config.<attr>`` with ``env.<attr>``.
        self.config = config

        # Initialize ``__dict__``.
        super().__init__(config.settings, mutable=True)

        # Cast ``self.HEAVEN`` to tuple because json doesn't support tuples.
        self.HEAVEN: Tuple[int, int] = tuple(self.HEAVEN)  # type: ignore

        # Construct object identifier dictionary.
        # HARDCODE
        self.obj_type_ids = {"agent": 0, "food": 1}
        self.obj_type_names = {0: "agent", 1: "food"}

        # Compute number of foods.
        num_squares = self.width * self.height
        self.initial_num_foods = math.floor(self.initial_food_density * num_squares)
        self.num_foods = 0
        self.food_regen_prob = self.initial_food_regen_prob

        # Construct ``self.grid`` and ``self.id_map``.
        self.grid = np.zeros((self.width, self.height, self.num_obj_types))
        self.id_map: List[List[Dict[int, Set[int]]]] = [
            [{} for y in range(self.height)] for x in range(self.width)
        ]

        # Construct observation and action spaces.
        # HARDCODE
        self.subaction_sizes = [5, 2, 2]
        self.num_actions = functools.reduce(lambda a, b: a * b, self.subaction_sizes)
        self.action_space = gym.spaces.Discrete(self.num_actions)

        obs_len = 2 * self.sight_len + 1
        low_obs = np.zeros((self.num_obj_types, obs_len, obs_len))
        high_obs = np.zeros((self.num_obj_types, obs_len, obs_len))
        self.observation_space = gym.spaces.Box(low_obs, high_obs)

        self.agents: Dict[int, Agent] = {}
        self.agent_ids_created = 0

        self.avg_agent_lifetime: float = -1.0

        # Misc settings.
        self.dones: Dict[int, bool] = {}
        self.resetted = False
        self.iteration = 0

    def fill(self) -> None:
        """
        Populate the environment with food and agents.

        Updates
        -------
        self.grid : ``np.ndarray``.
            Grid containing agents and food.
            Shape: ``(width, height, num_obj_types)``.
        self.id_map : ``List[List[Dict[int, Set[int]]]]``.
            List of lists in the shape of the grid which maps object type ids to
            a set of object ids of objects of that type at that position in the grid.
        self.num_foods : ``int``.
            Number of foods in the environment.
        """
        # TODO: Add updates from calls to ``self._place()``.

        # Reset ``self.grid`` and ``self.id_map``.
        self.grid = np.zeros((self.width, self.height, self.num_obj_types))
        self.id_map = [[{} for y in range(self.height)] for x in range(self.width)]
        for obj_type_id in self.obj_type_ids.values():
            for x, y in itertools.product(range(self.width), range(self.height)):
                self.id_map[x][y][obj_type_id] = set()
        self.num_foods = 0

        # Set unique agent positions.
        grid_positions = list(itertools.product(range(self.width), range(self.height)))

        agent_positions = random.sample(grid_positions, self.num_agents)
        for i, (agent_id, agent) in enumerate(self.agents.items()):
            agent_pos = agent_positions[i]
            self._place(self.obj_type_ids["agent"], agent_pos, agent_id)
            agent.pos = agent_pos

        # Set unique food positions.
        assert self.num_foods == 0

        food_positions = random.sample(grid_positions, self.initial_num_foods)
        for food_pos in food_positions:
            self._place(self.obj_type_ids["food"], food_pos)
        self.num_foods = self.initial_num_foods

    def reset(self) -> Dict[int, np.ndarray]:
        """
        Reset the entire environment.

        Updates
        -------
        self.agents : ``Dict[int, Agent]``.
            Map from agent ids to ``Agent`` objects.
        self.iteration : ``int``.
            The current environment iteration.
        self.resetted : ``bool``.
            Whether the envrionment has been reset.
        self.dones : ``Dict[int, bool]``.
            Map from agent ids to death status.
        self.fill() : ``Callable``.
            All updates made during calls to this function.

        Returns
        -------
        obs : ``Dict[int, Tuple[Tuple[Tuple[int, ...], ...], ...]]``.
            Initial agent observations.
        """

        # Reconstruct agents.
        self.agents = {}
        self.agent_ids_created = 0
        for _ in range(self.num_agents):
            self.agents[self._new_agent_id()] = Agent(
                config=self.config,
                num_actions=self.num_actions,
                pos=(0, 0),
                initial_health=1,
            )

        self.iteration = 0
        self.resetted = True
        self.dones = {}
        self.fill()

        # Set initial agent observations
        for _, agent in self.agents.items():
            agent.observation = self._get_obs(agent.pos)
        obs = {i: agent.reset() for i, agent in self.agents.items()}

        return obs

    def _update_pos(self, pos: Tuple[int, int], move: int) -> Tuple[int, int]:
        """
        Compute new position from a given move.

        Parameters
        ----------
        pos : ``Tuple[int, int]``.
            An agent's current position.
        move : ``int``.
            Selected move action.

        Returns
        -------
        new_pos : ``Tuple[int, int]``.
            Resultant position after move.
        """
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
            raise ValueError("'%s' is not a valid action." % move)

        return new_pos  # type: ignore

    def _remove(
        self, obj_type_id: int, pos: Tuple[int, int], obj_id: Optional[int] = None
    ) -> None:
        """
        Remove an object of type ``obj_type_id`` from the grid at ``pos``.
        Optionally remove the object from ``self.id_map`` if it has an identifier.

        Parameters
        ----------
        obj_type_id : ``int``.
            The object type of the object being removed.
        pos : ``Tuple[int, int]``.
            The position of the object being removed.
        obj_id : ``int``, optional.
            The id of the object being removed, if its object type supports ids.

        Updates
        -------
        self.grid : ``np.ndarray``.
            Grid containing agents and food.
            Shape: ``(width, height, num_obj_types)``.
        self.id_map : ``List[List[Dict[int, Set[int]]]]``.
            List of lists in the shape of the grid which maps object type ids to
            a set of object ids of objects of that type at that position in the grid.
        """
        x = pos[0]
        y = pos[1]

        # Remove from ``self.grid``.
        grid_idx = pos + (obj_type_id,)
        if self.grid[grid_idx] != 1:
            raise ValueError(
                "Object '%s' does not exist at grid position '(%d, %d)'."
                % (self.obj_type_names[obj_type_id], x, y)
            )
        self.grid[grid_idx] = 0

        # Remove from ``self.id_map``.
        if obj_id is not None:
            object_map: Dict[int, Set[int]] = self.id_map[x][y]
            objects = object_map[obj_type_id]
            if obj_id not in objects:
                raise ValueError(
                    "Object of type '%s' with identifier '%d' cannot be removed from\
                     grid position '(%d, %d)' since it does not exist there."
                    % (self.obj_type_names[obj_type_id], obj_id, x, y)
                )
            self.id_map[x][y][obj_type_id].remove(obj_id)

    def _place(
        self, obj_type_id: int, pos: Tuple[int, int], obj_id: Optional[int] = None
    ) -> None:
        """
        Place an object of type ``obj_type_id`` at the grid at ``pos``.
        Optionally place the object in ``self.id_map`` if it has an identifier.

        Parameters
        ----------
        obj_type_id : ``int``.
            The object type of the object being removed.
        pos : ``Tuple[int, int]``.
            The position of the object being removed.
        obj_id : ``int``, optional.
            The id of the object being removed, if its object type supports ids.

        Updates
        -------
        self.grid : ``np.ndarray``.
            Grid containing agents and food.
            Shape: ``(width, height, num_obj_types)``.
        self.id_map : ``List[List[Dict[int, Set[int]]]]``.
            List of lists in the shape of the grid which maps object type ids to
            a set of object ids of objects of that type at that position in the grid.
        """
        x = pos[0]
        y = pos[1]

        # Add to ``self.grid``.
        grid_idx = pos + (obj_type_id,)
        if obj_type_id == self.obj_type_ids["agent"] and self.grid[grid_idx] == 1:
            raise ValueError(
                "An agent already exists at grid position '(%d, %d)'." % (x, y)
            )

        self.grid[grid_idx] = 1

        # Add to ``self.id_map``.
        if obj_id is not None:
            object_map: Dict[int, Set[int]] = self.id_map[x][y]
            if obj_type_id not in object_map:
                object_map[obj_type_id] = set()
            objects = object_map[obj_type_id]
            if obj_id in objects:
                raise ValueError(
                    "Object of type '%s' with " % self.obj_type_names[obj_type_id]
                    + "identifier '%d' cannot be placed at grid position " % obj_id
                    + "'(%d, %d)' since an object of the same type with the " % (x, y)
                    + "same id already exists there."
                )
            self.id_map[x][y][obj_type_id].add(obj_id)

    def _obj_exists(self, obj_type_id: int, pos: Tuple[int, int]) -> bool:
        """
        Check if an object of object type ``obj_type_id`` exists at the given position.

        Parameters
        ----------
        obj_type_id : ``int``.
            The object type of the object being removed.
        pos : ``Tuple[int, int]``.
            The position of the object being removed.

        Returns
        -------
        in_grid : ``bool``.
            Whether there is an object of that object type at ``pos``.

        Raises
        ------
        ValueError
            If ``pos[i]`` < 0 for any i, or if ``obj_type_id`` is invalid, or if
            ``self.grid`` and ``self.ip_map`` don't agree.
        """

        # Make sure position indices are nonnegative.
        # TODO: Convert all error strings to f-strings, or externalize them.
        if pos[0] < 0 or pos[1] < 0:
            raise ValueError(f"Pos ``{pos}`` shouldn't have negative elements.")

        # Make sure ``obj_type_id`` is valid.
        if (
            obj_type_id not in self.obj_type_ids.values()
            or obj_type_id not in self.obj_type_names
        ):
            raise ValueError(f"Object type id ``{obj_type_id}`` invalid.")

        # Check grid.
        grid_idx: Tuple[int, int, int] = pos + (obj_type_id,)
        in_grid: bool = self.grid[grid_idx] == 1

        # HARDCODE: foods do not have unique identifiers, not in ``id_map``.
        if obj_type_id == self.obj_type_ids["food"]:
            return in_grid

        # Check id_map.
        x = pos[0]
        y = pos[1]
        object_map: Dict[int, Set[int]] = self.id_map[x][y]
        if obj_type_id not in object_map:
            in_id_map = False
        else:
            objects = self.id_map[x][y][obj_type_id]
            in_id_map = False
            if objects:
                in_id_map = True

        # Check for conflicts.
        if in_grid != in_id_map:
            raise ValueError(
                "Conflict between ``self.grid`` and ``self.id_map``: ``self.grid`` returns '"
                + str(in_grid)
                + "' when asked if an object of type '"
                + self.obj_type_names[obj_type_id]
                + "' is at grid position '(%d, %d)', while ``self.id_map`` returns '"
                % (x, y)
                + str(in_id_map)
                + "'."
            )

        return in_grid

    def _plant(self) -> None:
        """
        Plant k new foods in the grid, where k is Gaussian.

        Updates
        -------
        self._place() : ``Callable``.
            Updates all variables updated by this function.
        self.num_foods : ``int``.
            Number of foods in the environment.
        """

        # Compute new food density with adaptive population control.
        if self.adaptive_food:
            agent_density = len(self.agents) / (self.width * self.height)
            delta_density = self.target_agent_density - agent_density
            self.food_regen_prob += delta_density / NORMALIZER
            self.food_regen_prob = max(self.food_regen_prob, 0.0)
            self.food_regen_prob = min(self.food_regen_prob, 1.0)

        # Sample whether or not to regenerate food for each square.
        regen_samples = np.random.rand(self.width, self.height)
        regen_locations = np.argwhere(regen_samples <= self.food_regen_prob)

        # Set new food positions.
        for food_pos in regen_locations:
            food_pos_tuple: Tuple[int, int] = tuple(food_pos)  # type: ignore
            assert len(food_pos_tuple) == 2
            if not self._obj_exists(self.obj_type_ids["food"], food_pos_tuple):
                self._place(self.obj_type_ids["food"], food_pos_tuple)
                self.num_foods += 1

    def _move(
        self, action_dict: Dict[int, Tuple[int, int, int]]
    ) -> Dict[int, Tuple[int, int, int]]:
        """
        Moves agents according to the move subactions in ``action_dict``. Checks for
        conflicts before moving in order to avoid collisions. Updates ``action_dict``
        with the actual, conflict-free actions taken by each agent.

        Parameters
        ----------
        action_dict : ``Dict[int, Tuple[int, int, int]]``.
            Maps agent ids to tuples of integer subactions.

        Updates
        -------
        self.agents : ``Dict[int, Agent]``.
            Map from agent ids to ``Agent`` objects.
        self._remove() : ``Callable``.
            All variables updated by this function call.
        self._place() : ``Callable``.
            All variables updated by this function call.

        Returns
        -------
        action_dict : ``Dict[int, Tuple[int, int, int]]``.
            Maps agent ids to tuples of integer subactions.
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

            if out_of_bounds or self._obj_exists(self.obj_type_ids["agent"], new_pos):
                action_dict[agent_id] = (self.STAY, consume, mate)
            else:
                self._remove(self.obj_type_ids["agent"], pos, agent_id)
                self._place(self.obj_type_ids["agent"], new_pos, agent_id)
                agent.pos = new_pos

        return action_dict

    def _consume(self, action_dict: Dict[int, Tuple[int, int, int]]) -> None:
        """
        Takes as input a collision-free ``action_dict`` and
        executes the ``consume`` action for all agents.

        Parameters
        ----------
        action_dict : ``Dict[int, Tuple[int, int, int]]``.
            Maps agent ids to tuples of integer subactions.

        Updates
        -------
        self.agents : ``Dict[int, Agent]``.
            Map from agent ids to ``Agent`` objects.
        self._remove() : ``Callable``.
            All variables updated by this function call.
        self.num_foods : ``int``.
            Number of foods in the environment.
        """
        for agent_id, action in action_dict.items():

            agent = self.agents[agent_id]
            pos = agent.pos

            # If the agent is dead, don't do anything
            if agent.health <= 0.0:
                continue

            # If they try to eat when there's nothing there, do nothing.
            _, consume, _ = action
            if self._obj_exists(self.obj_type_ids["food"], pos) and consume == self.EAT:
                self._remove(self.obj_type_ids["food"], pos)
                self.num_foods -= 1
                food_size = np.random.normal(self.food_size_mean, self.food_size_stddev)
                food_size = max(0, food_size)
                agent.health = min(1, agent.health + food_size)

    def _mate(self, action_dict: Dict[int, Tuple[int, int, int]]) -> Set[int]:
        """
        Takes as input a collision-free ``action_dict`` and
        executes the ``mate`` action for all agents.
        Returns a set of the ids of the newly created children.

        Parameters
        ----------
        action_dict : ``Dict[int, Tuple[int, int, int]]``.
            Maps agent ids to tuples of integer subactions.

        Updates
        -------
        self.agents : ``Dict[int, Agent]``.
            Map from agent ids to ``Agent`` objects.
        self._place() : ``Callable``.
            All variables updated by this function call.

        Returns
        -------
        child_ids : ``Set[int]``.
            Ids of the newly created child agents.
        """
        child_ids: Set[int] = set()

        # HARDCODE
        wants = lambda x: x == self.MATE
        wants_child = {
            agent_id: wants(action[2]) for agent_id, action in action_dict.items()
        }
        for mom_id in action_dict:
            mom = self.agents[mom_id]
            pos = mom.pos

            # If the agent is dead, don't do anything.
            if mom.health <= 0.0:
                continue

            # Grab action, do nothing if the agent chose not to mate.
            if not wants_child[mom_id]:
                continue

            # If the agent is not mature, do nothing.
            if not mom.is_mature:
                continue

            # Search adjacent positions for possible mates and find mate.
            adj_positions = self._get_adj_positions(pos)
            next_to_agent = False
            for adj_pos in adj_positions:
                if self._obj_exists(self.obj_type_ids["agent"], adj_pos):
                    mate_pos = adj_pos
                    x = mate_pos[0]
                    y = mate_pos[1]

                    # Get copy of set of agent ids to retrieve ``dad_id``.
                    agent_id_set = self.id_map[x][y][self.obj_type_ids["agent"]].copy()
                    dad_id = agent_id_set.pop()
                    dad = self.agents[dad_id]

                    # Otherwise, continue looking for mate
                    if dad_id not in child_ids and wants_child[dad_id]:
                        next_to_agent = True
                        break

            # If there is another agent in an adjacent position, spawn child.
            if next_to_agent:

                # Check ``mom`` and ``dad`` cooldown.
                if mom.mating_cooldown > 0 or dad.mating_cooldown > 0:
                    continue

                # Check ``dad`` health and maturity.
                if dad.health <= 0.0 or not dad.is_mature:
                    continue

                # Choose child location.
                open_positions = self._get_adj_positions(pos)
                open_positions += self._get_adj_positions(mate_pos)
                open_positions = list(set(open_positions))
                open_positions = [
                    open_pos
                    for open_pos in open_positions
                    if not self._obj_exists(self.obj_type_ids["agent"], open_pos)
                ]

                # Only create a new child if there are valid open positions.
                if open_positions != []:
                    child_pos = random.choice(open_positions)

                    # Update ``mating_cooldown`` for ``mom`` and ``dad``.
                    mom.mating_cooldown = mom.mating_cooldown_len
                    dad.mating_cooldown = dad.mating_cooldown_len

                    # Increment total child counters.
                    mom.num_children += 1
                    dad.num_children += 1

                    # Crossover and mutate parent DNA.
                    reward_weights, reward_biases = get_child_reward_network(
                        mom, dad, self.mut_sigma, self.mut_p
                    )

                    # Place child and add to ``self.grid``.
                    # child_health = min(dad.health, mom.health)
                    child_health = (dad.health + mom.health) / 2
                    child = Agent(
                        config=self.config,
                        num_actions=self.num_actions,
                        pos=child_pos,
                        initial_health=child_health,
                        reward_weights=reward_weights,
                        reward_biases=reward_biases,
                    )
                    child_id = self._new_agent_id()
                    self.agents[child_id] = child
                    child_ids.add(child_id)

                    self._place(self.obj_type_ids["agent"], child_pos, child_id)
                    wants_child[mom_id] = False
                    wants_child[dad_id] = False

        return child_ids

    # TODO: Remove from this class so that ``Env`` is framework-agnostic.
    def get_optimal_action_dists(
        self, greedy_temperature: float
    ) -> Dict[int, torch.Tensor]:
        """
        Iterates over the action space and compute the optimal action distribution for
        each agent.

        Parameters
        ----------
        greedy_temperature : ``float``.
            Greedy temperature for computation of optimal action distributions. As the
            value of this variable goes to zero, the optimal distribution gets more
            greedy. This value should be between 0 and 1.

        Returns
        -------
        optimal_action_dists : ``Dict[int, torch.Tensor]``.
            A mapping from ``agent_id`` to the optimal action distribution of that
            agent.
        """

        optimal_action_dists: Dict[int, torch.Tensor] = {}

        for agent_id, agent in self.agents.items():
            action_rewards = torch.zeros((self.num_actions,))
            for action in range(self.num_actions):
                action_rewards[action] = agent.compute_reward(action)

            # Flatten action_rewards, perform softmax, and return to original shape.
            # This is because torch.nn.functional.softmax only computes softmax along
            # a single dimension.
            action_rewards = torch.reshape(action_rewards, (-1,))
            optimal_action_dists[agent_id] = F.softmax(
                action_rewards / greedy_temperature, dim=0
            )
            action_rewards = torch.reshape(action_rewards, (self.num_actions,))
            optimal_action_dists[agent_id] = torch.reshape(
                optimal_action_dists[agent_id], (self.num_actions,)
            )

        return optimal_action_dists

    def _get_obs(self, pos: Tuple[int, int]) -> np.ndarray:
        """
        Returns an observation given an agent ``pos``.

        Parameters
        ----------
        pos : ``Tuple[int, int]``.
            A grid position.

        Returns
        -------
        agent_obs : ``np.ndarray``.
            The observation from the given position.
            Shape: ``(num_obj_types, obs_len, obs_len)``.
        """

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
        agent_obs = np.zeros((obs_len, obs_len, self.num_obj_types))
        pad_x_len = obs_len - pad_left - pad_right
        pad_y_len = obs_len - pad_top - pad_bottom
        agent_obs[
            pad_left : pad_left + pad_x_len, pad_bottom : pad_bottom + pad_y_len
        ] = self.grid[sight_left : sight_right + 1, sight_bottom : sight_top + 1]

        # Policy network expects number of channels in first dimension.
        agent_obs = np.swapaxes(agent_obs, 0, 2)

        return agent_obs

    def step(
        self, action_dict: Dict[int, int]
    ) -> Tuple[
        Dict[int, np.ndarray], Dict[int, float], Dict[Any, bool], Dict[int, Any]
    ]:
        """
        ``action_dict`` has agent indices as keys and a dict of the form
        ``{"move": <move>, "consume": <consume>)`` where the dict values
        are strings from the sets
            ``movements = set(["up", "down", "left", "right", "stay"])``
            ``consumptions = set(["eat", "noeat"])``.

        Parameters
        ----------
        action_dict : ``Dict[int, Tuple[int, int, int]]``.
            Maps agent ids to actions as multibinary numpy arrays.

        Returns
        -------
        obs : ``Dict[int, np.ndarray]``.
            Maps agent ids to observations.
        rew : ``Dict[int, float]``.
            Maps agent ids to rewards.
        done : ``Dict[int, bool]``.
            Maps agent ids to done status.
        info : ``Dict[int, Any]``.
            Maps agent ids to various per-agent info.
        """

        # Convert flat action_dict to tuple.
        tuple_action_dict: Dict[int, Tuple[int, int, int]] = {
            agent_id: flat_action_to_tuple(action, self.subaction_sizes)  # type: ignore
            for agent_id, action in action_dict.items()
        }

        # Execute move, consume, and mate actions, and calculate reward
        obs: Dict[int, np.ndarray] = {}
        rew: Dict[int, float] = {}
        done: Dict[int, bool] = {}
        info: Dict[int, Any] = {}

        # Set previous health values.
        for agent_id, agent in self.agents.items():
            agent.prev_health = agent.health

        # Execute actions (move, consume, and mate).
        tuple_action_dict = self._move(tuple_action_dict)
        self._consume(tuple_action_dict)
        child_ids = self._mate(tuple_action_dict)

        # Initialize ``info`` dicts. This must happen after the actions are
        # executed so that agents which are born from the call to _mate() are included
        # as keys in ``info``.
        for agent_id in self.agents:
            info[agent_id] = {}

        # Plant new food.
        self._plant()

        # Compute reward.
        for agent_id, agent in self.agents.items():
            if agent_id not in child_ids:
                # Note that ``compute_reward`` takes the action in integer form, so we
                # use ``action_dict`` here instead of ``tuple_action_dict``.
                rew[agent_id] = agent.compute_reward(action_dict[agent_id])
            # First reward for children is zero.
            elif agent_id in child_ids:
                rew[agent_id] = 0

        # Compute optimal action distribution for each agent for this timestep.
        # This "+1" is here because we don't increment ``self.iteration`` until
        # the end of this function, and the policy score is computed in trainer.py
        # after this increment happens.
        if (self.iteration + 1) % self.policy_score_frequency == 0:
            optimal_action_dists = self.get_optimal_action_dists(
                greedy_temperature=self.greedy_temperature
            )
            for agent_id in self.agents:
                info[agent_id]["optimal_action_dist"] = optimal_action_dists[agent_id]

        # Decrease agent health, compute observations and dones.
        killed_agent_ids = []
        for agent_id, agent in self.agents.items():
            if self.aging_type == "linear":
                agent.health -= self.aging_rate
            elif self.aging_type == "quadratic":
                agent.health -= self.aging_rate * agent.age
            else:
                raise NotImplementedError

            # Update mating cooldown.
            agent.mating_cooldown = max(0, agent.mating_cooldown - 1)

            obs[agent_id] = self._get_obs(agent.pos)
            agent.observation = obs[agent_id]

            done[agent_id] = agent.health <= 0.0

            # Kill agent if ``done[agent_id]`` and remove from ``self.grid``.
            if done[agent_id]:

                self._remove(self.obj_type_ids["agent"], agent.pos, agent_id)
                agent.pos = self.HEAVEN
                killed_agent_ids.append(agent_id)

                # Update average agent lifetime
                if self.avg_agent_lifetime < -1.0:
                    self.avg_agent_lifetime = agent.age
                else:
                    self.avg_agent_lifetime = (
                        ALPHA * self.avg_agent_lifetime + (1 - ALPHA) * agent.age
                    )

            # Update agent ages and update info dictionary
            agent.age += 1
            info[agent_id]["age"] = agent.age

        # Remove killed agents from self.agents
        for killed_agent_id in killed_agent_ids:
            self.agents.pop(killed_agent_id)

        self.dones = dict(done)

        if self.increment:
            self.iteration += 1

        return obs, rew, done, info

    def _get_adj_positions(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Returns the positions adjacent (left, right, up, and down) to a given position.

        Parameters
        ----------
        pos : ``Tuple[int, int]``.
            A grid position.

        Returns
        -------
        validated_adj_positions : ``List[Tuple[int, int]]``.
            List of the adjacent positions in the grid to ``pos``.
        """

        x, y = pos
        adj_positions = [
            (x + dX, y + dY) for dX, dY in [(0, 1), (0, -1), (1, 0), (-1, 0)]
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

        Returns
        -------
        output : ``str``.
            Formatted config file and state variable values.
        """
        output = pformat(self.config)
        return output

    def visual(self) -> str:
        """
        Returns a representation of the environment state.

        Returns
        -------
        output : ``str``.
            ASCII image of grid along with various statistics and metrics.
        """
        output = "\n"

        # Print grid.
        for y in range(self.height):
            for x in range(self.width):

                pos = (x, y)
                object_id = "_"

                # Check if there is an agent in ``pos``.
                if self._obj_exists(self.obj_type_ids["agent"], pos):
                    object_id = "B"
                # NOTE: ``B`` currently overwrites ``*``.
                # Check if there is a food in ``pos``.
                elif self._obj_exists(self.obj_type_ids["food"], pos):
                    object_id = "*"

                output += object_id + " "

            output += "\n"

        output += "\n"
        output += "===LOGPLAY_ANCHOR===\n\n"

        # Print agent stats.
        if PRINT_AGENT_STATS:
            num_agents_printed = 0
            num_living_agents = 0
            for agent_id, agent in self.agents.items():
                if agent.health > 0.0:
                    num_living_agents += 1
                if (
                    agent.health > 0.0
                    and num_agents_printed < self.config.num_displayed_agents
                ):
                    output += "Agent %d: " % agent_id
                    output += agent.__repr__()
                    num_agents_printed += 1
            output += "Num living agents: %d.\n" % num_living_agents
            output += "Num foods: %d.\n" % self.num_foods
            output += "\n"

        output += "Step: %d\n" % self.iteration

        return output

    def log_state(self, env_log: TextIO, visual_log: TextIO) -> None:
        """
        Logs the state of the environment as a string to a
        prespecified log file path.
        """

        # Write to json log for environment state.
        env_state = self._env_json_state()
        env_log.write(str(env_state) + "\n")

        # Write to visual log, and print visualization if settings["env"]["print"].
        visual = self.visual()
        if self.print_repr:
            if PRINT_FAST:
                print(chr(27) + "[2J")
            else:
                os.system("clear")
            print(visual)
        visual_log.write(visual + 40 * "\n")

    def _new_agent_id(self) -> int:
        """
        Grabs a new unique agent id.

        Updates
        -------
        self.agent_ids_created : ``int``.
            The number of agent ids created so far in this episode.

        Returns
        -------
        new_id : ``int``.
            A new unique agent id.
        """
        new_id = self.agent_ids_created
        self.agent_ids_created += 1

        return new_id

    def _env_json_state(self) -> Dict[str, Any]:
        """
        Returns a state of the environment as a json-style dictionary.
        """

        state: Dict[str, Any] = {}
        state["iteration"] = self.iteration
        state["agents"] = {
            agent_id: agent.agent_state() for agent_id, agent in self.agents.items()
        }
        state["num_foods"] = self.num_foods
        state["avg_agent_lifetime"] = self.avg_agent_lifetime

        return state

    def _env_state(self) -> Dict[str, Any]:
        """
        Returns a state of the environment as a dictionary.
        """

        state = self._env_json_state()
        env_attrs = ["grid", "id_map", "agent_ids_created", "action_space"]
        for env_attr in env_attrs:
            state[env_attr] = getattr(self, env_attr)
        agent_attrs = ["reward_weights", "reward_biases"]
        for agent_id in state["agents"]:
            for agent_attr in agent_attrs:
                state["agents"][agent_id][agent_attr] = getattr(
                    self.agents[agent_id], agent_attr
                )

        return state

    def save(self, save_path: str) -> None:
        """
        Saves a .pkl representation of the environment state.

        Parameters
        ----------
        save_path : str.
            Path to save environment pickle object.
        """

        state = self._env_state()
        with open(save_path, "wb") as f:
            pickle.dump(state, f)

    def load(self, load_path: str) -> None:
        """
        Loads a .pkl representation (from self.save()) of the environment state.

        Parameters
        ----------
        load_path : str.
            Path to load environment pickle object from.
        """

        with open(load_path, "rb") as f:
            state = pickle.load(f)

        # Get environment attributes
        env_attrs = [
            "iteration",
            "num_foods",
            "avg_agent_lifetime",
            "grid",
            "id_map",
            "agent_ids_created",
        ]
        for env_attr in env_attrs:
            setattr(self, env_attr, state[env_attr])

        # Construct agents
        self.agents = {}
        for agent_id, agent_state in state["agents"].items():
            self.agents[agent_id] = Agent(
                config=self.config,
                num_actions=self.num_actions,
                pos=agent_state["pos"],
                initial_health=agent_state["initial_health"],
                reward_weights=agent_state["reward_weights"],
                reward_biases=agent_state["reward_biases"],
            )
            for attr, value in agent_state.items():
                setattr(self.agents[agent_id], attr, value)

        # Construct agent observations
        for agent_id, agent in self.agents.items():
            agent.observation = self._get_obs(agent.pos)

        print("loaded!")
