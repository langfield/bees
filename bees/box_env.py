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

# Bees imports.
from agent import Agent
from genetics import get_child_reward_network
from utils import get_logs

# Set global log file
REPR_LOG, REW_LOG = get_logs()

# Settings for ``__repr__()``.
PRINT_AGENT_STATS = True
PRINT_DONES = True

# pylint: disable=bad-continuation

class Env:
    """
    Environment with bees in it.

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
    food_density : ``float``.
        The initial proportion of food in the grid.
    food_size_mean : ``float``.
        The mean of the Gaussian from which food size is sampled.
    food_size_stddev : ``float``.
        The standard deviation of the Gaussian from which food size is sampled.
    plant_foods_mean : ``float``.
        The mean of the Gaussian from which the number of foods to add to the
        grid at each timestep is sampled.
    plant_foods_stddev : ``float``.
        The standard deviation of the Gaussian from which the number of foods
        to add to the grid at each timestep is sampled.
    food_plant_retries : ``int``.
        How many times to attempt to add a food item to the environment given
        a conflict before giving up.
    mating_cooldown_len : ``int``.
        How long agents must wait in between mate actions.
    min_mating_health : ``float``.
        The minimum health bar value at which agents can mate.
    agent_init_x_upper_bound : ``int``.
        The right bound on the top left grid section in which the first agents
        are initialized.
    agent_init_y_upper_bound : ``int``.
        The bottom bound on the top left grid section in which the first agents
        are initialized.
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

    # TODO: make optional arguments consistent throughout nested init calls.

    def __init__(
        self,
        width: int,
        height: int,
        sight_len: int,
        num_obj_types: int,
        num_agents: int,
        aging_rate: float,
        food_density: float,
        food_size_mean: float,
        food_size_stddev: float,
        plant_foods_mean: float,
        plant_foods_stddev: float,
        food_plant_retries: int,
        mating_cooldown_len: int,
        min_mating_health: float,
        agent_init_x_upper_bound: int,
        agent_init_y_upper_bound: int,
        n_layers: int,
        hidden_dim: int,
        reward_weight_mean: float,
        reward_weight_stddev: float,
        mut_sigma: float,
        mut_p: float,
        consts: Dict[str, Any],
    ) -> None:

        self.width = width
        self.height = height
        self.sight_len = sight_len
        self.num_obj_types = num_obj_types
        self.num_agents = num_agents
        self.aging_rate = aging_rate
        self.food_density = food_density
        self.food_size_mean = food_size_mean
        self.food_size_stddev = food_size_stddev
        self.plant_foods_mean = plant_foods_mean
        self.plant_foods_stddev = plant_foods_stddev
        self.food_plant_retries = food_plant_retries
        self.mating_cooldown_len = mating_cooldown_len
        self.min_mating_health = min_mating_health
        self.agent_init_x_upper_bound = agent_init_x_upper_bound
        self.agent_init_y_upper_bound = agent_init_y_upper_bound

        # HARDCODE
        self.agent_init_x_upper_bound = min(
            math.ceil(2 * math.sqrt(num_agents)), self.width
        )
        self.agent_init_x_upper_bound = min(
            math.ceil(2 * math.sqrt(num_agents)), self.height
        )

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.reward_weight_mean = reward_weight_mean
        self.reward_weight_stddev = reward_weight_stddev

        self.mut_sigma = mut_sigma
        self.mut_p = mut_p

        # pylint: disable=invalid-name
        # Get constants.
        self.consts = consts
        self.LEFT = consts["LEFT"]
        self.RIGHT = consts["RIGHT"]
        self.UP = consts["UP"]
        self.DOWN = consts["DOWN"]
        self.STAY = consts["STAY"]
        self.EAT = consts["EAT"]
        self.NO_EAT = consts["NO_EAT"]
        self.MATE = consts["MATE"]
        self.NO_MATE = consts["NO_MATE"]
        self.HEAVEN: Tuple[int, int] = tuple(consts["BEE_HEAVEN"])  # type: ignore

        # Construct object identifier dictionary.
        self.obj_type_ids = {"agent": 0, "food": 1}
        self.obj_type_name = {0: "agent", 1: "food"}

        # Compute number of foods.
        num_squares = self.width * self.height
        self.initial_num_foods = math.floor(self.food_density * num_squares)
        self.num_foods = 0

        # Construct ``self.grid``.
        self.grid = np.zeros((self.width, self.height, self.num_obj_types))
        self.id_map: List[List[Dict[int, Set[int]]]] = [
            [{} for y in range(self.height)] for x in range(self.width)
        ]

        # Construct observation and action spaces.
        # HARDCODE
        self.action_space = gym.spaces.Tuple(
            (gym.spaces.Discrete(5), gym.spaces.Discrete(2), gym.spaces.Discrete(2))
        )
        # ===MOD===
        # Action space is a MultiBinary space, where the toggles are:
        # Stay/Move, Left/Right, Up/Down, Eat/NoEat, Mate,NoMate.
        self.action_space = gym.spaces.MultiBinary(5)
        # ===MOD===

        num_actions = 5 + 2 + 2
        self.num_actions = num_actions

        # Each observation is a k * k matrix with values from a discrete
        # space of size self.num_obj_types, where k = 2 * self.sight_len + 1
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

        # ===MOD===
        obs_len = 2 * self.sight_len + 1
        low_obs = np.zeros((self.num_obj_types, obs_len, obs_len))
        high_obs = np.zeros((self.num_obj_types, obs_len, obs_len))
        self.observation_space = gym.spaces.Box(low_obs, high_obs)
        # ===MOD===

        self.agents: Dict[int, Agent] = {}
        self.agent_ids_created = 0

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

        # Reset ``self.grid``.
        self.grid = np.zeros((self.width, self.height, self.num_obj_types))
        self.id_map = [[{} for y in range(self.height)] for x in range(self.width)]
        # MOD
        self.num_foods = 0

        # Set unique agent positions.
        grid_positions = list(itertools.product(range(self.height), range(self.width)))

        # HARDCODE: agents bounded in bottom left corner upon initialization.
        x_bound = self.agent_init_x_upper_bound
        y_bound = self.agent_init_y_upper_bound
        grid_corner = [(x, y) for x, y in grid_positions if x < x_bound and y < y_bound]

        agent_positions = random.sample(grid_corner, self.num_agents)
        for i, (agent_id, agent) in enumerate(self.agents.items()):
            agent_pos = agent_positions[i]
            REPR_LOG.write(
                "Initializing agent '%d' at '%s'.\n" % (agent_id, str(agent_pos))
            )
            self._place(self.obj_type_ids["agent"], agent_pos, agent_id)
            agent.pos = agent_pos

        # Set unique food positions.
        assert self.num_foods == 0
        food_positions = random.sample(grid_positions, self.initial_num_foods)
        for food_pos in food_positions:
            self._place(self.obj_type_ids["food"], food_pos)
        self.num_foods = self.initial_num_foods

    def reset(self) -> Dict[int, Tuple[Tuple[Tuple[int, ...], ...], ...]]:
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
        # TODO: reconcile ``self.iteration`` and ``steps_completed``.

        # Get average rewards for agents from previous episode.
        if self.agents != []:
            avg_reward = np.mean(
                [agent.total_reward for _, agent in self.agents.items()]
            )
            REPR_LOG.write("{:.10f}".format(avg_reward) + "\n")

        # MOD
        # Reconstruct agents.
        self.agents = {}
        self.agent_ids_created = 0
        for _ in range(self.num_agents):
            self.agents[self._new_agent_id()] = Agent(
                sight_len=self.sight_len,
                num_obj_types=self.num_obj_types,
                consts=self.consts,
                n_layers=self.n_layers,
                hidden_dim=self.hidden_dim,
                num_actions=self.num_actions,
                reward_weight_mean=self.reward_weight_mean,
                reward_weight_stddev=self.reward_weight_stddev,
                mating_cooldown_len=self.mating_cooldown_len,
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
            REPR_LOG.close()
            raise ValueError("'%s' is not a valid action.")

        return new_pos  # type: ignore

    def _remove(
        self, obj_type_id: int, pos: Tuple[int, int], obj_id: int = None
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
            REPR_LOG.close()
            raise ValueError(
                "Object '%s' does not exist at grid position '(%d, %d)'."
                % (self.obj_type_name[obj_type_id], x, y)
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
                    % (self.obj_type_name[obj_type_id], obj_id, x, y)
                )
            self.id_map[x][y][obj_type_id].remove(obj_id)

    def _place(
        self, obj_type_id: int, pos: Tuple[int, int], obj_id: int = None
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
            REPR_LOG.close()
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
                    "Object of type '%s' with " % self.obj_type_name[obj_type_id]
                    + "identifier '%d' cannot be placed at grid position " % obj_id
                    + "'(%d, %d)' since an object of the same type with the " % (x, y)
                    + "same id already exists there."
                )
            self.id_map[x][y][obj_type_id].add(obj_id)

    def _obj_exists(self, obj_type_id: int, pos: Tuple[int, int]) -> bool:
        """
        Check if an object of object type ``obj_typ_id`` exists at the given position.

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
        """

        # Check grid.
        grid_idx = pos + (obj_type_id,)
        in_grid = self.grid[grid_idx] == 1

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
                + self.obj_type_name[obj_type_id]
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

        # Generate and validate the number of foods to plant.
        food_ev = np.random.normal(self.plant_foods_mean, self.plant_foods_stddev)
        num_new_foods = round(food_ev)
        num_new_foods = max(0, num_new_foods)
        num_new_foods = min(self.height * self.width, num_new_foods)

        # Set new food positions.
        grid_positions = list(itertools.product(range(self.height), range(self.width)))
        food_positions = random.sample(grid_positions, num_new_foods)
        for food_pos in food_positions:
            if self._obj_exists(self.obj_type_ids["food"], food_pos):

                # Retry for an unoccupied position a fixed number of times.
                # HARDCODE
                for _ in range(self.food_plant_retries):
                    food_pos = random.sample(grid_positions, 1)[0]
                    if not self._obj_exists(self.obj_type_ids["food"], food_pos):
                        break

            # If unsuccessful, skip.
            if self._obj_exists(self.obj_type_ids["food"], food_pos):
                continue

            self._place(self.obj_type_ids["food"], food_pos)
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
        child_ids = set()
        for mom_id, action in action_dict.items():
            mom = self.agents[mom_id]
            pos = mom.pos

            # If the agent is dead, don't do anything.
            # MOD: agents can only reproduce with sufficient health.
            if mom.health <= self.min_mating_health:
                continue

            # Grab action, do nothing if the agent chose not to mate.
            _, _, mate = action
            if mate == self.NO_MATE:
                continue

            # Search adjacent positions for possible mates and find mate.
            adj_positions = self._get_adj_positions(pos)
            next_to_agent = False
            for adj_pos in adj_positions:
                if self._obj_exists(self.obj_type_ids["agent"], adj_pos):
                    next_to_agent = True
                    mate_pos = adj_pos
                    x = mate_pos[0]
                    y = mate_pos[1]

                    # Get copy of set of agent ids to retrieve ``dad_id``.
                    agent_id_set = self.id_map[x][y][self.obj_type_ids["agent"]].copy()
                    dad_id = agent_id_set.pop()
                    dad = self.agents[dad_id]
                    break

            # If there is another agent in an adjacent position, spawn child.
            if next_to_agent:

                # Check ``mom`` and ``dad`` cooldown.
                if mom.mating_cooldown > 0 or dad.mating_cooldown > 0:
                    continue

                # MOD
                if dad.health <= self.min_mating_health:
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

                    # Crossover and mutate parent DNA.
                    reward_weights, reward_biases = get_child_reward_network(
                        mom, dad, self.mut_sigma, self.mut_p
                    )

                    # Place child and add to ``self.grid``.
                    child = Agent(
                        sight_len=self.sight_len,
                        num_obj_types=self.num_obj_types,
                        consts=self.consts,
                        n_layers=self.n_layers,
                        hidden_dim=self.hidden_dim,
                        num_actions=self.num_actions,
                        pos=child_pos,
                        reward_weight_mean=self.reward_weight_mean,
                        reward_weight_stddev=self.reward_weight_stddev,
                        reward_weights=reward_weights,
                        reward_biases=reward_biases,
                        mating_cooldown_len=self.mating_cooldown_len,
                    )
                    child_id = self._new_agent_id()
                    self.agents[child_id] = child
                    REPR_LOG.write("Adding child with id '%d'.\n" % child_id)
                    child_ids.add(child_id)

                    REPR_LOG.write(
                        "Placing child '%d' at '%s'.\n" % (child_id, str(child_pos))
                    )
                    self._place(self.obj_type_ids["agent"], child_pos, child_id)

        return child_ids

    def _get_obs(self, pos: Tuple[int, int]) -> np.ndarray:
        """
        Returns an observation given an agent ``pos``.

        Parameters
        ----------
        pos : ``Tuple[int, int]``.
            A grid position.

        Returns
        -------
        obs : ``np.ndarray``.
            The observation from the given position.
            Shape: ``(obs_len, obs_len, num_obj_types)``.
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
        obs = np.zeros((obs_len, obs_len, self.num_obj_types))
        pad_x_len = obs_len - pad_left - pad_right
        pad_y_len = obs_len - pad_top - pad_bottom
        obs[
            pad_left : pad_left + pad_x_len, pad_bottom : pad_bottom + pad_y_len
        ] = self.grid[sight_left : sight_right + 1, sight_bottom : sight_top + 1]

        # ===MOD===
        obs = np.swapaxes(obs, 0, 2)
        # obs = convert_obs_to_tuple(obs)
        # ===MOD===

        return obs

    def get_action_dict(self) -> Dict[int, Tuple[int, int, int]]:
        """
        Constructs ``action_dict`` by querying individual agents for their actions
        based on their observations. Used only for dummy training runs.

        Returns
        -------
        action_dict : ``Dict[int, Tuple[int, int, int]]``.
            Maps agent ids to tuples of integer subactions.
        """
        action_dict = {}

        for agent_id, agent in self.agents.items():
            action_dict[agent_id] = agent.get_action()

        return action_dict

    def step(
        self, action_array_dict: Dict[int, np.ndarray]
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
        action_array_dict : ``Dict[int, np.ndarray]``.
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
        REPR_LOG.write("===STEP===\n")
        REPR_LOG.flush()

        # Action dict conversion.
        action_dict: Dict[int, Tuple[int, int, int]] = {}
        for agent_id, action_array in action_array_dict.items():
            integer_action_array = action_array.astype(int)
            stay_bit = integer_action_array[0]
            axis_bit = integer_action_array[1]
            direction_bit = integer_action_array[2]
            consume_bit = integer_action_array[3]
            mate_bit = integer_action_array[4]

            if stay_bit == 0:
                move = self.STAY
            elif axis_bit == 0:
                if direction_bit == 0:
                    move = self.LEFT
                else:
                    move = self.RIGHT
            else:
                if direction_bit == 0:
                    move = self.DOWN
                else:
                    move = self.UP

            if consume_bit == 0:
                consume = self.EAT
            else:
                consume = self.NO_EAT

            if mate_bit == 0:
                mate = self.MATE
            else:
                mate = self.NO_MATE

            action_dict[agent_id] = (move, consume, mate)

        # Execute move, consume, and mate actions, and calculate reward
        obs: Dict[int, np.ndarray] = {}
        rew: Dict[int, float] = {}
        done: Dict[int, bool] = {}
        info: Dict[int, Any] = {}

        # Execute actions (move, consume, and mate).
        prev_health = {
            agent_id: agent.health for agent_id, agent in self.agents.items()
        }
        action_dict = self._move(action_dict)
        self._consume(action_dict)
        child_ids = self._mate(action_dict)

        # Plant new food.
        self._plant()

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
            # REPR_LOG.write("Agent '%d' health: '%f'.\n" % (agent_id, agent.health))
            if agent.health > 0.0:
                agent.health -= self.aging_rate

                # Update mating cooldown.
                agent.mating_cooldown = max(0, agent.mating_cooldown - 1)

                obs[agent_id] = self._get_obs(agent.pos)
                agent.observation = obs[agent_id]

                # MOD: removed ``self.num_foods <= 0`` condition.
                done[agent_id] = agent.health <= 0.0

                # Kill agent if ``done[agent_id]`` and remove from ``self.grid``.
                if done[agent_id]:

                    REPR_LOG.write("Killing agent '%d'.\n" % agent_id)
                    self._remove(self.obj_type_ids["agent"], agent.pos, agent_id)
                    agent.pos = self.HEAVEN
                    killed_agent_ids.append(agent_id)

                # Update agent ages and update info dictionary
                agent.age += 1
                info[agent_id] = {"age": agent.age}

        # Remove killed agents from self.agents
        for killed_agent_id in killed_agent_ids:
            self.agents.pop(killed_agent_id)

        self.dones = dict(done)

        # Write environment representation to log
        self._log_state()

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
            # HARDCODE
            agent_print_bound = 20
            agents_printed = 0
            num_living_agents = 0
            for agent_id, agent in self.agents.items():
                if agent.health > 0.0:
                    num_living_agents += 1
                if agent.health > 0.0 and agents_printed < agent_print_bound:
                    output += "Agent %d: " % agent_id
                    output += agent.__repr__()
                    agents_printed += 1
            output += "Num living agents: %d.\n" % num_living_agents
            output += "Num foods: %d.\n" % self.num_foods
            output += "\n"

        # Print dones.
        if PRINT_DONES:
            output += "Dones: " + str(self.dones) + "\n"

        # Push past next window size.
        # HARDCODE
        """
        for _ in range(40):
            output += "\n"
        """

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
