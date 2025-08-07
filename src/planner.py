import random
import numpy as np
import math
import copy
from helper import uav_position


class MCTSState:
    """Lightweight state representation for MCTS planning."""
    
    def __init__(self, uav_pos, belief_summary, remaining_steps):
        """
        Initialize MCTS state.
        
        Args:
            uav_pos: uav_position object representing UAV location and altitude
            belief_summary: Compressed representation of belief map
            remaining_steps: Number of planning steps remaining
        """
        self.uav_pos = uav_pos
        self.belief_summary = belief_summary
        self.remaining_steps = remaining_steps
    
    def is_terminal(self):
        """Check if this is a terminal state."""
        return self.remaining_steps <= 0
    
    def __hash__(self):
        """Make state hashable for use in dictionaries."""
        return hash((self.uav_pos, tuple(self.belief_summary.flatten()), self.remaining_steps))
    
    def __eq__(self, other):
        """Check equality of states."""
        if not isinstance(other, MCTSState):
            return False
        return (self.uav_pos == other.uav_pos and 
                np.array_equal(self.belief_summary, other.belief_summary) and
                self.remaining_steps == other.remaining_steps)


class MCTSNode:
    """Node in the MCTS tree."""
    
    def __init__(self, state, parent=None, action=None):
        """
        Initialize MCTS node.
        
        Args:
            state: MCTSState object
            parent: Parent MCTSNode (None for root)
            action: Action that led to this state from parent
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}  # action -> MCTSNode
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = None  # Will be set when expanded
    
    def is_fully_expanded(self):
        """Check if all actions have been tried from this node."""
        return self.untried_actions is not None and len(self.untried_actions) == 0
    
    def is_leaf(self):
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0
    
    def ucb1(self, exploration_param=math.sqrt(2)):
        """Calculate UCB1 value for action selection."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.total_reward / self.visits
        exploration = exploration_param * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def select_child(self, exploration_param=math.sqrt(2)):
        """Select child with highest UCB1 value."""
        return max(self.children.values(), key=lambda child: child.ucb1(exploration_param))
    
    def add_child(self, action, state):
        """Add a new child node."""
        child = MCTSNode(state, parent=self, action=action)
        self.children[action] = child
        return child
    
    def update(self, reward):
        """Update node statistics with reward from simulation."""
        self.visits += 1
        self.total_reward += reward
    
    def backup(self, reward):
        """Backpropagate reward up the tree."""
        node = self
        while node is not None:
            node.update(reward)
            node = node.parent


class MCTS:
    """Monte Carlo Tree Search implementation for UAV planning."""
    
    def __init__(self, planner, max_depth=10, iterations=1000, exploration_param=math.sqrt(2)):
        """
        Initialize MCTS.
        
        Args:
            planner: Planning object with UAV and environment information
            max_depth: Maximum planning depth (number of steps to look ahead)
            iterations: Number of MCTS iterations to run
            exploration_param: UCB1 exploration parameter
        """
        self.planner = planner
        self.max_depth = max_depth
        self.iterations = iterations
        self.exploration_param = exploration_param
    
    def compress_belief_map(self, belief_map):
        """
        Compress the full belief map into a lightweight summary.
        
        Args:
            belief_map: Full belief map of shape (rows, cols, 2)
            
        Returns:
            Compressed representation suitable for state comparison
        """
        # Use more aggressive spatial downsampling for better performance
        # For large grids, sample even more sparsely
        rows, cols = belief_map.shape[:2]
        if rows > 200 or cols > 200:
            step = 8  # More aggressive for large grids
        else:
            step = 4  # Standard for smaller grids
        
        compressed = belief_map[::step, ::step, 1]  # Only keep P(m=1)
        return compressed
    
    def get_permitted_actions(self, state):
        """Get permitted actions from the given state."""
        # Temporarily set UAV to the state position
        old_pos = self.planner.uav.position
        old_alt = self.planner.uav.altitude
        
        self.planner.uav.set_position(state.uav_pos.position)
        self.planner.uav.set_altitude(state.uav_pos.altitude)
        
        actions = self.planner.uav.permitted_actions(self.planner.uav)
        
        # Restore original UAV state
        self.planner.uav.set_position(old_pos)
        self.planner.uav.set_altitude(old_alt)
        
        return actions
    
    def simulate_action(self, state, action):
        """
        Simulate taking an action from the given state.
        
        Args:
            state: Current MCTSState
            action: Action to take
            
        Returns:
            New MCTSState after taking the action
        """
        # Temporarily set UAV to the state position
        old_pos = self.planner.uav.position
        old_alt = self.planner.uav.altitude
        
        self.planner.uav.set_position(state.uav_pos.position)
        self.planner.uav.set_altitude(state.uav_pos.altitude)
        
        # Get future UAV position
        future_pos_tuple = self.planner.uav.x_future(action)
        future_uav_pos = uav_position(future_pos_tuple)
        
        # For now, keep the same belief summary (would need sensor model for full simulation)
        new_belief_summary = state.belief_summary.copy()
        
        # Restore original UAV state
        self.planner.uav.set_position(old_pos)
        self.planner.uav.set_altitude(old_alt)
        
        return MCTSState(future_uav_pos, new_belief_summary, state.remaining_steps - 1)
    
    def calculate_reward(self, state, action):
        """
        Calculate the reward for taking an action from the given state.
        Uses the existing information gain calculation.
        """
        # Temporarily set UAV to the state position
        old_pos = self.planner.uav.position
        old_alt = self.planner.uav.altitude
        
        self.planner.uav.set_position(state.uav_pos.position)
        self.planner.uav.set_altitude(state.uav_pos.altitude)
        
        # Calculate information gain using existing method
        x_future = uav_position(self.planner.uav.x_future(action))
        
        # Get observable area for this future position
        [[obsd_m_i_min, obsd_m_i_max], [obsd_m_j_min, obsd_m_j_max]] = (
            self.planner.uav.get_range(
                position=x_future.position,
                altitude=x_future.altitude,
                index_form=True,
            )
        )
        
        # Get current belief map subset
        obs_M = self.planner.M[obsd_m_i_min:obsd_m_i_max, obsd_m_j_min:obsd_m_j_max, 1]
        reward = np.sum(self.planner.info_gain(obs_M, x_future))
        
        # Restore original UAV state
        self.planner.uav.set_position(old_pos)
        self.planner.uav.set_altitude(old_alt)
        
        return reward
    
    def selection(self, root):
        """Selection phase: traverse tree using UCB1 until leaf node."""
        node = root
        while not node.is_leaf() and not node.state.is_terminal():
            if not node.is_fully_expanded():
                return node  # Return for expansion
            node = node.select_child(self.exploration_param)
        return node
    
    def expansion(self, node):
        """Expansion phase: add a new child node for an untried action."""
        if node.state.is_terminal():
            return node
        
        if node.untried_actions is None:
            # First time expanding this node
            node.untried_actions = self.get_permitted_actions(node.state)
        
        if len(node.untried_actions) == 0:
            return node  # No more actions to try
        
        # Pick a random untried action
        action = node.untried_actions.pop(random.randint(0, len(node.untried_actions) - 1))
        
        # Create new state by simulating the action
        new_state = self.simulate_action(node.state, action)
        
        # Add child node
        child = node.add_child(action, new_state)
        return child
    
    def simulation(self, node):
        """Simulation phase: random rollout to estimate value."""
        current_state = node.state
        total_reward = 0.0
        steps = 0
        
        # Random rollout for remaining depth
        while not current_state.is_terminal() and steps < self.max_depth:
            actions = self.get_permitted_actions(current_state)
            if not actions:
                break
            
            action = random.choice(actions)
            reward = self.calculate_reward(current_state, action)
            total_reward += reward
            
            current_state = self.simulate_action(current_state, action)
            steps += 1
        
        return total_reward
    
    def backpropagation(self, node, reward):
        """Backpropagation phase: update statistics back to root."""
        node.backup(reward)
    
    def search(self, initial_state):
        """
        Run MCTS search from the initial state.
        
        Args:
            initial_state: MCTSState representing current situation
            
        Returns:
            Best action to take
        """
        root = MCTSNode(initial_state)
        
        for _ in range(self.iterations):
            # Selection
            leaf = self.selection(root)
            
            # Expansion
            expanded_node = self.expansion(leaf)
            
            # Simulation
            reward = self.simulation(expanded_node)
            
            # Backpropagation
            self.backpropagation(expanded_node, reward)
        
        # Select best action based on visit counts (most robust)
        if not root.children:
            # No children created, return random action
            actions = self.get_permitted_actions(initial_state)
            return random.choice(actions) if actions else "hover"
        
        best_action = max(root.children.keys(), 
                         key=lambda action: root.children[action].visits)
        return best_action


class planning:
    def __init__(self, grid_info, uav, strategy, conf_dict=None, optimal_alt=21.6, 
                 mcts_depth=10, mcts_iterations=1000, mcts_exploration=math.sqrt(2)):
        # Initialize belief map (each cell has a default probability of 0.5) and set UAV planning parameters
        self.M = np.full((grid_info.shape[0], grid_info.shape[1], 2), 0.5)
        self.uav = uav
        self.last_action = None
        self.strategy = strategy
        self.conf_dict = conf_dict
        self.optimal_altitude = optimal_alt
        self.sweep_direction = None
        
        # MCTS parameters - adjust defaults based on grid size
        grid_size = grid_info.shape[0] * grid_info.shape[1]
        if grid_size > 100000:  # Large grids (e.g., 400x400)
            default_depth = 8
            default_iterations = 500
        elif grid_size > 10000:  # Medium grids (e.g., 100x100)
            default_depth = 10
            default_iterations = 1000
        else:  # Small grids
            default_depth = 12
            default_iterations = 1500
            
        self.mcts_depth = mcts_depth if mcts_depth != 10 else default_depth
        self.mcts_iterations = mcts_iterations if mcts_iterations != 1000 else default_iterations
        self.mcts_exploration = mcts_exploration
        self.mcts = None  # Will be initialized when needed

    def reset(self, conf_dict=None):
        """Reset UAV and planning state, and reinitialize the belief map."""
        self.uav.reset()
        self.conf_dict = conf_dict
        self.last_action = None
        self.M = np.ones_like(self.M) * 0.5

    def info_gain(self, var, x_future):
        """Calculate information gain for a belief state given a future UAV state."""
        ig = self.H(var) - self._expected_entropy(var, x_future)

        return ig

    def H(self, var):
        """ "Compute binary entropy of a random variable (or belief map)."""

        assert not np.any(np.isnan(var)), f"NaN detected in var: {var}"
        var = np.clip(var, 0.0, 1.0)  # Clamps values to the range [0, 1]

        v1 = var
        v2 = 1.0 - var

        if isinstance(var, np.ndarray):
            v1 = np.where(v1 == 0.0, 1.0, v1)
            v2 = np.where(v2 == 0.0, 1.0, v2)
        else:
            if v1 == 0.0:
                v1 = 1.0
            if v2 == 0.0:
                v2 = 1.0

        l1 = np.log2(v1)
        l2 = np.log2(v2)

        assert np.all(np.less_equal(l1, 0.0))
        assert np.all(np.less_equal(l2, 0.0))

        entropy = -(v1 * l1 + v2 * l2)

        assert np.all(np.greater_equal(entropy, 0.0))

        return entropy

    def cH(self, var, sigma0, sigma1):
        """
        Compute the conditional entropy for a binary random variable using sensor model likelihoods.
        """

        # probability of the evidence
        # p(z = 0) = p(z = 0|m = 0)p(m = 0) + p(z = 0|m = 1)p(m = 1)
        sigma0 = np.clip(sigma0, 0.0, 1.0)
        sigma1 = np.clip(sigma1, 0.0, 1.0)
        a = (1.0 - sigma0) * (1.0 - var) + (sigma1 * var)  # p(z=0)
        # p(z = 1) = 1 - p(z = 0)
        b = 1.0 - a + 1e-6  # p(z=1) with stability epsilon

        assert np.all(np.greater_equal(var, 0.0)), f"{var[np.isnan(var)]}"
        assert np.all(np.less_equal(var, 1.0)), f"{var[np.isnan(var)]}"

        # posterior distribution probabilities
        # p(m = 1|z = 0) = (p(z = 0|m = 1)p(m = 1))/p(z = 0)
        p10 = (sigma1 * var) / a  # p(m=1|z=0)
        # p(m = 1|z = 1) = (p(z = 1|m = 1)p(m = 1))/p(z = 1)
        p11 = ((1.0 - sigma1) * var) / b  # p(m=1|z=1)

        assert np.all(np.greater_equal(np.round(p10, decimals=2), 0.0)) and np.all(
            np.less_equal(np.round(p10, decimals=2), 1.0)
        ), f"{p10}"
        assert np.all(np.greater_equal(p11, 0.0)) and np.all(
            np.less_equal(p11, 1.0)
        ), f"{sigma1}-{var[np.greater(p11, 1.0)]}-{b[np.greater(p11, 1.0)]}"

        # conditional entropy: average of the entropy of the posterior distribution probabilities
        # H(m|z) = p(z = 0)H(p(m = 1|z = 0)) + p(z = 1)H(p(m = 1|z = 1))
        cH = a * self.H(p10) + b * self.H(p11)

        assert np.all(np.greater_equal(cH, 0.0))

        return cH

    def _expected_entropy(self, var, x_future):
        """Compute expected entropy based on future UAV state and sensor model parameters."""
        a = 1
        b = 0.015
        sigma = a * (1 - np.exp(-b * x_future.altitude))

        if self.conf_dict is not None:
            s0, s1 = self.conf_dict[np.round(x_future.altitude, decimals=2)]
        else:
            s0, s1 = sigma, sigma

        return self.cH(var, s0, s1)

    def sweep(self, permitted_actions, visited_x):
        """Select a sweeping action based on UAV altitude and visited positions."""
        if (
            self.uav.get_x().altitude < self.optimal_altitude
            and "up" in permitted_actions
        ):
            self.last_action = "up"
            return "up", None

        sweep_actions = []
        for action in permitted_actions:
            x_future = uav_position(self.uav.x_future(action))
            if x_future not in visited_x and action != "up" and action != "down":
                sweep_actions.append(action)

        if self.sweep_direction is None:
            if len(sweep_actions) == 1:
                self.sweep_direction = (
                    "LeftRight"
                    if sweep_actions[0] in ["left", "right"]
                    else "BackFront"
                )
            else:
                self.sweep_direction = random.choice(["LeftRight", "BackFront"])

        # self.sweep_direction = "LeftRight"
        if self.sweep_direction == "LeftRight":
            # give propriority to left or right (if one of them is present in sweep_actions, only one can be present at a time)
            if "left" in sweep_actions:
                self.last_action = "left"
            elif "right" in sweep_actions:
                self.last_action = "right"
            elif "front" in sweep_actions:
                self.last_action = "front"
            elif "back" in sweep_actions:
                self.last_action = "back"
            else:
                self.last_action = "hover"
        if self.sweep_direction == "BackFront":
            # give propriority to back or front (if one of them is present in sweep_actions, only one can be present at a time)
            if "back" in sweep_actions:
                self.last_action = "back"
            elif "front" in sweep_actions:
                self.last_action = "front"
            elif "left" in sweep_actions:
                self.last_action = "left"
            elif "right" in sweep_actions:
                self.last_action = "right"
            else:
                self.last_action = "hover"
        return self.last_action, None

    def ig_based(self, permitted_actions):
        """Select an action based on the maximum information gain."""
        info_gain_action = {}
        for action in permitted_actions:
            # UAV position after taking action a
            x_future = uav_position(self.uav.x_future(action))
            info_gain_action_a = 0
            [[obsd_m_i_min, obsd_m_i_max], [obsd_m_j_min, obsd_m_j_max]] = (
                self.uav.get_range(
                    position=x_future.position,
                    altitude=x_future.altitude,
                    index_form=True,
                )
            )
            obs_M = self.M[obsd_m_i_min:obsd_m_i_max, obsd_m_j_min:obsd_m_j_max, 1]
            info_gain_action_a = np.sum(self.info_gain(obs_M, x_future))
            info_gain_action[action] = info_gain_action_a

        # Find the maximum information gain
        max_gain = max(info_gain_action.values())

        # Collect actions with the maximum info gain
        max_gain_actions = [
            action for action, gain in info_gain_action.items() if gain == max_gain
        ]

        next_action = random.choice(max_gain_actions)
        # Update previous action for the next step
        self.last_action = next_action
        return next_action, info_gain_action

    def mcts_based(self, permitted_actions):
        """Select an action using Monte Carlo Tree Search for multi-step lookahead."""
        # Initialize MCTS if not already done
        if self.mcts is None:
            self.mcts = MCTS(
                planner=self,
                max_depth=self.mcts_depth,
                iterations=self.mcts_iterations,
                exploration_param=self.mcts_exploration
            )
        
        # Create current state representation
        current_uav_pos = self.uav.get_x()
        belief_summary = self.mcts.compress_belief_map(self.M)
        current_state = MCTSState(current_uav_pos, belief_summary, self.mcts_depth)
        
        # Run MCTS search
        best_action = self.mcts.search(current_state)
        
        # Calculate information gains for comparison/logging (optional)
        info_gain_action = {}
        for action in permitted_actions:
            x_future = uav_position(self.uav.x_future(action))
            [[obsd_m_i_min, obsd_m_i_max], [obsd_m_j_min, obsd_m_j_max]] = (
                self.uav.get_range(
                    position=x_future.position,
                    altitude=x_future.altitude,
                    index_form=True,
                )
            )
            obs_M = self.M[obsd_m_i_min:obsd_m_i_max, obsd_m_j_min:obsd_m_j_max, 1]
            info_gain_action[action] = np.sum(self.info_gain(obs_M, x_future))
        
        # Update previous action for the next step
        self.last_action = best_action
        return best_action, info_gain_action

    def select_action(self, belief, visited_x):
        """Select the next UAV action based on the current belief and the chosen strategy."""
        self.M = belief

        permitted_actions = self.uav.permitted_actions(self.uav)  # at UAV position x
        if self.strategy == "sweep":
            return self.sweep(permitted_actions, visited_x)
        elif self.strategy == "mcts":
            return self.mcts_based(permitted_actions)

        return self.ig_based(permitted_actions)
