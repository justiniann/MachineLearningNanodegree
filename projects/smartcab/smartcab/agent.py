import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import random

"""
The concept of Q-Learning is fairly straightforward:
For every state the agent visits, create an entry in the Q-table for all state-action pairs available.
Then, when the agent encounters a state and performs an action, update the Q-value associated with that state-action pair based on the reward received and the iterative update rule implemented.
Of course, additional benefits come from Q-Learning, such that we can have the agent choose the *best* action for each state based on the Q-values of each state-action pair possible.
For this project, you will be implementing a *decaying,* $\epsilon$*-greedy* Q-learning algorithm with *no* discount factor.

Furthermore, note that you are expected to use a *decaying* $\epsilon$ *(exploration) factor*.
Hence, as the number of trials increases, $\epsilon$ should decrease towards 0.
This is because the agent is expected to learn from its behavior and begin acting on its learned behavior. Additionally,
The agent will be tested on what it has learned after $\epsilon$ has passed a certain threshold (the default threshold is 0.05).
For the initial Q-Learning implementation, you will be implementing a linear decaying function for $\epsilon$.

decay funciton for e = e(t+1) = e(t) - 0.05, where t = trial number
"""

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning  # Whether the agent is expected to learn
        self.Q = dict()  # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon    # Random exploration factor
        self.alpha = alpha        # Learning factor
        self.trial_num = 1

        # initial_action_map is the dict that is initialized for newly visited actions
        self.initial_action_map = {}
        for k in self.valid_actions:
            self.initial_action_map[k] = 0.0

    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """
        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        if testing:
            self.trial_num = 0
            self.epsilon = 0
            self.alpha = 0
        else:
            # self.epsilon = 1 / math.pow(self.trial_num, 1.2)
            self.epsilon = 1 / math.pow(2, (self.trial_num*0.2))
            self.trial_num += 1
        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the
            environment. The next waypoint, the intersection inputs, and the deadline
            are all features available to the agent. """
        # Collect data about the environment
        waypoint = self.planner.next_waypoint()  # The next waypoint
        inputs = self.env.sense(self)            # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)   # Remaining deadline
        return waypoint, inputs['light'], inputs['oncoming']

    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """
        # Calculate the maximum Q-value of all actions for a given state
        maxQ = -999
        if state in self.Q:
            for action in self.Q[state]:
                if self.Q[state][action] > maxQ:
                    maxQ = self.Q[state][action]
        return maxQ

    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        if self.learning:
            if state not in self.Q:
                self.Q[state] = dict(self.initial_action_map)

    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """
        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        # Otherwise, choose an action with the highest Q-value for the current state
        # Be sure that when choosing an action with highest Q-value that you randomly select between actions that "tie".
        if not self.learning or self.should_choose_random():
            return self.valid_actions[random.randint(0, len(self.valid_actions)-1)]
        else:
            max_q = self.get_maxQ(state)
            best_actions = [a for a in self.Q[state] if self.Q[state][a] == max_q]
            if len(best_actions) > 1:
                return best_actions[random.randint(0, len(best_actions) - 1)]
            else:
                return best_actions[0]

    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives a reward. This function does not consider future rewards
            when conducting learning. """
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        if self.learning:
            # utility_of_current_state = reward + (self.epsilon * self.get_maxQ(state))
            # self.Q[state][action] = ((1-self.alpha) * utility_of_current_state) + (self.alpha * self.Q[state][action])
            self.Q[state][action] = self.alpha * reward + (1 - self.alpha) * self.Q[state][action]

    def update(self):
        """ The update function is called when a time step is completed in the
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """
        state = self.build_state()  # Get current state
        self.createQ(state)  # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action)  # Receive a reward
        self.learn(state, action, reward)  # Q-learn

    def should_choose_random(self):
        return random.random() < self.epsilon

def run():
    """ Driving function for running the simulation.
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment()
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent, learning=True, alpha=0.4)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline=True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, update_delay=0.1, log_metrics=False, display=True, optimized=True)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(n_test=10)


if __name__ == '__main__':
    run()
