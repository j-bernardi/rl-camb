import os, random, itertools, sys, pprint, h5py, math

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from utils import utils

from collections import deque

from models.standard_agent import StandardAgent


class DQNSolver(StandardAgent):
    """A standard dqn_solver.
    Implements a simple DNN that predicts values.
    """

    def __init__(self, experiment_name, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size
        self.memory_len = 100000
        self.memory = deque(maxlen=self.memory_len)
        self.gamma = 1.0    # discount rate was 1
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.learning_rate_decay = 0.01
        self.batch_size = 64

        self.model_name = "dqn"
        self.model = self.build_model()

        self.optimizer = Adam(
            lr=self.learning_rate, 
            decay=self.learning_rate_decay)

        super(DQNSolver, self).__init__(self.model_name + "_" + experiment_name)

        self.load_state()

    def build_model(self):

        tf.keras.backend.set_floatx('float64')

        model = tf.keras.Sequential(name=self.model_name)
        model.add(Dense(24, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(48, activation='tanh'))
        model.add(Dense(self.action_size, activation='linear'))

        model.build()

        return model

    def show_example(self, env_wrapper, steps):
        """Show a quick view of the environemt, 
        without trying to solve.
        """
        env = env_wrapper.env
        env.reset()
        for _ in range(steps):
            self.env.render()
            self.env.step(env.action_space.sample())
        env.close()

    def do_random_runs(self, env_wrapper, episodes, steps, verbose=False, wait=0.0):
        """Run some episodes with random actions, stopping on 
        actual failure / win conditions. Just for viewing.
        """
        env = env_wrapper.env
        for i_episode in range(episodes):
            observation = env.reset()
            print("Episode {}".format(i_episode+1))
            for t in range(steps):
                self.env.render()
                if verbose:
                    print(observation)
                # take a random action
                action = env.action_space.sample()
                observation, reward, done, info = env.step(action)
                time.sleep(wait)
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break
        env.close()

    def solve(self, env_wrapper, max_episodes, verbose=False, render=False):
        env = env_wrapper.env

        for episode in range(max_episodes):
            state = env.reset()
            # Take steps until failure / win
            for step in itertools.count():
                if render:
                    env.render()
                action = self.act(state, eps=self.epsilon)
                observation, reward, done, _ = env.step(action)
                state_next = observation
                # Custom reward if required by env wrapper
                reward = env_wrapper.reward_on_step(state, state_next, reward, done)
                self.memory.append((state, np.int32(action), reward, state_next, done))
                state = observation
                
                print(f"\rEpisode {episode + 1}/{max_episodes} - "
                      f"steps {step} ({self.total_t + 1})", 
                      end="")
                sys.stdout.flush()

                self.total_t += 1
                if done:
                    break

            self.learn()
            
            # Calculate a (optionally custom) score for this episode
            score = env_wrapper.get_score(state, state_next, reward, step)
            self.scores.append(score) 

            solved, overall_score = env_wrapper.check_solved_on_done(state, self.scores, verbose=verbose)

            if episode % 25 == 0 or solved:
                print(f"\rEpisode {episode + 1}/{max_episodes} - steps {step} - "
                      f"score {int(overall_score)}/{int(env_wrapper.score_target)}")
                self.save_state()

            if solved:
                return True

        return False

    def act(self, state, eps=0.0):
        """
        Take a state and return a random action or the most 
        valuable predictedaction, based on the agent's model. 
        """
        assert state.shape == (self.state_size,) or state.shape == (1, self.state_size)

        # If in exploration
        if np.random.rand() <= eps:
            return random.randrange(self.action_size)

        if len(state.shape) == 1:
            state = np.reshape(state, (1, state.shape[0]))
        
        act_values = self.model(state)

        return np.argmax(act_values, axis=-1)[0]
    
    def learn(self):
        """Updated the agent's decision network based
        on a sample of previous decisions it has seen.
        Here, we combine the target and action networks.
        """

        minibatch_i =  np.random.choice(
                min(self.batch_size, len(self.memory)),
                self.batch_size)

        minibatch = [self.memory[i] for i in minibatch_i]

        loss_value = self.take_training_step(
            *tuple(map(np.array, zip(*minibatch)))
            )

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss_value

    @tf.function
    def take_training_step(self, sts, a, r, n_sts, d):

        # TODO - can also learn OFF policy; e.g. q_predictions_off_a
        future_q_pred = tf.math.reduce_max(self.model(n_sts), axis=1)
        future_q_pred = tf.where(d, tf.zeros((1,), dtype=tf.dtypes.float64), future_q_pred)
        q_targets = r + self.gamma * future_q_pred

        loss_value, grads = self.squared_diff_loss_at_a(sts, a, q_targets)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return loss_value

    @tf.function
    def squared_diff_loss_at_a(self, states, action_mask, targets_from_memory):
        """
        A squared difference loss function 
        Diffs the Q model's predicted values with 
        the actual values plus the discounted next state by the target Q network
        """
        with tf.GradientTape() as tape:

            q_predictions = self.model(states)
            
            # TODO - can also learn OFF policy; e.g. q_predictions_off_a
            gather_indices = tf.range(self.batch_size) * tf.shape(q_predictions)[-1] + action_mask
            q_predictions_at_a = tf.gather(tf.reshape(q_predictions, [-1]), gather_indices)

            losses = tf.math.squared_difference(q_predictions_at_a, targets_from_memory)
            reduced_loss = tf.math.reduce_mean(losses)

        return reduced_loss, tape.gradient(reduced_loss, self.model.trainable_variables)

    def save_state(self):
        """Save a (trained) model with its weights to a specified file.
        Metadata should be passed to keep information avaialble.
        """

        self.model.save(self.model_location)

        data_dict = {key: getattr(self, key) for key in (
            "model_location", "scores", "total_t",
            "epsilon")
        }

        data_dict["trained_episodes"] = len(self.scores)
        utils.add_to_h5(self.model_location, data_dict)

        utils.add_to_h5(
            self.model_location,
            self.optimizer.get_config(),
            group_name="optimizer_config")

        mem_as_tuple_of_arrays = tuple(map(np.array, zip(*self.memory)))
        memory_arrays = {"tup_" + str(i): mem_as_tuple_of_arrays[i] 
                         for i in range(len(mem_as_tuple_of_arrays))}

        utils.add_to_h5(self.model_location, memory_arrays, group_name="memory")

    def load_state(self):
        """Load a model with the specified name"""

        print("Loading weights from", self.model_location + "...", end="")
        if os.path.exists(self.model_location):
            self.model = tf.keras.models.load_model(self.model_location)
            self.model.summary()
            print(" Loaded.")

            with h5py.File(self.model_location, 'r') as hf:
                
                self.optimizer = self.optimizer.from_config({
                    k: hf["optimizer_config"][k][()] 
                    for k in hf["optimizer_config"].keys()})
                
                self.memory = deque(maxlen=self.memory_len)
                mem = {k: np.array(hf["memory"][k]) for k in hf["memory"].keys()}
                mem_as_tuple_of_arrays = tuple([mem["tup_" + str(i)] for i in range(len(mem.keys()))])
                for tup in zip(*mem_as_tuple_of_arrays):
                    assert all([val is not None for val in tup])
                    self.memory.append(tup)

                other = {k: hf["custom_group"][k] for k in hf["custom_group"].keys()}
                self.scores = list(other["scores"])
                self.epsilon = other["epsilon"][()]
                self.total_t = other["total_t"][()]
                
                data_dict = {
                    "trained_episodes": len(self.scores),
                    "epsilon": self.epsilon,
                    "total_t": self.total_t,
                    "memory_loaded": len(self.memory)
                }

            print("Loaded state:")
            pprint.pprint(data_dict, depth=1)

        else:
            print(" Model not yet saved at loaction.")
