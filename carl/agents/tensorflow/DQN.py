#%%
import os
import learnrl as rl
import tensorflow as tf

from carl.agents.tensorflow.memory import Memory
from copy import deepcopy

class Control():

    def __init__(self, exploration=0, exploration_decay=0, exploration_minimum=0):
        self.exploration = exploration
        self.exploration_decay = exploration_decay
        self.exploration_minimum = exploration_minimum

    def update_exploration(self):
        self.exploration *= 1 - self.exploration_decay
        self.exploration = max(self.exploration, self.exploration_minimum)

    def act(self, Q):
        raise NotImplementedError('You must define act(self, Q) when subclassing Control')

    def __call__(self, Q, greedy):
        if greedy:
            return tf.argmax(Q, axis=-1, output_type=tf.int32)
        else:
            return self.act(Q)

class EpsGreedy(Control):

    def __init__(self, *args):
        super().__init__(*args)
        assert self.exploration <= 1 and self.exploration >= 0, \
            "Exploration must be in [0, 1] for EpsGreedy"

    def act(self, Q):
        batch_size = Q.shape[0]
        action_size = Q.shape[1]

        actions_random = tf.random.uniform((batch_size,), 0, action_size, dtype=tf.int32) #pylint: disable=all
        actions_greedy = tf.argmax(Q, axis=-1, output_type=tf.int32)

        rd = tf.random.uniform((batch_size,), 0, 1)
        actions = tf.where(rd <= self.exploration, actions_random, actions_greedy)

        return actions

class Evaluation():

    def __init__(self, discount):
        self.discount = discount

    def eval(self, rewards, dones, next_observations, action_value):
        raise NotImplementedError('You must define eval when subclassing Evaluation')

    def __call__(self, rewards, dones, next_observations, action_value):
        return self.eval(rewards, dones, next_observations, action_value)

class QLearning(Evaluation):

    def eval(self, rewards, dones, next_observations, action_value):
        futur_rewards = rewards

        ndones = tf.logical_not(dones)
        if tf.reduce_any(ndones):
            next_values = tf.reduce_max(action_value(next_observations[ndones]), axis=-1)

            ndones_indexes = tf.where(ndones)
            futur_rewards = tf.tensor_scatter_nd_add(futur_rewards, ndones_indexes, self.discount * next_values)

        return futur_rewards


class DQNAgent(rl.Agent):

    def __init__(self, action_value:tf.keras.Model=None,
            control:Control=None,
            memory:Memory=None,
            evaluation:Evaluation=None,
            sample_size=32,
            learning_rate=1e-4,
            training_period=4,
            update_period=1,
            update_factor=1
        ):
        
        self.action_value = action_value
        self.action_value_opt = tf.keras.optimizers.Adam(learning_rate)
        self.target_av = action_value

        self.control = Control() if control is None else control
        self.memory = memory
        self.evaluation = evaluation

        self.sample_size = sample_size
        
        self.step = 0
        self.training_period = training_period
        self.update_period = update_period
        self.update = update_factor

    @tf.function
    def act(self, observation, greedy=False):
        observations = tf.expand_dims(observation, axis=0)
        Q = self.action_value(observations)
        action = self.control(Q, greedy)[0]
        return action

    def learn(self):
        if self.step % self.update_period == 0:
            # update parameters in target action value
            weights = self.action_value.get_weights()
            for i, target_weight in enumerate(self.target_av.get_weights()):
                weights[i] = self.update * weights[i] + (1-self.update) * target_weight
            self.target_av.set_weights(weights)
        
        if self.step % self.training_period != 0:
            # skip learning step
            return
        
        observations, actions, rewards, dones, next_observations = self.memory.sample(self.sample_size)
        expected_futur_rewards = self.evaluation(rewards, dones, next_observations, self.target_av)

        
        with tf.GradientTape() as tape:
            Q = self.action_value(observations)

            action_index = tf.stack( (tf.range(len(actions)), actions) , axis=-1)
            Q_action = tf.gather_nd(Q, action_index)

            loss = tf.keras.losses.mse(expected_futur_rewards, Q_action)

        grads = tape.gradient(loss, self.action_value.trainable_weights)
        self.action_value_opt.apply_gradients(zip(grads, self.action_value.trainable_weights))

        metrics = {
            'value': tf.reduce_mean(Q_action).numpy(),
            'loss': loss.numpy(),
            'exploration': self.control.exploration,
            'learning_rate': self.action_value_opt.lr.numpy()
        }

        self.control.update_exploration()
        return metrics

    def remember(self, observation, action, reward, done, next_observation=None, info={}, **param):
        self.memory.remember(observation, action, reward, done, next_observation)
        self.step += 1
    
    def save(self, filename):
        filename += ".h5"
        tf.keras.models.save_model(self.action_value, filename)
        print(f'Model saved at {filename}')

    def load(self, filename):
        self.action_value = tf.keras.models.load_model(filename, custom_objects={'tf': tf})

if __name__ == "__main__":
    from carl.environment import Environment
    from carl.agents.callbacks import ScoreCallback, CheckpointCallback
    from carl.utils import generate_circuit
    import numpy as np
    kl = tf.keras.layers

    class Config():
        def __init__(self, config):
            for key, val in config.items():
                setattr(self, key, val)

    config_baseline = {
        'model_name': 'baseline',
        'max_memory_len': 40960,

        'exploration': 0.1,
        'exploration_decay': 1e-4,
        'exploration_minimum': 5e-2,

        'discount': 0.90,

        'dense_1_size': 512,
        'dense_1_activation': 'tanh',
        'dense_2_size': 256,
        'dense_2_activation': 'relu',
        'dense_3_size': 128,
        'dense_3_activation': 'relu',

        'sample_size': 4096,
        'learning_rate': 2e-4,
        
        'training_period': 1,
        'update_period': 50,
        'update_factor': 0.2
    }
    
    config = {
        'model_name': 'Ferrarlvg04',
        'max_memory_len': 40960,

        'exploration': 0.15,
        'exploration_decay': 0.5e-4,
        'exploration_minimum': 3e-2,

        'discount': 0.9,  # test avec 0.8

        'dense_1_size': 256,
        'dense_1_activation': 'relu',
        'dense_2_size': 128,
        'dense_2_activation': 'relu',
        'dense_3_size': 128,
        'dense_3_activation': 'relu',
        'dense_4_size': 64,
        'dense_4_activation': 'relu',
        
        'sample_size': 4096,
        'learning_rate': 0.2e-4,
        
        'training_period': 1,
        'update_period': 1,
        'update_factor': 0.3
    }

    config = Config(config)

    circuits = [
        [(0.5, 0), (2.5, 0), (3, 1), (3, 2), (2, 3), (1, 3), (0, 2), (0, 1)],
        [(0, 0), (1, 2), (0, 4), (3, 4), (2, 2), (3, 0)],
        [(0.5, 0), (2.5, 0), (3, 1), (3, 2), (2, 3), (1, 3), (0, 2), (0, 1)],
        [(1, 0), (6, 0), (6, 1), (5, 1), (5, 2), (6, 2), (6, 3),
        (4, 3), (4, 2), (2, 2), (2, 3), (0, 3), (0, 1)],
        [(2, 0), (5, 0), (5.5, 1.5), (7, 2), (7, 4), (6, 4), (5, 3), (4, 4),
        (3.5, 3), (3, 4), (2, 3), (1, 4), (0, 4), (0, 2), (1.5, 1.5)],
        generate_circuit(n_points=25, difficulty=0),
        generate_circuit(n_points=20, difficulty=10),
        generate_circuit(n_points=15, difficulty=0),
        [(0.5, 0), (2.5, 0), (3, 1), (3, 2), (2, 3), (1, 3), (0, 2), (0, 1)],
        [(0, 0), (1, 2), (0, 4), (3, 4), (2, 2), (3, 0)],
        [(0, 0), (0.5, 1), (0, 2), (2, 2), (3, 1), (6, 2), (6, 0)],
        [(1, 0), (6, 0), (6, 1), (5, 1), (5, 2), (6, 2), (6, 3),
        (4, 3), (4, 2), (2, 2), (2, 3), (0, 3), (0, 1)],
        [(2, 0), (5, 0), (5.5, 1.5), (7, 2), (7, 4), (6, 4), (5, 3), (4, 4),
        (3.5, 3), (3, 4), (2, 3), (1, 4), (0, 4), (0, 2), (1.5, 1.5)],
        [(2, 0), (5, 0), (5.5, 1.5), (7, 2), (7, 4), (6, 4), (5, 3), (4, 4),
        (3.5, 3), (3, 4), (2, 3), (1, 4), (0, 4), (0, 2), (1.5, 1.5)],
        [(2, 0), (5, 0), (5.5, 1.5), (7, 2), (7, 4), (6, 4), (5, 3), (4, 4),
        (3.5, 3), (3, 4), (2, 3), (1, 4), (0, 4), (0, 2), (1.5, 1.5)],
        generate_circuit(n_points=15, difficulty=0),
        generate_circuit(n_points=20, difficulty=5),
        generate_circuit(n_points=20, difficulty=5),
        generate_circuit(n_points=25, difficulty=10),
        generate_circuit(n_points=20, difficulty=10),
        generate_circuit(n_points=25, difficulty=10),
    ]
    n_circuits = len(circuits)
    env = Environment(circuits, names=config.model_name.capitalize(),
                    n_sensors=7, fov=np.pi*220/180)

    memory = Memory(config.max_memory_len)
    control = EpsGreedy(
        config.exploration,
        config.exploration_decay,
        config.exploration_minimum
    )
    evaluation = QLearning(config.discount)

    init_re = tf.keras.initializers.HeUniform()
    init_th = tf.keras.initializers.GlorotUniform()
    
    # action_value = tf.keras.models.Sequential((
    #     kl.Dense(config.dense_1_size, activation=config.dense_1_activation,
    #              kernel_initializer=init_re),
    #     kl.BatchNormalization(),
    #     kl.Dense(config.dense_2_size, activation=config.dense_2_activation,
    #              kernel_initializer=init_re),
    #     kl.Dense(config.dense_3_size, activation=config.dense_3_activation,
    #              kernel_initializer=init_re),
    #     kl.Dense(config.dense_4_size, activation=config.dense_4_activation,
    #              kernel_initializer=init_re),
    #     kl.Dense(env.action_space.n, activation='linear',
    #              kernel_initializer=init_re)
    # ))
    pre_model_name = 'Ferrarlvg03.h5'
    action_value = tf.keras.models.load_model(f'models/DQN/{pre_model_name}')
    
    agent = DQNAgent(
        action_value=action_value,
        control=control,
        memory=memory,
        evaluation=evaluation,
        sample_size=config.sample_size,
        learning_rate=config.learning_rate,
        training_period=config.training_period,
        update_period=config.update_period,
        update_factor=config.update_factor
    )
    
    metrics=[
        ('reward~env-rwd', {'steps': 'sum', 'episode': 'sum'}),
        ('handled_reward~reward', {'steps': 'sum', 'episode': 'sum'}),
        'loss',
        'exploration~exp',
        'value~Q'
    ]

    score = ScoreCallback(print_circuits=True)
    check = CheckpointCallback(os.path.join('models', 'DQN', f"{config.model_name}"))

    pg = rl.Playground(env, agent)
    pg.fit(
        n_circuits*950, verbose=2, metrics=metrics,
        episodes_cycle_len=n_circuits,
        callbacks=[check]
    )
    # score for each circuit (please ignore 'n°XX')
    pg.test(len(circuits), verbose=1, episodes_cycle_len=1, callbacks=[score])
    # final score
    print('\nscore final :')
    pg.test(len(circuits), verbose=0, callbacks=[ScoreCallback()])