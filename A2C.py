import os
import learnrl as rl
import tensorflow as tf
import tensorflow_probability as tfp

from carl.agents.tensorflow.memory import Memory
from gym.spaces import Box


class A2CAgent(rl.Agent):

    def __init__(self, actions_space: Box,
                 actor: tf.keras.Model = None, actor_lr=1e-4,
                 value: tf.keras.Model = None, value_lr=1e-4,
                 memory: Memory = None,
                 sample_size=32, discount=0.99, entropy_reg=0,
                 exploration=0, exploration_decay=0, exploration_minimal=0,
                 act_training_period: int = 1,
                 val_training_period: int = 1,
                 act_update_period: int = 1,
                 val_update_period: int = 1,
                 update_factor: float = 1):

        self.actor = actor
        self.actor_opt = tf.keras.optimizers.Adam(actor_lr)
        self.target_actor = tf.keras.models.clone_model(actor)

        self.value = value
        self.value_opt = tf.keras.optimizers.Adam(value_lr)
        self.target_value = tf.keras.models.clone_model(value)

        self.discount = discount

        self.memory = memory
        self.sample_size = sample_size

        assert exploration >= 0
        self.exploration = exploration
        self.exploration_decay = exploration_decay
        self.exploration_minimal = exploration_minimal

        self.entropy_reg = entropy_reg

        assert isinstance(actions_space, Box)
        self.actions_space = actions_space

        self.act_training_period = act_training_period
        self.val_training_period = val_training_period
        self.act_update_period = act_update_period
        self.val_update_period = val_update_period
        self.update_factor = update_factor

        self.step = 0

    def act(self, observation, greedy=False):
        observations = tf.expand_dims(observation, axis=0)
        action_dist = self.actor(observations)
        if greedy:
            action = action_dist.mean()[0]
        else:
            action = action_dist.sample()[0]
            action = self.explore(action)
        return tf.clip_by_value(action, self.actions_space.low, self.actions_space.high)

    def explore(self, action):
        return action + tf.random.normal(action.shape, 0, self.exploration)

    def learn(self):

        if self.step % self.act_update_period == 0:
            self.update(self.actor, self.target_actor)

        if self.step % self.val_update_period == 0:
            self.update(self.value, self.target_value)

        datas = self.memory.sample(self.sample_size)

        metrics = {
            'exploration': self.update_exploration(),
        }

        if self.step % self.val_training_period == 0:
            metrics.update(self.update_value(datas))

        if self.step % self.act_training_period == 0:
            metrics.update(self.update_actor(datas))

        return metrics

    def update_exploration(self):
        self.exploration *= 1 - self.exploration_decay
        self.exploration = max(self.exploration, self.exploration_minimal)
        return self.exploration

    def update(self, initial_net, target_net):
        tau = self.update_factor
        val_weights = initial_net.get_weights()
        for i, target_weight in enumerate(target_net.get_weights()):
            val_weights[i] = tau * val_weights[i] + (1-tau) * target_weight
        target_net.set_weights(val_weights)

    def update_value(self, datas):
        observations, actions, rewards, dones, next_observations = datas
        expected_futur_rewards = self.evaluate(
            rewards, dones, next_observations, self.value)

        with tf.GradientTape() as tape:
            inputs = tf.concat([observations, actions], axis=-1)
            V = self.value(inputs, training=True)[..., 0]
            loss = tf.keras.losses.mse(expected_futur_rewards, V)

        self.update_network(self.value, loss, tape, self.value_opt)
        value_metrics = {
            'value_loss': loss.numpy(),
            'value': tf.reduce_mean(V).numpy()
        }
        return value_metrics

    def update_actor(self, datas):
        observations, actions, rewards, dones, next_observations = datas

        futur_rewards = self.evaluate(
            rewards, dones, next_observations, self.value)
        choosen_actions = self.actor(observations, training=True)
        inputs = tf.concat([observations, choosen_actions], axis=-1)
        V = self.value(inputs)
        critic = futur_rewards - V

        with tf.GradientTape() as tape:
            actions_dist = self.actor(observations, training=True)
            log_prob = actions_dist.log_prob(actions)
            entropy = - tf.exp(log_prob) * log_prob
            loss = - tf.reduce_mean(critic * log_prob -
                                    self.entropy_reg * entropy)

        self.update_network(self.actor, loss, tape, self.actor_opt)
        return {
            'actor_loss': loss.numpy(),
            'entropy': tf.reduce_mean(entropy).numpy()
        }

    def update_network(self, network: tf.keras.Model, loss, tape, opt: tf.keras.optimizers.Optimizer):
        grads = tape.gradient(loss, network.trainable_weights)
        opt.apply_gradients(zip(grads, network.trainable_weights))

    def evaluate(self, rewards, dones, next_observations, value):
        futur_rewards = rewards

        ndones = tf.logical_not(dones)
        if tf.reduce_any(ndones):
            next_actions = self.target_actor(next_observations[ndones])
            next_inputs = tf.concat([next_observations[ndones], next_actions],
                                    axis=-1)
            next_value = self.value(next_inputs)[..., 0]
            ndones_ind = tf.where(ndones)
            futur_rewards = tf.tensor_scatter_nd_add(
                futur_rewards,
                ndones_ind,
                self.discount * next_value
            )

        return futur_rewards

    def remember(self, observation, action, reward, done, next_observation=None, info={}, **param):
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        self.memory.remember(observation, action, reward,
                             done, next_observation)
        self.step += 1

    @staticmethod
    def _get_filenames(filename):
        if filename.endswith('.h5'):
            return filename, None
        actor_path = os.path.join(filename, "actor.h5")
        value_path = os.path.join(filename, "value.h5")
        return actor_path, value_path

    def save(self, filename: str):
        fn_actor = filename + "_act.h5"
        fn_value = filename + "_val.h5"
        tf.keras.models.save_model(self.actor, fn_actor)
        tf.keras.models.save_model(self.value, fn_value)
        print(f'Models saved at {filename + "_val.h5(_act.h5)"}')

    def load(self, filename: str, load_actor: bool = True, load_value: bool = True):
        if load_actor:
            fn_actor = filename + "_act.h5"
            self.actor = tf.keras.models.load_model(fn_actor,
                                                    custom_objects={'tf': tf})
            self.target_actor = tf.keras.models.clone_model(self.actor)
        if load_value:
            fn_value = filename + "_val.h5"
            self.value = tf.keras.models.load_model(fn_value,
                                                    custom_objects={'tf': tf})
            self.target_value = tf.keras.models.clone_model(self.value)


if __name__ == "__main__":
    import numpy as np
    from carl.agents.callbacks import CheckpointCallback
    from carl.environment import Environment
    from carl.utils import generate_circuit

    kl = tf.keras.layers
    tfpl = tfp.layers

    class Config():
        def __init__(self, config):
            for key, val in config.items():
                setattr(self, key, val)

    def circuits_pipeline(mode='aleat'):

        if mode == 'easy':
            pipeline = [
                generate_circuit(n_points=20, difficulty=0),
                generate_circuit(n_points=15, difficulty=0),
                generate_circuit(n_points=10, difficulty=0),
                generate_circuit(n_points=20, difficulty=0),
                generate_circuit(n_points=15, difficulty=0),
            ]
        elif mode == 'samples':
            pipeline = [
                generate_circuit(n_points=20, difficulty=0),

                generate_circuit(n_points=25, difficulty=4),
                generate_circuit(n_points=20, difficulty=4),

                generate_circuit(n_points=25, difficulty=9),
                generate_circuit(n_points=20, difficulty=9),

                generate_circuit(n_points=25, difficulty=12),
                generate_circuit(n_points=20, difficulty=12),

                generate_circuit(n_points=25, difficulty=17),
                generate_circuit(n_points=25, difficulty=17),


                generate_circuit(n_points=25, difficulty=21),
                generate_circuit(n_points=25, difficulty=21),


                generate_circuit(n_points=25, difficulty=25),
                generate_circuit(n_points=25, difficulty=25),

            ]

        elif mode == 'aleat':
            pipeline = [
                generate_circuit(n_points=20, difficulty=0),
                generate_circuit(n_points=15, difficulty=0),
                generate_circuit(n_points=10, difficulty=0),

                generate_circuit(n_points=25, difficulty=4),
                generate_circuit(n_points=20, difficulty=4),
                generate_circuit(n_points=15, difficulty=4),
                generate_circuit(n_points=10, difficulty=4),

                generate_circuit(n_points=25, difficulty=9),
                generate_circuit(n_points=20, difficulty=9),
                generate_circuit(n_points=15, difficulty=9),
                generate_circuit(n_points=15, difficulty=9),
                generate_circuit(n_points=10, difficulty=9),

                generate_circuit(n_points=25, difficulty=12),
                generate_circuit(n_points=20, difficulty=12),
                generate_circuit(n_points=20, difficulty=12),
                generate_circuit(n_points=15, difficulty=12),
                generate_circuit(n_points=15, difficulty=12),
                generate_circuit(n_points=10, difficulty=12),

                generate_circuit(n_points=25, difficulty=17),
                generate_circuit(n_points=25, difficulty=17),
                generate_circuit(n_points=20, difficulty=17),
                generate_circuit(n_points=20, difficulty=17),
                generate_circuit(n_points=15, difficulty=17),
                generate_circuit(n_points=10, difficulty=17),

                generate_circuit(n_points=25, difficulty=21),
                generate_circuit(n_points=25, difficulty=21),
                generate_circuit(n_points=20, difficulty=21),
                generate_circuit(n_points=20, difficulty=21),
                generate_circuit(n_points=15, difficulty=21),

                generate_circuit(n_points=25, difficulty=25),
                generate_circuit(n_points=25, difficulty=25),
                generate_circuit(n_points=25, difficulty=25),
                generate_circuit(n_points=20, difficulty=25),
                generate_circuit(n_points=20, difficulty=25),
                generate_circuit(n_points=15, difficulty=25),
                generate_circuit(n_points=15, difficulty=25),
            ]

        elif mode == '6':
            pipeline = [
                [(0.5, 0), (2.5, 0), (3, 1), (3, 2), (2, 3), (1, 3),
                 (0, 2), (0, 1)],
                [(0, 0), (1, 2), (0, 4), (3, 4), (2, 2), (3, 0)],
                [(0, 0), (0.5, 1), (0, 2), (2, 2), (3, 1), (6, 2), (6, 0)],
                [(1, 0), (6, 0), (6, 1), (5, 1), (5, 2), (6, 2), (6, 3),
                 (4, 3), (4, 2), (2, 2), (2, 3), (0, 3), (0, 1)],
                [(2, 0), (5, 0), (5.5, 1.5), (7, 2), (7, 4), (6, 4), (5, 3),
                 (4, 4), (3.5, 3), (3, 4), (2, 3), (1, 4), (0, 4),
                 (0, 2), (1.5, 1.5)],
            ]

        elif mode == '1':
            pipeline = [[(0.5, 0), (2.5, 0), (3, 1), (3, 2), (2, 3), (1, 3),
                         (0, 2), (0, 1)]*10]

        elif mode == '2':
            pipeline = [[(0, 0), (1, 2), (0, 4), (3, 4), (2, 2), (3, 0)]*10]

        elif mode == '3':
            pipeline = [[(0, 0), (0.5, 1), (0, 2), (2, 2), (3, 1), (6, 2),
                         (6, 0)]*10]

        elif mode == '4':
            pipeline = [[(1, 0), (6, 0), (6, 1), (5, 1), (5, 2), (6, 2),
                         (6, 3), (4, 3), (4, 2), (2, 2), (2, 3), (0, 3),
                         (0, 1)]*10]

        elif mode == '5':
            pipeline = [[(2, 0), (5, 0), (5.5, 1.5), (7, 2), (7, 4), (6, 4),
                         (5, 3), (4, 4), (3.5, 3), (3, 4), (2, 3),
                         (1, 4), (0, 4), (0, 2), (1.5, 1.5)]*10]

        elif mode == '1345':
            pipeline = [[(0.5, 0), (2.5, 0), (3, 1), (3, 2), (2, 3), (1, 3),
                         (0, 2), (0, 1)],
                        [(0, 0), (0.5, 1), (0, 2), (2, 2), (3, 1), (6, 2),
                         (6, 0)],
                        [(1, 0), (6, 0), (6, 1), (5, 1), (5, 2), (6, 2),
                         (6, 3), (4, 3), (4, 2), (2, 2), (2, 3), (0, 3),
                         (0, 1)],
                        [(2, 0), (5, 0), (5.5, 1.5), (7, 2), (7, 4), (6, 4),
                         (5, 3), (4, 4), (3.5, 3), (3, 4), (2, 3),
                         (1, 4), (0, 4), (0, 2), (1.5, 1.5)]]

        elif mode == 'square':
            pipeline = [[(0, 0), (4, 0), (4, 4), (0, 4)]*5]

        elif mode == 'triangle':
            pipeline = [[(0, 0), (2.5, -5), (-2.5, -5)]*5]

        else:
            raise ValueError(f'pipeline circuits mode not understood : {mode}')

        return pipeline

    config = {
        # memory
        'max_memory_len': 10000,
        'mem_method': 'random',             # not yet implemented
        'sample_size': 256,
        # exploration
        'exploration': 1.,
        'exploration_decay': 5e-4,
        'exploration_min': 0.,
        # discount
        'discount': 0.99,
        # learning rate
        'actor_lr': 1e-6,
        'value_lr': 3e-5,
        'lr_decay': 1e-6,                    # not yet implemented
        # entropy
        'entropy_reg': 1e-2,
        # target nets & update parameters
        'val_training_period': 1,
        'act_training_period': 1,
        'val_update_period': 1,
        'act_update_period': 1,
        'update_factor': 0.1,
        # environment
        'speed_rwd': 1/20,
        'circuits_mode': '6',
        # load & save options
        'model_name': 'FerrarlAS6_01',
        'load_model': True,
        'load_model_name': "./models/A2C/FerrarlAS6_01/episode_19",
        'load_actor': True,
        'load_value': True,
        # train/test option
        'test_only': True                    # not yet implemented
    }

    config = Config(config)

    circuits = circuits_pipeline(mode=config.circuits_mode)
    env = Environment(circuits, action_type='continueous',
                      n_sensors=5, fov=np.pi*180/180, car_width=0.1, speed_unit=0.05,
                      speed_rwd=config.speed_rwd)

    memory = Memory(config.max_memory_len)

    # Networks

    init_re = tf.keras.initializers.HeNormal()
    init_th = tf.keras.initializers.GlorotNormal()
    init_fin = tf.keras.initializers.RandomUniform(-3e-3, 3e-3)
    n_obs = env.observation_space.shape[0]
    n_act = env.action_space.shape[0]

    inputs = kl.Input(shape=(n_obs,))
    x = kl.BatchNormalization()(inputs)
    speed = x[..., -1:]
    x = kl.Dense(256, activation='relu', kernel_initializer=init_re)(x)
    x = kl.Dense(128, activation='relu', kernel_initializer=init_re)(x)
    outputs = tf.concat([x, speed], axis=-1)
    feature_extractor = tf.keras.Model(inputs=inputs, outputs=outputs)

    inputs = kl.Input(shape=(n_obs+n_act,))
    x = kl.BatchNormalization()(inputs)
    speed = x[..., n_obs-1: n_obs]
    x = kl.Dense(256, activation='relu', kernel_initializer=init_re)(x)
    x = kl.Dense(128, activation='relu', kernel_initializer=init_re)(x)
    x = tf.concat([x, speed], axis=-1)
    outputs = kl.Dense(1, activation='linear', kernel_initializer=init_fin)(x)
    value_network = tf.keras.Model(inputs=inputs, outputs=outputs)
    value_network.summary()

    event_shape = [2]
    actor_network = tf.keras.models.Sequential([
        feature_extractor,
        kl.Dense(tfpl.IndependentNormal.params_size(
            event_shape), activation=None),
        tfpl.IndependentNormal(event_shape)
    ])
    actor_network.summary()

    agent = A2CAgent(
        actions_space=env.action_space,
        actor=actor_network,
        value=value_network,
        memory=memory,
        sample_size=config.sample_size,
        exploration=config.exploration,
        actor_lr=config.actor_lr,
        value_lr=config.value_lr,
        exploration_decay=config.exploration_decay,
        exploration_minimal=config.exploration_min,
        discount=config.discount,
        entropy_reg=config.entropy_reg,
        val_training_period=config.val_training_period,
        act_training_period=config.act_training_period,
        val_update_period=config.val_update_period,
        act_update_period=config.act_update_period,
        update_factor=config.update_factor

    )

    check = CheckpointCallback(
        os.path.join('models', 'A2C', config.model_name),
        save_every_cycle=True,
        run_test=True,
    )

    metrics = [
        ('reward~env-rwd', {'steps': 'sum', 'episode': 'sum'}),
        ('handled_reward~reward', {'steps': 'sum', 'episode': 'sum'}),
        'actor_loss~aloss',
        'value_loss~vloss',
        'entropy',
        'exploration~exp',
        'value~V'
    ]

    pg = rl.Playground(env, agent)

    if config.load_model == True:
        name = config.load_model_name
        agent.load(name, load_actor=config.load_actor,
                   load_value=config.load_value)

    pg.fit(3000, verbose=2, episodes_cycle_len=10,
           callbacks=[check], metrics=metrics,
           reward_handler=lambda reward, **kwargs: reward
           )
