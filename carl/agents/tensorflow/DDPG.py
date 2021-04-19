from abc import abstractmethod
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import learnrl as rl
import tensorflow as tf
import tensorflow.keras.backend as K
import math
import gym

from carl.agents.tensorflow.memory import Memory

class DdpgAgent(rl.Agent):

    def __init__(self, action_space:gym.spaces.Box, memory:Memory, 
                actor: tf.Module, value: tf.Module,
                discount: float=0.99,
                exploration: float=0.2,
                exploration_decay: float=1e-4,
                exploration_min: float=5e-2,
                sample_size: int=128,
                actor_lr: float=1e-6,
                value_lr: float=1e-4,
                lr_decay: float=1e-4,
                act_training_period: int=1,
                val_training_period: int=1,
                act_update_period: int=1,
                val_update_period: int=1,
                update_factor: float=1,
                mem_method: str='random',
                save_each_cycle: bool=False):
        
        self.action_space = action_space
        self.memory = memory
        
        self.actor = actor
        self.actor_lr_init = actor_lr
        self.target_actor = tf.keras.models.clone_model(actor)
        self.actor_opt = tf.optimizers.Adam(lr=actor_lr)
        actor.compile(self.actor_opt)
        
        self.value = value
        self.value_lr_init = value_lr
        self.target_value = tf.keras.models.clone_model(value)
        self.value_opt = tf.optimizers.Adam(lr=value_lr)
        value.compile(self.value_opt)
        
        self.lr_decay = lr_decay
        self.discount = discount
        self.exploration = exploration
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.sample_size = sample_size
        
        self.act_training_period = act_training_period
        self.val_training_period = val_training_period
        self.act_update_period = act_update_period
        self.val_update_period = val_update_period
        self.update_factor = update_factor
        
        self.mem_method = mem_method
        
        self.save_each_cycle = save_each_cycle
        if save_each_cycle:
            self.save_step = 0
            self.rwd_buffer = []
        
        self.step = 0

    def act(self, observation, greedy=False):
        observations = tf.expand_dims(observation, axis=0)
        action = self.actor(observations)[0]
        if greedy:
            return action
        else:
            return self.explore(action)

    def explore(self, action):
        action += tf.random.normal(action.shape, mean=0.0, 
                                   stddev=self.exploration)
        
        action = tf.clip_by_value(action, self.action_space.low, 
                                  self.action_space.high)
        
        self.exploration *= (1-self.exploration_decay)
        self.exploration = max(self.exploration, self.exploration_min)

        return action

    def update(self, initial_net, target_net):
        tau = self.update_factor
        val_weights = initial_net.get_weights()
        for i, target_weight in enumerate(target_net.get_weights()):
            val_weights[i] = tau * val_weights[i] + (1-tau) * target_weight
        target_net.set_weights(val_weights)
    
    def update_lr(self):
        lr_act = (1-self.lr_decay) * self.actor.optimizer.lr
        lr_val = (1-self.lr_decay) * self.value.optimizer.lr
        K.set_value(self.actor.optimizer.lr, lr_act)
        K.set_value(self.value.optimizer.lr, lr_val)
    
    def learn(self):
        
        # if len(self.memory) < self.sample_size :
        #     return
        
        if self.step % self.act_update_period == 0:
            self.update(self.actor, self.target_actor)
        
        if self.step % self.val_update_period == 0:
            self.update(self.value, self.target_value)
        
        experiences = self.memory.sample(self.sample_size, method=self.mem_method)
        
        metrics = {'exploration': self.exploration}
        
        if self.step % self.val_training_period == 0:
            metrics.update(self.update_value(experiences))
        
        if self.step % self.act_training_period == 0:
            metrics.update(self.update_actor(experiences)) # à tester si on peut pas trouver des experiences mieux pour les 2
        
        self.update_lr()

        return metrics
    
    def update_value(self, experiences):
        observations, actions, rewards, dones, next_observations = experiences
                                        
        expected_futur_rewards = self.evaluation(rewards, dones, next_observations)
        
        with tf.GradientTape() as tape:
            inputs = tf.concat([observations, actions], axis=-1)
            Q = self.value(inputs, training=True)[..., 0]
            value_loss = tf.reduce_mean(tf.math.square(Q - expected_futur_rewards))
        
        self.update_network(value_loss, tape, self.value, self.value_opt)
        
        value_metrics = {
            'val_loss': value_loss.numpy(),
            'Q': tf.reduce_mean(Q).numpy(),
            'val_lr': K.eval(self.value.optimizer.lr)
        }
        return value_metrics
    
    def update_actor(self, experiences):
        observations, _, _, _, _ = experiences
        
        with tf.GradientTape() as tape:
            choosen_actions = self.actor(observations, training=True)
            inputs = tf.concat([observations, choosen_actions], axis=-1)
            # actor_loss = -tf.reduce_mean(self.target_value(inputs))
            var = tf.math.reduce_std(self.value(inputs))
            actor_loss = -tf.reduce_mean(self.value(inputs))
        
        self.update_network(actor_loss, tape, self.actor, self.actor_opt)
        
        # var sur le batch ??
        value_metrics = {
            'act_loss': actor_loss.numpy(),
            'act_lr': K.eval(self.actor.optimizer.lr),
            'var Q batch': var.numpy()
        }
        return value_metrics   
    
    def update_network(self, loss, tape, network, optimizer):
        gradients = tape.gradient(loss, network.trainable_weights)
        optimizer.apply_gradients(zip(gradients, network.trainable_weights))
    
    def evaluation(self, rewards, dones, next_observations):
        futur_rewards = rewards
        ndones = tf.logical_not(dones)
        if tf.reduce_any(ndones):
            next_actions = self.target_actor(next_observations[ndones])
            next_inputs = tf.concat([next_observations[ndones], next_actions],
                                    axis=-1)
            next_value = self.target_value(next_inputs)[..., 0]
            
            ndones_indicies = tf.where(ndones)
            futur_rewards = tf.tensor_scatter_nd_add(
                            futur_rewards, ndones_indicies, 
                            self.discount * next_value
                            )
            
        return futur_rewards   
    
    def remember(self, observation, action, reward, done,
                 next_observation=None, info={}, **param):
        self.memory.remember(observation, action, reward, done,
                             next_observation)
        self.step += 1
        
        if self.save_each_cycle:
            self.rwd_buffer.append(reward)
            if len(self.rwd_buffer) > 500:
                self.rwd_buffer = self.rwd_buffer[1:]
    
    def save(self, filename: str):
        fn_actor = filename + "_act.h5"
        fn_value = filename + "_val.h5"
        tf.keras.models.save_model(self.actor, fn_actor)
        tf.keras.models.save_model(self.value, fn_value)
        print(f'Models saved at {filename + "_val.h5(_act.h5)"}')
        
        if self.save_each_cycle:
            # save models in specific folder
            self.save_step += 1
            filename_dir = os.path.join(*filename.split('/')[:-1])
            model_name = filename.split('/')[-1]
            filename_dir = os.path.join(filename_dir, model_name)
            if not os.path.isdir(filename_dir):
                os.mkdir(filename_dir)
            fn_actor = os.path.join(filename_dir, f'save_n°{self.save_step}_act.h5')
            fn_value = os.path.join(filename_dir, f'save_n°{self.save_step}_val.h5')
            tf.keras.models.save_model(self.actor, fn_actor)
            tf.keras.models.save_model(self.value, fn_value)
            
            # save local average reward
            log_name = os.path.join(filename_dir, "log rewards.txt")
            with open(log_name, "a") as file:
                np_buffer = np.array(self.rwd_buffer)
                txt = f"save_n°{self.save_step} reward {np.mean(np_buffer): 2.4}\n"
                file.write(txt)
    
    def load(self, filename: str, load_actor: bool=True, load_value: bool=True):
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
        
        # init optimizer with new lr
        self.actor_opt = tf.optimizers.Adam(lr=self.actor_lr_init)
        self.actor.compile(self.actor_opt)
        self.value_opt = tf.optimizers.Adam(lr=self.value_lr_init)
        self.value.compile(self.value_opt)


if __name__ == '__main__':
    from carl.environment import Environment
    from carl.agents.callbacks import ScoreCallback, CheckpointCallback
    from carl.utils import generate_circuit
    import numpy as np
    kl = tf.keras.layers

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

                generate_circuit(n_points=25, difficulty=4),
                generate_circuit(n_points=20, difficulty=4),
                generate_circuit(n_points=15, difficulty=4),
                generate_circuit(n_points=10, difficulty=4)
            ]
        
        elif mode == 'aleat':
            pipeline = [
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
            
        elif mode == 'fixed':
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
            
        elif mode == 'square':
            pipeline = [[(0, 0), (4, 0), (4, 4), (0, 4)]*5]
            
        elif mode == 'triangle':
            pipeline = [[(0, 0), (2.5, -5), (-2.5, -5)]*5]
        
        else:
            raise ValueError(f'pipeline circuits mode not understood : {mode}')
        
        return pipeline
    
    config = {
        # memory
        'max_memory_len': 49984,
        'mem_method': 'random',
        'sample_size': 1024,
        # exploration
        'exploration': 0.58,
        'exploration_decay': 0.2e-5,
        'exploration_min': 0.25,
        # discount
        'discount': 0.99,
        # learning rate
        'actor_lr': 3e-6,
        'value_lr': 1e-5,
        'lr_decay': 1e-6,
        # target nets & update parameters
        'val_training_period': 1,
        'act_training_period': 10,
        'val_update_period': 1,
        'act_update_period': 10,  
        'update_factor': 0.005,
        # environment
        'speed_rwd': -0.002,
        'circuits_mode': 'aleat',
        # load & save options
        'model_name': 'FerrarlVGa_03',
        'load_model': True,
        'load_model_name': "./models/DDPG/FerrarlVGa_02",
        'load_actor': True,
        'load_value': True,
        'save_each_cycle': False,
        # train/test option
        'test_only': True
    }

    config = Config(config)
    
    circuits = circuits_pipeline(mode=config.circuits_mode)

    env =  Environment(circuits=circuits, action_type='continuous',
                       fov=math.pi*220/180, n_sensors=9,
                       speed_rwd=config.speed_rwd)

    ## networks
    init_re = tf.keras.initializers.HeNormal()
    init_th = tf.keras.initializers.GlorotNormal()
    init_fin = tf.keras.initializers.RandomUniform(-3e-3, 3e-3)
    n_obs = env.observation_space.shape[0]
    n_act = env.action_space.shape[0]
    # actor net
    inputs = kl.Input(shape=(n_obs,))
    speed = inputs[..., -1:]
    x = kl.BatchNormalization()(inputs)
    x = kl.Dense(512, activation='relu', kernel_initializer=init_re)(x)
    x = kl.Dense(256, activation='relu', kernel_initializer=init_re)(x)
    x = tf.concat([x, speed], axis=-1)
    outputs = kl.Dense(n_act, activation='tanh',
                 kernel_initializer=init_fin)(x)
    actor_network = tf.keras.Model(inputs=inputs, outputs=outputs)
    # value net
    inputs = kl.Input(shape=(n_obs + n_act,))
    speed = inputs[..., n_obs-1 : n_obs]
    x = kl.BatchNormalization()(inputs)
    x = kl.Dense(512, activation='relu', kernel_initializer=init_re)(x)
    x = kl.Dense(256, activation='relu', kernel_initializer=init_re)(x)
    x = tf.concat([x, speed], axis=-1)
    outputs = kl.Dense(1, activation='linear',
                 kernel_initializer=init_fin)(x)
    value_network = tf.keras.Model(inputs=inputs, outputs=outputs)


    agent = DdpgAgent(action_space=env.action_space,
                      memory=Memory(config.max_memory_len),
                      actor=actor_network,
                      value=value_network,
                      actor_lr=config.actor_lr,
                      value_lr=config.value_lr,
                      lr_decay=config.lr_decay,
                      discount=config.discount,
                      exploration=config.exploration,
                      exploration_decay=config.exploration_decay,
                      exploration_min = config.exploration_min,
                      sample_size=config.sample_size,
                      act_training_period=config.act_training_period,
                      val_training_period=config.val_training_period,
                      act_update_period=config.act_update_period,
                      val_update_period=config.val_update_period,
                      update_factor=config.update_factor,
                      mem_method=config.mem_method,
                      save_each_cycle=config.save_each_cycle,
                      )

    if config.load_model:
        # import previous model
        file_name = config.load_model_name
        agent.load(file_name, load_actor=config.load_actor,
                   load_value=config.load_value)

    metrics=[
        ('reward~env-rwd', {'steps': 'sum', 'episode': 'sum'}),
        ('handled_reward~reward', {'steps': 'sum', 'episode': 'sum'}),
        'act_loss',
        'val_loss',
        'act_lr',
        'val_lr',
        'exploration~exp',
        'Q',
        'var Q batch',
    ]

    check = CheckpointCallback(os.path.join('models', 'DDPG',
                                            f"{config.model_name}"))
    score_callback = ScoreCallback(print_circuits=False)

    pg = rl.Playground(env, agent)

    if not config.test_only:
        pg.fit(100000, verbose=2, metrics=metrics, episodes_cycle_len=len(circuits),
            reward_handler=lambda reward, **kwargs: 0.1*reward,
            callbacks=[check])
    
    # score printed at the end
    pg.test(len(circuits), verbose=1, episodes_cycle_len=1,
            callbacks=[score_callback])
