import tensorflow as tf
import tensorflow.keras.backend as K
kl = tf.keras.layers
from carl.utils import generate_circuit
from carl.environment import Environment
import math

circuits = [
        generate_circuit(n_points=20, difficulty=0),
        generate_circuit(n_points=15, difficulty=0),
        generate_circuit(n_points=10, difficulty=0),]

env =  Environment(circuits=circuits, action_type='continuous',
                       fov=math.pi*220/180, n_sensors=9,
                       speed_rwd=0.)

n_obs = env.observation_space.shape[0]

init_re = tf.keras.initializers.HeNormal()
init_th = tf.keras.initializers.GlorotNormal()
init_fin = tf.keras.initializers.RandomUniform(-3e-3, 3e-3)
n_obs = env.observation_space.shape[0]
n_act = env.action_space.shape[0]

model_test = tf.keras.Sequential([
    kl.BatchNormalization(),
    kl.Dense(512, activation='relu', kernel_initializer=init_re),
    kl.Dense(256, activation='relu', kernel_initializer=init_re),
    kl.Dense(n_act, activation='tanh',
                 kernel_initializer=init_fin)
])

model_test.build(input_shape=(1, n_obs))

filename = "./models/DDPG/model_test_act.h5"

tf.keras.models.save_model(model_test, filename)
print(n_obs)
print(n_act)