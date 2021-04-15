import os
import learnrl as rl
import tensorflow as tf
from tensorflow.python.keras.layers.core import Lambda
from tensorflow.python.keras.layers.recurrent_v2 import LSTM

from carl.agents.tensorflow.memory import Memory
from copy import deepcopy