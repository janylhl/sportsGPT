"""
Main script of the repo
"""

import tensorflow as tf


# Checking GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
