import tensorflow as tf
import numpy as np

from nbt_pilot.utils.tob_utils import decode_image
from nbt_pilot.utils.preprocess_utils import thermal_image_preprocessing

# =============================================================================
# Loaders
# =============================================================================

def thermal_image_dataset(file_paths, params, device):
        
    with tf.device(device):
        
        # Create a tf.data.Dataset from the file paths
        dataset = tf.data.Dataset.from_tensor_slices(file_paths)
        
        dataset = dataset.map(lambda x: tf.py_function(decode_image, [x], tf.int16))
        
        # Image preprocessing  
        dataset = dataset.map(lambda x: 
                              thermal_image_preprocessing(
                                    x,
                                    params["mode"],
                                    params["gmm"],
                                    params["room"]),
                              num_parallel_calls=tf.data.AUTOTUNE)
                                      
        # Define batch size
        dataset = dataset.batch(params["batch_size"], drop_remainder=False)
        
        # Prefetch
        dataset = dataset.prefetch(tf.data.AUTOTUNE)  
        
    return dataset

