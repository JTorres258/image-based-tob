# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:37:30 2024

@author: 2924443
"""

import os
import time
from absl import app
from absl import flags
import numpy as np
import json
import time
import tensorflow as tf
import cv2

from utils.model_utils import CNN_load
from utils.model_utils import GradCAMConv2D
from utils.tob_utils import get_GMM_params
from utils.tob_utils import create_timeline
from utils.tob_utils import fix_thermal_video
from utils.data_loader import thermal_image_dataset

flags.DEFINE_bool("GradCAM", False, "Extract GradCAM from images.")
flags.DEFINE_enum("preprocess_device", 'CPU', ['CPU', 'GPU'],
        "Device used for preprocessing.")
FLAGS = flags.FLAGS

os.chdir(os.path.dirname(os.path.realpath(__file__)))

###############################################################################


def main(argv):
    del argv
    
    # Select device
    device_mapping = {'GPU': '/gpu:0', 
                      'CPU': '/cpu:0'}
    device = device_mapping.get(FLAGS.preprocess_device)

    # Prepare input data
    label_map = './model/label_map.json'
    input_shape = (252, 336)   # NN input  
        
    # Randomizing the state
    tf.keras.utils.set_random_seed(0)

    # Load classes to be used and order
    with open(label_map) as json_file:
        class_map = json.load(json_file)
    classes = list(class_map.keys())
    num_classes = len(classes)

    # Load RGB model
    model_name = [x for x in os.listdir('./model') if '.h5' in x][0]
    path_model = './model/' + model_name

    model = CNN_load(path_model, input_shape+(3,), len(classes))
        
    # Find video
    path_video = '../data/raw_images'
    list_raw = sorted(os.listdir(path_video))
    
    # Fix missing frames (autocalibration)
    timestamps = [int(x.split('ms')[0].split('_')[1]) for x in list_raw]
    new_raw, _ = fix_thermal_video(list_raw, timestamps, False)
    new_raw = [os.path.join(path_video, x) for x in new_raw] # Full path

    # Read until video is completed
    pred_label_list = []
    prob_label = []

    gmm_start_time = time.time()
    with tf.device(device):
        if 'gauss' in model_name:
            means, covs, weights, data_info = get_GMM_params(
                path_video, n_components=3)
            x_mean, x_var, x_min, x_max = data_info
            gmm_params = (means, covs, weights, x_var)
            mode = 'gmm'
        else:
            gmm_params = (None, None, None, None)
            mode = 'max-min'
    print(f'Time GMM: {time.time() - gmm_start_time:.2f} seconds')
        
    # Create data loader
    data_params = {
            "mode" : mode,
            "gmm" : gmm_params,
            "room" : 'Labour',
            "batch_size": 16,
            }
    
    dataset = thermal_image_dataset(new_raw, data_params, device)

    # Start timer
    process_start_time = time.time()

    # Inference
    prob_label = model.predict(dataset, verbose=1)
    
    # Find index with highest probability
    predictions = tf.argmax(prob_label, axis=1)

    # Convert index to label
    pred_label_list = [classes[i] for i in predictions]
              
    print(f'Time elapsed: {time.time() - process_start_time:.2f} seconds')
    print(f'Avg FPS: {len(new_raw)/(time.time() - process_start_time):.2f}')

    # Write predictions
    output_path = './output/output_simulation.txt'
    with open(output_path, 'w', encoding='utf8') as f:
        f.write('\n'.join('%s\t%s' % p for p in zip(pred_label_list, prob_label)))
    
    create_timeline(
        output_path=output_path,
        anns_path= '../data/anns.txt',
        save_path='./output/ToB_example.png')   
        
    # GradCAM
    if FLAGS.GradCAM:
        
        # Prepare video object
        fourcc = cv2.VideoWriter_fourcc('M','P','4','V')
        fps = 8.33
        vid = cv2.VideoWriter('./output/gradcam.mp4', fourcc, fps, (336, 252), True) # Use color
    
        for batch_img in dataset:
            
            # Evaluate each batch
            preds = model.predict(batch_img)
            pred_idx = np.argmax(preds, axis=1)
            
            # Estimate the heatmap
            cam = GradCAMConv2D(model, pred_idx, layerName=None)
            heatmap = cam.compute_heatmap(batch_img)
            
            # Scale the batch and overlay
            batch_scale = tf.cast((batch_img*255),'uint8').numpy()
            (heatmap, output_heatmap) = cam.overlay_heatmap(heatmap, batch_scale, alpha=0.5, colormap=cv2.COLORMAP_JET)
            
            # Write in the video object
            for map_ in output_heatmap:
                vid.write(map_)

        vid.release() 
        
        
if __name__ == '__main__':
    app.run(main)
