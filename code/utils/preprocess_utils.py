import tensorflow as tf

def thermal_image_preprocessing(img, mode, gmm_params, room):
    
    # Implementation of GMM in TF
    means, covs, weights, data_var = gmm_params
    
    converted_img = tf.cast(img, tf.float64) * 0.04 - 273.15
    
    # Criterion for selecting the gaussian component
    idx_ = tf.logical_and(covs <= 2 * data_var, weights > 0.15)
    mean_ = tf.reduce_max(tf.boolean_mask(means, idx_)) 
    
    # Define range of temperatures
    if room == 'Operation':
        coeff_l, coeff_h = -2.5, 12.5
    else:
        coeff_l, coeff_h = -5, 10
    
    tem_low = mean_ + coeff_l 
    tem_high = mean_ + coeff_h 
    
    # Find values out of the range
    idx_low = converted_img < tem_low
    idx_high = converted_img > tem_high
    
    # Convert image
    converted_img = (converted_img - tem_low) / (tem_high - tem_low)
    converted_img = tf.where(idx_low, tf.zeros_like(converted_img), converted_img)
    converted_img = tf.where(idx_high, tf.ones_like(converted_img), converted_img)

    # Add color channels
    frame = tf.expand_dims(converted_img, axis=[-1])
    frame = tf.repeat(frame, repeats=3, axis=2)

    # Normalization to [-1, 1]
    frame = tf.cast(frame, tf.float32) * (1./0.5)-1
    
    return frame
