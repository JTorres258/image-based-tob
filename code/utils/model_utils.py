import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
import numpy as np
from scipy.ndimage import zoom
import cv2

# =============================================================================
# Models
# =============================================================================

def CNN_load(sPath:str, tuImageShape:tuple, nClasses:int) -> Model:
    """ Keras load_model plus input/output shape checks
    """
    
    # Extra Metrics
    mcc = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=nClasses)
    
    print("\nLoad trained I3D model from %s ..." % sPath)
    keModel = load_model(sPath, custom_objects={'RMSpropW':tfa.optimizers.extend_with_decoupled_weight_decay(tf.keras.optimizers.RMSprop),
                                                'AdamW':tfa.optimizers.AdamW,
                                                'SGDW':tfa.optimizers.SGDW})
    
    tuInputShape = keModel.input_shape[1:]
    tuOutputShape = keModel.output_shape[1:]
    
    print("Loaded input shape %s, output shape %s" % (str(tuInputShape), str(tuOutputShape)))

    assert tuInputShape == tuImageShape, "Unexpected input shape"
    assert tuOutputShape == (nClasses,), "Unexpected output shape"

    return keModel
    

class GradCAMConv2D:
    def __init__(self, model, classIdx, layerName, NestedLayer=None):
		# store the model, the class index used to measure the class
		# activation map, the layer to be used when visualizing
		# the class activation map and any nested model
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        self.NestedLayer = NestedLayer
        
        # Find nested layers
        self.nested_model = self.model.get_layer(self.NestedLayer) \
            if self.NestedLayer else None
        
        # if the layer name is None, attempt to automatically find
		# the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()
            
        
    def find_target_layer(self):
		# attempt to find the final convolutional layer in the network
		# by looping over the layers of the network in reverse order
        model_ = self.model if not self.NestedLayer else self.nested_model
        for layer in reversed(self.model.layers):
			# check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
		# otherwise, we could not find a 4D layer so the GradCAM
		# algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
		        
                
    def compute_heatmap(self, image, eps=1e-8):
		# construct our gradient model by supplying (1) the inputs
		# to our pre-trained model, (2) the output of the (presumably)
		# final 4D layer in the network, and (3) the output of the
		# softmax activations from the model
        model_ = self.model if not self.NestedLayer else self.nested_model

        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[model_.get_layer(self.layerName).output,
                self.model.output])

        gradModel.layers[-1].activation = tf.keras.activations.linear
        
        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
			# cast the image tensor to a float-32 data type, pass the
			# image through the gradient model, and grab the loss
			# associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = [predictions[:, i] for i in self.classIdx]
            
		# use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, grads.dtype)
        castGrads = tf.cast(grads > 0, grads.dtype)
        guidedGrads = castConvOutputs * castGrads * grads
        
		# the convolution and guided gradients have a batch dimension
		# (which we don't need) so let's grab the volume itself and
		# discard the batch
        convOutputs = convOutputs
        guidedGrads = guidedGrads

        # compute the average of the gradient values, and using them
		# as weights, compute the ponderation of the filters with
		# respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        heatmap = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
        heatmap = tf.cast(heatmap, "float32").numpy()

        # grab the spatial dimensions of the input image and resize
		# the output class activation map to match the input image
		# dimensions
        b, h, w, c = image.shape
        
        heatmap_zoom = np.zeros((b, h, w))
        for i in range(b):
            heatmap_zoom[i] = zoom(heatmap[i], [h/heatmap[i].shape[0], 
                            w/heatmap[i].shape[1]]
                            )
            
            # normalize the heatmap such that all values lie in the range
            # [0, 1], scale the resulting values to the range [0, 255],
            # and then convert to an unsigned 8-bit integer
            heat_max, heat_min = np.max(heatmap_zoom[i]), np.min(heatmap_zoom[i])
            numer = heatmap_zoom[i] - heat_min
            denom = (heat_max - heat_min) + eps
            heatmap_zoom[i] = numer / denom
            heatmap_zoom[i] = (heatmap_zoom[i] * 255)

		# return the resulting heatmap to the calling function
        return heatmap_zoom.astype("uint8")
        

    def overlay_heatmap(self, heatmap, image, alpha=0.5,
        colormap=cv2.COLORMAP_VIRIDIS):
		# apply the supplied color map to the heatmap and then
		# overlay the heatmap on the input image

        color_heatmap = np.zeros_like(image)
        output = np.zeros_like(image)
        for i in range(len(heatmap)):
            if image[i].shape[-1] == 1:
                image[i] = cv2.cvtColor(image[i], cv2.COLOR_GRAY2RGB)

            color_heatmap[i] = cv2.applyColorMap(heatmap[i], colormap)
            output[i] = cv2.addWeighted(image[i], alpha, color_heatmap[i], 1 - alpha, 0)
		# return a 2-tuple of the color mapped heatmap and the output,
		# overlaid image

        return (color_heatmap, output)
