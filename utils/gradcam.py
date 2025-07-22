import tensorflow as tf
import cv2
import numpy as np

def generate_gradcam(model, img_array, layer_name='conv5_block3_out'):
    grad_model = tf.keras.models.Model([model.inputs],
                                       [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_output, prediction = grad_model(img_array)
        loss = prediction[:, tf.argmax(prediction[0])]
    grads = tape.gradient(loss, conv_output)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.dot(conv_output[0], weights.numpy())
    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / cam.max()
    return heatmap