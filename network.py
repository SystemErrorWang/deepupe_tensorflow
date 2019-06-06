import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from mobilenetv2 import backbone
from utils import apply_bilateral_grid
from tensorflow.python.framework import graph_util


def get_tensor_shape(x):
    a = x.get_shape().as_list()
    b = [tf.shape(x)[i] for i in range(len(a))]
    return [aa if type(aa) is int else bb for aa, bb in zip(a, b)]



def coef_mobilenetv2(inputs, width=0.75, luma_bins=8, is_training='False', name='coefficients'):
    with tf.variable_scope(name):
        
        with slim.arg_scope([slim.separable_convolution2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu6):
            with slim.arg_scope([slim.batch_norm], 
                                is_training=is_training, center=True, scale=True):
        
                x = backbone(inputs, width=width, is_training=is_training)
                
                for _ in range(2):
                    x = slim.convolution2d(x, 48, [3, 3], stride=1)
                    
                pool = tf.reduce_mean(x, axis=[1, 2], keepdims=False)
                
                fc1 = slim.fully_connected(pool, 192)
                
                fc2 = slim.fully_connected(fc1, 96)
                
                fc3 = slim.fully_connected(fc2, 48)
                
                feat1 = slim.convolution2d(x, 48, [3, 3], stride=1)
                
                feat2 = slim.convolution2d(feat1, 48, [3, 3], stride=1,
                                           normalizer_fn=None, activation_fn=None)
                
                bs, ch = tf.shape(fc3)[0], tf.shape(fc3)[1]
                fc_reshape = tf.reshape(fc3, [bs, 1, 1, ch])
                fusion = tf.nn.relu6(feat2 + fc_reshape)
                
                conv7 = slim.convolution2d(fusion, 24*luma_bins, [1, 1], stride=1,
                                           normalizer_fn=None, activation_fn=None)
                
                stack1 = tf.stack(tf.split(conv7, 24, axis=3), axis=4)
                stack2 = tf.stack(tf.split(stack1, 4, axis=4), axis=5)
                #print(stack2.get_shape().as_list())
                # [1, 16, 16, 8, 9, 4]
                b, h, w, ch1, ch2, ch3 = get_tensor_shape(stack2)
                stack2 = tf.reshape(stack2, [b, h, w, ch1*ch2*ch3])
                return stack2
            
            
            
'''
def coefficients(inputs, luma_bins=8, is_training='False', name='coefficients'):
    with tf.variable_scope(name):
        
        with slim.arg_scope([slim.separable_convolution2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu6):
            with slim.arg_scope([slim.batch_norm], 
                                is_training=is_training, center=True, scale=True):
        
                x = slim.convolution2d(inputs, 32, [3, 3], stride=2)
                
                x = slim.convolution2d(x, 64, [3, 3], stride=2)
                
                x = slim.convolution2d(x, 96, [3, 3], stride=2)
                
                x = slim.convolution2d(x, 128, [3, 3], stride=2)
                
                
                conv4 = x
                
                for _ in range(2):
                    x = slim.convolution2d(x, 48, [3, 3], stride=2)
                    
                
                pool = tf.reduce_mean(x, axis=[1, 2], keepdims=False)
                
                fc1 = slim.fully_connected(pool, 192)
                
                fc2 = slim.fully_connected(fc1, 96)
                
                fc3 = slim.fully_connected(fc2, 48)
                
                conv5 = slim.convolution2d(conv4, 48, [3, 3], stride=1)
                
                conv6 = slim.convolution2d(conv5, 48, [3, 3], stride=1,
                                           normalizer_fn=None, activation_fn=None)
                
                bs, ch = tf.shape(fc3)[0], tf.shape(fc3)[1]
                fc_reshape = tf.reshape(fc3, [bs, 1, 1, ch])
                fusion = tf.nn.relu6(conv6 + fc_reshape)
                
                conv7 = slim.convolution2d(fusion, 24*luma_bins, [1, 1], stride=1,
                                           normalizer_fn=None, activation_fn=None)
                
                stack1 = tf.stack(tf.split(conv7, 24, axis=3), axis=4)
                stack2 = tf.stack(tf.split(stack1, 4, axis=4), axis=5)
                #print(stack2.get_shape().as_list())
                # [1, 16, 16, 8, 9, 4]
                b, h, w, ch1, ch2, ch3 = get_tensor_shape(stack2)
                stack2 = tf.reshape(stack2, [b, h, w, ch1*ch2*ch3])
                return stack2
'''
            

def guide(inputs, is_training=False, name='guide'):
    with tf.variable_scope(name):
        in_ch = inputs.get_shape().as_list()[-1]
        idtity = np.identity(in_ch, dtype=np.float32)\
                    + np.random.randn(1).astype(np.float32)*1e-4
        ccm = tf.get_variable('ccm', dtype=tf.float32, initializer=idtity)   
        ccm_bias = tf.get_variable('ccm_bias', shape=[in_ch,], dtype=tf.float32, 
                                   initializer=tf.constant_initializer(0.0))

        guidemap = tf.matmul(tf.reshape(inputs, [-1, in_ch]), ccm)
        guidemap = tf.nn.bias_add(guidemap, ccm_bias, name='ccm_bias_add')
        guidemap = tf.reshape(guidemap, tf.shape(inputs))             
        
        shifts_ = np.linspace(0, 1, 16, endpoint=False, dtype=np.float32)
        shifts_ = shifts_[np.newaxis, np.newaxis, np.newaxis, :]
        shifts_ = np.tile(shifts_, (1, 1, in_ch, 1))

        guidemap = tf.expand_dims(guidemap, 4)
        shifts = tf.get_variable('shifts', dtype=tf.float32, initializer=shifts_)

        slopes_ = np.zeros([1, 1, 1, in_ch, 16], dtype=np.float32)
        slopes_[:, :, :, :, 0] = 1.0
        slopes = tf.get_variable('slopes', dtype=tf.float32, initializer=slopes_)

        guidemap = tf.reduce_sum(slopes*tf.nn.relu6(guidemap-shifts), reduction_indices=[4])
        guidemap = slim.convolution2d(guidemap, 1, [1, 1], activation_fn=None,
                                weights_initializer=tf.constant_initializer(1.0/in_ch))
        guidemap = tf.clip_by_value(guidemap, 0, 1)
        guidemap = tf.squeeze(guidemap, squeeze_dims=[3,])
        
        return guidemap
    
    

def inference(hr_input, width=0.75, lr_size=(256, 256), is_training=False, name='inference'):
    with tf.variable_scope(name):
        
        lr_input = tf.image.resize_images(hr_input, lr_size,
                                          tf.image.ResizeMethod.BILINEAR)
        coeffs = coef_mobilenetv2(lr_input, width=width, is_training=is_training)
        #coeffs = coeffient(lr_input, is_training=is_training)
        guidemap = guide(hr_input, is_training=is_training)
        output = apply_bilateral_grid(coeffs, guidemap, hr_input)
        return output
        

            
        
if __name__ == '__main__':

    hr_input = tf.placeholder(tf.float32, [1, 1024, 1024, 3])
    outputs = inference(hr_input, is_training=False)
    print(outputs.get_shape().as_list())
    
