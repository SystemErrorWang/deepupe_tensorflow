import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
    

def init_conv(x, out_channel, scope='init_conv', is_training=True):
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.convolution2d],
                            normalizer_fn=slim.batch_norm, 
                            activation_fn=tf.nn.relu6):
            with slim.arg_scope([slim.batch_norm],
                                  is_training=is_training, 
                                  center=True, scale=True):
                x = slim.convolution2d(x, out_channel, [3, 3], stride=2, padding='same', 
                                biases_initializer=None,  biases_regularizer=None)
                #x = slim.batch_norm(x, scale=True, is_training=is_training)
                return x



def init_resblock(x, out_channel, stride, 
                  scope='init_resblock', is_training=True):
    with tf.variable_scope(scope):
        
        with slim.arg_scope([slim.separable_convolution2d],
                            normalizer_fn=slim.batch_norm, 
                            activation_fn=tf.nn.relu6):
            with slim.arg_scope([slim.batch_norm],
                                  is_training=is_training, 
                                  center=True, scale=True):
                x = slim.separable_convolution2d(x, None, [3, 3], depth_multiplier=1, stride=stride, 
                                                 biases_initializer=None, biases_regularizer=None)
                
        with slim.arg_scope([slim.convolution2d],
                            normalizer_fn=slim.batch_norm):
            with slim.arg_scope([slim.batch_norm],
                                  is_training=is_training, 
                                  center=True, scale=True):
                x = slim.convolution2d(x, out_channel, [1, 1], 
                                    stride=1, padding='same', 
                                    biases_initializer=None, 
                                    biases_regularizer=None)
                
                return x 


def inverte_resblock(x, in_channel, out_channel, stride, expand_radio=6, 
                    res_connect=True, scope='inverte_resblock', is_training=True):
    mid_channel = in_channel * expand_radio
    with tf.variable_scope(scope):
        if res_connect:
            short_cut = x
            
        with slim.arg_scope([slim.convolution2d],
                            normalizer_fn=slim.batch_norm, 
                            activation_fn=tf.nn.relu6):
            with slim.arg_scope([slim.batch_norm],
                                  is_training=is_training, 
                                  center=True, scale=True):
                x = slim.convolution2d(x, mid_channel, [1, 1], stride=1, padding='same', 
                                biases_initializer=None, biases_regularizer=None)
                
        with slim.arg_scope([slim.separable_convolution2d],
                            normalizer_fn=slim.batch_norm, 
                            activation_fn=tf.nn.relu6):
            with slim.arg_scope([slim.batch_norm],
                                  is_training=is_training, 
                                  center=True, scale=True):
                x = slim.separable_convolution2d(x, None, [3, 3], depth_multiplier=1, stride=stride, 
                                                 biases_initializer=None, biases_regularizer=None)
        
        with slim.arg_scope([slim.convolution2d],
                            normalizer_fn=slim.batch_norm):
            with slim.arg_scope([slim.batch_norm],
                                  is_training=is_training, 
                                  center=True, scale=True):
                x = slim.convolution2d(x, out_channel, [1, 1], 
                                stride=1, padding='same', 
                                biases_initializer=None, 
                                biases_regularizer=None)
               
                if res_connect:
                    return x + short_cut
                else:
                    return x


def backbone(inputs, width=0.75, scope='backbone', is_training=True, reuse=False):
   
    if width == 1:
        channel = np.array([32, 16, 24, 32, 64, 96, 160, 320])
    elif width == 0.75:
        channel = np.array([24, 16, 24, 24, 48, 72, 120, 240])
    elif width == 0.5:
        channel = np.array([16, 8, 16, 16, 32, 48, 80, 160])
    
    with tf.variable_scope(scope, reuse=reuse):
        
        x = init_conv(inputs, channel[0], is_training=is_training)
        x1_out = init_resblock(x, channel[1], stride=1, is_training=is_training)

        x = inverte_resblock(x1_out, in_channel=channel[1], out_channel=channel[2], stride=2, 
                             res_connect=False, scope='invert1_1', is_training=is_training)
        x2_out = inverte_resblock(x, in_channel=channel[2], out_channel=channel[2], stride=1, 
                                  res_connect=True, scope='invert1_2', is_training=is_training)

        x = inverte_resblock(x2_out, in_channel=channel[2], out_channel=channel[3], stride=2, 
                             res_connect=False, scope='invert2_1', is_training=is_training)
        x = inverte_resblock(x, in_channel=channel[3], out_channel=channel[3], stride=1, 
                             res_connect=True, scope='invert2_2', is_training=is_training)
        x3_out = inverte_resblock(x, in_channel=channel[3], out_channel=channel[3], stride=1, 
                                  res_connect=True, scope='invert2_3', is_training=is_training)

        x = inverte_resblock(x3_out, in_channel=channel[3], out_channel=channel[4], stride=2, 
                             res_connect=False, scope='invert3_1', is_training=is_training)
        for i in range(3):
            x = inverte_resblock(x, in_channel=channel[4], out_channel=channel[4], stride=1, 
                                 res_connect=True, scope='invert3_{}'.format(i+2), 
                                 is_training=is_training)

        x = inverte_resblock(x, in_channel=channel[4], out_channel=channel[5], stride=1, 
                             res_connect=False, scope='invert4_1', is_training=is_training)
        x = inverte_resblock(x, in_channel=channel[5], out_channel=channel[5], stride=1, 
                             res_connect=True, scope='invert4_2', is_training=is_training)
        
        x4_out = inverte_resblock(x, in_channel=channel[5], out_channel=channel[5], stride=1, 
                                  res_connect=True, scope='invert4_3', is_training=is_training)
    
    
        x = inverte_resblock(x4_out, in_channel=channel[5], out_channel=channel[6], stride=2, 
                             res_connect=False, scope='invert5_1', is_training=is_training)
        
        for j in range(2):
            x = inverte_resblock(x, in_channel=channel[6], out_channel=channel[6], 
                        stride=1, res_connect=True, scope='invert5_{}'.format(j+2), 
                        is_training=is_training)
        
        x5_out = inverte_resblock(x, in_channel=channel[6], out_channel=channel[7], stride=1, 
                                  res_connect=False, scope='invert5_4', is_training=is_training)
        
        return x5_out


if __name__ == '__main__':

    
    inputs = tf.placeholder(tf.float32, [1, 256, 256, 3])
    outputs = backbone(inputs, 0.5)
    for xxx in outputs:
        print(xxx.get_shape().as_list())
    
    '''
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    converter = tf.lite.TFLiteConverter.from_session(
                            sess, [inputs], [outputs])
    
    converter.default_ranges_stats=(0, 6)
    converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
    input_arrays = converter.get_input_arrays()
    converter.quantized_input_stats = {input_arrays[0] : (0., 1.)}  # mean, std_dev
    tflite_model = converter.convert()
    open("narrow_unet.tflite", "wb").write(tflite_model)
    '''

    '''
    adb push narrow_unet.tflite /data/local/tmp
    adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/narrow_unet.tflite --num_threads=1
    '''
    
  
    
    
   
    
    


    

