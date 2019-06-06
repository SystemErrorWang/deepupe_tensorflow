import os
import cv2
import tensorflow as tf
import numpy as np
from network import inference
from dataset import get_test_loader
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"]="0"


def test():
    

    input_image = tf.placeholder(tf.float32, [None, None, None, 3])
    result_image = inference(input_image, name='generator') 
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    saver = tf.train.Saver()
    
    if not os.path.exists('results'):
        os.mkdir('results')
        
    with tf.device('/device:GPU:0'):

        data_dir = 'data_loacation_in_your_computer'
        dataloader = get_test_loader(data_dir)

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint('saved_models'))
        
        for idx, batch in tqdm(enumerate(dataloader)):
            result = sess.run([result_image], feed_dict={input_image: batch[0]})
            result = np.squeeze(result)*255
            result = np.clip(result, 0, 255).astype(np.float32)
            ground_truth = np.squeeze(batch[1])
            save_out_path = os.path.join('results', '{}.jpg'.format(str(idx).zfill(4)))
            save_gt_path = os.path.join('results', '{}_gt.jpg'.format(str(idx).zfill(4)))
            cv2.imwrite(save_out_path, result)
            cv2.imwrite(save_gt_path, ground_truth)
            
if __name__ == '__main__':
    test()
   