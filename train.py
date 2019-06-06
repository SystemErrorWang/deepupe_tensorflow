import os
import math
import tensorflow as tf
import numpy as np
from network import inference
from dataset import get_train_loader
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"]="0"


def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator


def color_loss(image, label, len_reg=0):
    
    vec1 = tf.reshape(image, [-1, 3])
    vec2 = tf.reshape(label, [-1, 3])
    clip_value = 0.999999
    norm_vec1 = tf.nn.l2_normalize(vec1, 1)
    norm_vec2 = tf.nn.l2_normalize(vec2, 1)
    dot = tf.reduce_sum(norm_vec1*norm_vec2, 1)
    dot = tf.clip_by_value(dot, -clip_value, clip_value)
    angle = tf.acos(dot) * (180/math.pi)

    return tf.reduce_mean(angle)


def smoothness_loss(image):
    clip_low, clip_high = 0.000001, 0.999999
    image = tf.clip_by_value(image, clip_low, clip_high)
    image_h, image_w = tf.shape(image)[1], tf.shape(image)[2]
    tv_x = tf.reduce_mean((image[:, 1:, :, :]-image[:, :image_h-1, :, :])**2)
    tv_y = tf.reduce_mean((image[:, :, 1:, :]-image[:, :, :image_w-1, :])**2)
    total_loss = (tv_x + tv_y)/2
    '''
    log_image = tf.log(image)
    log_tv_x = tf.reduce_mean((log_image[:, 1:, :, :]-
                              log_image[:, :image_h-1, :, :])**1.2)
    log_tv_y = tf.reduce_mean((log_image[:, :, 1:, :]-
                               log_image[:, :, :image_w-1, :])**1.2)
    total_loss = tv_x / (log_tv_x + 1e-4) + tv_y / (log_tv_y + 1e-4)
    '''
    return total_loss
   
    

def reconstruct_loss(image, label):
    l2_loss = tf.reduce_mean(tf.square(label-image))
    return l2_loss
    

def cal_psnr(pred, label):
    label_tmp, pred_tmp = label*255, pred*255
    mse = tf.reduce_mean(tf.squared_difference(label_tmp, pred_tmp))
    mse = tf.cast(mse, tf.float32)
    train_psnr = tf.constant(10, dtype=tf.float32)*\
    		log10(tf.constant(255**2, dtype=tf.float32)/mse)
    return train_psnr


def train():
    total_epoch, total_iter = 100, 0
    best_loss, init_lr = 1e10, 5e-5
    batch_size, image_h, image_w = 8, 512, 512


    image = tf.placeholder(tf.float32, [None, image_h, image_w, 3])
    label = tf.placeholder(tf.float32, [None, image_h, image_w, 3])
    lr = tf.placeholder(tf.float32)
    
    pred = inference(image, width=0.75, is_training=True)
    c_loss = color_loss(pred, label)
    s_loss = smoothness_loss(pred)
    r_loss = reconstruct_loss(pred, label)
    total_loss = 1e-2*c_loss + 1e2*s_loss + r_loss
    #total_loss = c_loss + r_loss
    
    all_vars = tf.trainable_variables()
    backbone_vars = [var for var in all_vars if 'backbone' in var.name]
    train_psnr = cal_psnr(pred, label)
    

    tf.summary.scalar('loss', total_loss)
    tf.summary.scalar('color_loss', c_loss)
    tf.summary.scalar('smoothness_loss', s_loss)
    tf.summary.scalar('reconstruct_loss', r_loss)
    tf.summary.scalar('psnr', train_psnr)

    
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = optimizer.minimize(total_loss)
    train_op = tf.group([train_op, update_ops])
    
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    '''
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    '''
    train_writer = tf.summary.FileWriter('train_log', sess.graph)
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()

    with tf.device('/device:GPU:0'):

        sess.run(tf.global_variables_initializer())
        
        weight = np.load('mobilenetv2_075.npy', allow_pickle=True)
        assign_ops = []
        for var, para in zip(backbone_vars, weight):
            assign_ops.append(var.assign(para))
        sess.run(assign_ops)
        
        
        data_dir = 'data_loacation_in_your_computer'
        dataloader = get_train_loader((image_h, image_w), batch_size, data_dir)
    
        for epoch in range(total_epoch):
            for batch in tqdm(dataloader):
                total_iter += 1
                
                _, train_info, loss = sess.run([train_op, summary_op, total_loss], 
                                                  feed_dict={image: batch[0], 
                                                             label: batch[1], 
                                                             lr: init_lr})      
                train_writer.add_summary(train_info, total_iter)
            
                if np.mod(total_iter, 20) == 0:
                    print('{}th epoch, {}th iter, loss: {}'.format(epoch, total_iter, loss))
                    if loss < best_loss:
                        best_loss = loss
                        saver.save(sess, 'saved_models/model', global_step=total_iter)
                
                
                    
            
if __name__ == '__main__':
    train()
   
    #test()