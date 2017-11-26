import tensorflow as tf
import os, sys
slim = tf.contrib.slim
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nets import model_cmc
import os
from data_operations_lite import read_data_from_folder
import matplotlib.pyplot as plt
from sklearn import metrics
import scipy.io as sio

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"


tf.app.flags.DEFINE_boolean(
    'post_processing', False, 'post-processing')
tf.app.flags.DEFINE_string(
    'checkpoint_dir', 'Logs_cmc/', 'path to checkpoint')
tf.app.flags.DEFINE_string(
    'model', 'unet', 'Model to eval')
FLAGS = tf.app.flags.FLAGS
# checkpoint_dir = './tmp/'
checkpoint_dir = 'Logs2/'
datasetDir = 'validation_original_cmc/'

import numpy as np

def compute_iou(confmat_log):
    iou_log = np.zeros((confmat_log.shape[0], 1))
    for j in range(confmat_log.shape[0]):
        true_positive = confmat_log[j, j]
        row_sum = np.sum(confmat_log[j, :])
        col_sum = np.sum(confmat_log[:, j])
        cur_iou = true_positive / float(col_sum + row_sum - true_positive)
        iou_log[j] = cur_iou

    return iou_log

def normalize_streams(data):
    mu_0, std_0 = np.mean(data[:, :, :,:, 0]), np.std(data[:, :, :,:, 0])
    mu_1, std_1 = np.mean(data[:, :, :,:, 1]), np.std(data[:, :, :,:, 1])
    mu_2, std_2 = np.mean(data[:, :, :,:, 2]), np.std(data[:, :, :,:, 2])
    mu_3, std_3 = np.mean(data[:, :, :,:, 3]), np.std(data[:, :, :,:, 3])
    data[:, :, :,:, 0] = (data[:, :, :,:, 0] - mu_0) / std_0
    data[:, :, :,:, 1] = (data[:, :, :,:, 1] - mu_1) / std_1
    data[:, :, :,:, 2] = (data[:, :, :,:, 2] - mu_2) / std_2
    data[:, :, :,:, 3] = (data[:, :, :,:, 3] - mu_3) / std_3

    return data

num_train = 7
num_val = 1
num_test = 1
output_window_size = 240

dataset_path = 'masks_image_pairs/'

trX,trY, experiment_sizes = read_data_from_folder(dataset_path)
train_sample_size = sum(experiment_sizes[0:num_train])

tr_range = np.arange(train_sample_size)
np.random.shuffle(tr_range)
trX[0:train_sample_size] = trX[tr_range]
trY[0:train_sample_size] = trY[tr_range]

trX = normalize_streams(trX)

batchsize = 1

print experiment_sizes
print sum(experiment_sizes[0:num_train])
print sum(experiment_sizes[0:num_train+num_val])
print sum(experiment_sizes[0:num_train+num_val+num_test])

T = 3
target = 5
num_steps = 30
stride = (T - 1)/2

img_input = tf.placeholder(tf.float32, shape=(batchsize, T, output_window_size, output_window_size, 4))
la_input = tf.placeholder(tf.int32, shape=(batchsize, T, output_window_size, output_window_size, 1))
is_training = tf.placeholder(tf.bool)
la_input_onehot = la_input #tf.one_hot(la_input,2)

net = model_cmc.Model()
logits, _ = net.net(img_input, is_training)
logits_packed = tf.stack(logits)
logits_softmaxed = tf.nn.softmax(logits)
logits_amax = tf.squeeze(tf.argmax(logits_packed, axis=4))
loss_op, mean_iou_op, growth = net.weighted_losses_growth_term(logits, la_input_onehot)
mean_iou_scalar = mean_iou_op[0]
optimizer = tf.train.AdamOptimizer(1e-03) # 1e-03 was good
#train_step = optimizer.minimize(loss_op)

train_step = slim.learning.create_train_op(loss_op,optimizer)

saver = tf.train.Saver()

config = tf.ConfigProto(device_count = {'GPU': 3}, allow_soft_placement=True)

init_op = tf.initialize_all_variables()
init_loc = tf.initialize_local_variables()

with tf.Session(config=config) as sess:
    #dir = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    #saver.restore(sess, dir)
    #sess.run(tf.local_variables_initializer())
    #print("Model restore!")
    sess.run(init_loc)
    sess.run(init_op)

    val_log = []

    current_time_str = str(time.time())
    max_val_accuracy = 0

    for step in range(num_steps):
        lss_log = []
        print step
        for i in range(0,sum(experiment_sizes[0:num_train]),batchsize):
            image_batch = np.reshape(trX[i:(i+batchsize),target-stride:target+stride+1,:,:,:],[batchsize,T,output_window_size,output_window_size,4]) #np.random.random((1, T, 240, 240, 4))
            label_batch = np.reshape(trY[i:(i+batchsize),target-stride:target+stride+1,:,:,:],[batchsize,T,output_window_size,output_window_size,1]) #np.random.random((1, T, 240, 240, 1))


            grwth, _, sample_logits, lss, iou, current_logits_amax = sess.run([growth,train_step,logits_softmaxed,loss_op,mean_iou_scalar, logits_amax], feed_dict={
                img_input: image_batch, la_input: label_batch,
                is_training: True})
            #print sample_logits

            #lss_log.append(lss)
            #print lss

            #lss, sample_logits, current_logits_amax = sess.run([loss_op, logits_packed, logits_amax], feed_dict={
            #    img_input: image_batch, la_input: label_batch,
            #    is_training: False})
            #print lss

        #print np.mean(lss_log)
        if step > -1: # set this to some non-negative value to skip first for n iterations without doing val and test
            confmat_log = np.zeros((2,2))
            for k in range(sum(experiment_sizes[0:num_train]),sum(experiment_sizes[0:num_train+num_val]),batchsize):
                image_batch = np.reshape(trX[k:k+batchsize,target-stride:target+stride+1,:,:,:],[batchsize,T,output_window_size,output_window_size,4]) #np.random.random((1, T, 240, 240, 4))
                label_batch = np.reshape(trY[k:k+batchsize,target-stride:target+stride+1,:,:,:],[batchsize,T,output_window_size,output_window_size,1]) #np.random.random((1, T, 240, 240, 1))

                lss, sample_logits, current_logits_amax = sess.run([loss_op,logits_packed, logits_amax], feed_dict={
                    img_input: image_batch, la_input: label_batch,
                    is_training: False})

                sample_logits = sample_logits[stride]

                if T > 1:

                    current_logits_amax = current_logits_amax[stride]

                label_batch = label_batch[0,stride]


                label_batch_calc = np.squeeze(label_batch)

                confmat = metrics.confusion_matrix(np.ndarray.flatten(label_batch_calc.astype(dtype=np.int64)),np.ndarray.flatten(current_logits_amax))

                confmat_log += confmat

            iou_log = compute_iou(confmat_log)

            current_val_acc = np.mean(iou_log)
            print current_val_acc

            if current_val_acc > max_val_accuracy:
                max_val_accuracy = current_val_acc

                test_confmat_log = np.zeros((2,2))

                image_batch_log = np.zeros((sum(experiment_sizes[0:num_train+num_val + num_test])-sum(experiment_sizes[0:num_train+num_val]) + 1,1,output_window_size,output_window_size,4))
                label_log = np.zeros((sum(experiment_sizes[0:num_train+num_val + num_test])-sum(experiment_sizes[0:num_train+num_val]) + 1,1,output_window_size,output_window_size,1))
                preds_log = np.zeros((sum(experiment_sizes[0:num_train+num_val + num_test])-sum(experiment_sizes[0:num_train+num_val]) + 1, 1, output_window_size, output_window_size, 1))

                for k in range(sum(experiment_sizes[0:num_train+num_val]),sum(experiment_sizes[0:num_train+num_val + num_test]),batchsize):
                    image_batch = np.reshape(trX[k:k+batchsize, target-stride:target+stride+1, :, :, :], [batchsize, T, output_window_size, output_window_size, 4])
                    label_batch = np.reshape(trY[k:k+batchsize, target-stride:target+stride+1, :, :, :], [batchsize, T, output_window_size, output_window_size, 1])


                    lss, sample_logits, current_logits_amax = sess.run([loss_op, logits_packed, logits_amax], feed_dict={
                        img_input: image_batch, la_input: label_batch,
                        is_training: False})

                    sample_logits = sample_logits[stride]

                    if T > 1:

                        current_logits_amax = current_logits_amax[stride]

                    label_batch = label_batch[0, stride]


                    label_batch_calc = np.squeeze(label_batch)


                    image_batch_log[k-sum(experiment_sizes[0:num_train+num_val])]  = image_batch[0,stride]
                    label_log[k-sum(experiment_sizes[0:num_train+num_val])] = label_batch
                    preds_log[k-sum(experiment_sizes[0:num_train+num_val]),0] = np.reshape(current_logits_amax,[output_window_size,output_window_size,1])


                    confmat = metrics.confusion_matrix(np.ndarray.flatten(label_batch_calc.astype(dtype=np.int64)),
                                                       np.ndarray.flatten(current_logits_amax))

                    test_confmat_log += confmat

                test_iou = compute_iou(test_confmat_log)

                sio.savemat('params.mat',{'images':image_batch_log,'labels':label_log,'preds':preds_log})

                print "Test iou: " + str(np.mean(test_iou))



    current_filename = 'logs/window_sweep/' + current_time_str + ".txt"

    f = open(current_filename,'a')
    f.write(str(np.mean(test_iou))+"\n")
    f.close()

    #print iou

    #imgplot = plt.imshow(np.squeeze(current_logits_amax[4, ...]))
    #plt.show()
    #plt.savefig('sample.jpg')
    #plt.show()
