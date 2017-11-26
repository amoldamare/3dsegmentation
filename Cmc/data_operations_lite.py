from scipy.misc import imread, imresize
import glob
import os
import numpy as np
import tensorflow as tf
import scipy
import classifier
import scipy.io as sio
import matplotlib.pyplot as plt


large_frame_size = 512

output_window_size = 240

T = 20



def read_data_from_folder(path):
    dirs = glob.glob(path+"*/")

    #dirs.remove(dirs[4])

    #dirs = dirs[:-1]

    #complete_dataset = np.zeros((len(dirs),20,512,512,3))
    #complete_labels = np.zeros((len(dirs),20,512,512,1))

    cnt = 0
    list_of_input = []
    list_of_labels = []

    valdir = dirs[7]

    dirs[7] = dirs[8]

    dirs[8] = valdir

    for dir in dirs:
        experiment_name = dir.split("/")[-2]

        img_files = glob.glob(dir+"/"+experiment_name+"_images/*.png")
        mat_files = glob.glob(dir+"/"+experiment_name+"_images/*.mat")
        labels = glob.glob(dir+"/"+experiment_name+"_masks/*.png")

        crop_locations = sio.loadmat(dir + experiment_name + "_coords.mat")

        current_coords = crop_locations['coords'];

        img_files = sorted(img_files)
        mat_files = sorted(mat_files)
        labels = sorted(labels)

        current_large_video = np.zeros((T,large_frame_size,large_frame_size,3))
        current_large_labels = np.zeros((T,large_frame_size,large_frame_size,1))





        for i in range(len(img_files)):
            current_im = imread(img_files[i])
            current_modalities = sio.loadmat(mat_files[i])
            current_label = imread(labels[i])

            current_red = current_modalities["current_red"]
            current_green = current_modalities["current_green"]

            current_sample = np.zeros((large_frame_size,large_frame_size,3))

            current_sample[...,0] = current_im
            current_sample[...,1] = current_im
            current_sample[...,2] = current_im
            #current_sample[...,1] = current_red
            #current_sample[...,2] = current_green


            current_large_video[i] = current_sample

            current_label = current_label[...,0]

            current_large_labels[i,:,:,0] = current_label


        X,Y = extract_tubes(current_large_video,current_large_labels,current_coords)

        list_of_input.append(X)
        list_of_labels.append(Y)


        #complete_dataset[cnt,...] = current_large_video
        #complete_labels[cnt,...] = current_large_labels

        cnt += 1

    num_train = 7

    num_val = 1

    num_test = 1

    experiment_sizes = [x.shape[0] for x in list_of_input]

    trX= np.zeros((sum(experiment_sizes[:num_train+num_val+num_test]),T,output_window_size,output_window_size,4))

    trY = np.zeros((sum(experiment_sizes[:num_train+num_val+num_test]), T, output_window_size, output_window_size, 1))

    for i in range(num_train+num_val+num_test):
        trX[sum(experiment_sizes[:i]):sum(experiment_sizes[:i+1])] = list_of_input[i]
        trY[sum(experiment_sizes[:i]):sum(experiment_sizes[:i + 1])] = list_of_labels[i]


#    for i in range(num_train,num_train + num_val):
#        trX[sum(experiment_sizes[:i]):sum(experiment_sizes[:i+1])] = X[i]
#        trY[sum(experiment_sizes[:i]):sum(experiment_sizes[:i + 1])] = Y[i]

#    for i in range(num_train + num_val, num_train + num_val + num_test):
#        trX[sum(experiment_sizes[:i]):sum(experiment_sizes[:i+1])] = X[i]
#        trY[sum(experiment_sizes[:i]):sum(experiment_sizes[:i + 1])] = Y[i]
    print trX.shape
    return trX, trY, experiment_sizes



def extract_tubes(full_video,full_labels,coordinates):

    full_video = np.reshape(full_video, (1,T,large_frame_size,large_frame_size,3))

    full_labels = np.reshape(full_labels, (1, T, large_frame_size, large_frame_size, 1))

    window_size = 128

    half_size = window_size / 2


    tube_centers = coordinates

    tube_centers = np.array([[x[1], x[0]] for x in tube_centers])

    #bg_coords = choose_background_crops(tube_centers)

    num_tubes = tube_centers.shape[0]


    all_tubes = np.zeros((num_tubes,T,output_window_size,output_window_size,4))

    all_labels = np.zeros((num_tubes,T,output_window_size,output_window_size,1))

    for i in range(num_tubes):

        current_tube = full_video[:,:,max(tube_centers[i,0]-half_size,0):min(tube_centers[i,0]+half_size,large_frame_size-1),max(tube_centers[i,1]-half_size,0):min(tube_centers[i,1]+half_size,large_frame_size-1),:]

        current_label = full_labels[:,:,max(tube_centers[i,0]-half_size,0):min(tube_centers[i,0]+half_size,large_frame_size-1),max(tube_centers[i,1]-half_size,0):min(tube_centers[i,1]+half_size,large_frame_size-1),:]




        print current_tube.shape

        current_tube_large = np.zeros((1,T,output_window_size,output_window_size,4))
        current_label_large = np.zeros((1,T,output_window_size,output_window_size,1))
        for j in range(T):
            #pad_crops(current_tube, current_label)

            current_tube_large[0,j,:,:,:3] = scipy.misc.imresize(np.squeeze(current_tube[0,j,...]),(output_window_size,output_window_size,3),interp='nearest')
            current_tube_large[0,j,:,:,3] = current_tube_large[0,j,:,:,0]

            current_label_large[0,j,:,:,:] = np.reshape(scipy.misc.imresize(np.squeeze(current_label[0,j,...]),(output_window_size,output_window_size),interp='nearest'),(output_window_size,output_window_size,1)) == 255
        all_tubes[i,...] = current_tube_large

        all_labels[i,...] = current_label_large
    return all_tubes, all_labels
