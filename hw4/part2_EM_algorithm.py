import os
import struct
import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt

def load_mnist(path, kind='train'):
    label_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    image_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

    with open(label_path, 'rb') as lbpath:
        # '>' for big endian
        # 'I' for unsigned int
        magic, num = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(image_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 28*28)

    return images, labels

def classify_img_to_class(images, p, K):
    N = images.shape[0]
    classification = np.zeros((N, K))
    tmp = 1 - p
    for n in range(N):
            for k in range(K):
                    classification[n, k] = np.prod(p[k, images[n,:]==1]) * np.prod(tmp[k, images[n,:]==0])
    
    return np.argmax(classification, axis=1)

def calc_confusion_matrix(prediction, ground_truth, K):
    for i in range(K):
        target_gt = ground_truth[prediction == i]
        TP = np.sum(target_gt == i)
        FP = np.sum(target_gt != i)
        target_gt = ground_truth[prediction != i]
        FN = np.sum(target_gt == i)
        TN = np.sum(target_gt != i)
        
        sensitivity = TP / (TP + FN)
        specificity = FP / (FP + TN)
        
        print('Confusion matrix {}:'.format(i))
        print('{:20}{:^20}{}{:^20}{}'.format(' ','Is number ', i, 'Isnt number ', i))
        print('{:<20}{}{:^20d}{:^20d}'.format('Predict number ', i, TP, FP))
        print('{:<20}{}{:^20d}{:^20d}\n'.format('Predict number ', i, FN, TN))
        print('{}: {}'.format('Sensitivity: ', sensitivity))
        print('{}: {}\n'.format('Specificity: ', specificity))
        print('-' * 30)

if __name__ == '__main__':
    # read data
    # images_.shape = (60000, 28*28)
    # labels_.shape = (60000,)
    images_train, labels_train = load_mnist('./', 'train')
    images_test, labels_test = load_mnist('./', 't10k')
    # convert each pixel value to 0 or 1
    images_train = (images_train > 127).astype(int)
    images_test = (images_test > 127).astype(int)
    # images_train = images_train[:600]
    # labels_train = labels_train[:600]

    N = images_train.shape[0]                   # number of train images
    K = 10                                      # number of classes
    D = images_train.shape[1]                   # dimension of each image
    
    lbda = np.ones((K, 1)) * (1 / K)            # lambda: the chance of class i
    p = np.random.uniform(0.25, 0.75, (K, D))
    summ = np.sum(p, axis=1)
    for i in range(p.shape[1]):
        p[:, i] = p[:, i] / summ
    
    w = np.zeros((N, K)).astype(float)

    # EM algorithm
    iter = 0
    while(1):
        # E step
        for n in range(N):
            for k in range(K):
                w[n, k] = 1
                for i in range(D):
                    if images_train[n, i] == 1:
                        w[n, k] = w[n, k] * p[k, i]
                    else:
                        w[n, k] = w[n, k] * (1 - p[k, i])
        
        w = w * lbda.T
        summ = np.sum(w, axis=1).astype(float)
        summ[summ == 0] = 1 / K
        for i in range(w.shape[1]):
            w[:, i] = w[:, i] / summ
        
        # M step
        effNum = np.sum(w, axis=0).astype(float)
        p = np.dot(w.T, images_train)
        p = p / effNum[:, None]
        
        # convergence
        iter = iter + 1
        if iter >= 10:
            break
    
    # print imagination of each class
    for i, number in enumerate(p):
        print('\nclass %d:' % (i))
        number_str = np.array2string((number > 0.5).astype(int), separator=' ', max_line_width=28 * 2 + 1)
        print(' ' + number_str[1: -1])
    
    train_img_classes = classify_img_to_class(images_train, p, K)

    # assign class to the most likely label(actual number)
    # build class to label matrix
    class2label_mat = np.zeros((K, K))
    for c in range(K):
        idx = np.argwhere(train_img_classes == c)
        target_labels = labels_train[idx]
        unique, counts = np.unique(target_labels, return_counts=True)
        for i in range(unique.shape[0]):
            class2label_mat[c, unique[i]] = counts[i]
    
    class2label_mat = class2label_mat.astype(int)
    
    avail_class = np.array(range(10)).astype(int)
    avail_label = np.array(range(10)).astype(int)
    class2label = np.zeros(K)
    table = class2label_mat.astype(int)
    
    while(avail_class.size > 0):
        # find MAX in the class to label matrix
        MAX_idx = np.unravel_index(np.argmax(table), table.shape)
        target_class, target_label = MAX_idx[0], MAX_idx[1]
        if  target_class in avail_class and target_label in avail_label:
            # assign target_label to target_class
            class2label[target_class] = target_label
            # remove target_class, target_label from avail_class, avail_label
            avail_class = np.delete(avail_class, np.where(avail_class == target_class))
            avail_label = np.delete(avail_label, np.where(avail_label == target_label))
        
        table[target_class, target_label] = -1
        if avail_class.size == 1:
            class2label[avail_class[0]] = avail_label[0]
            break
    
    # print labeled class
    print('-' * 30)
    for label in range(K):
        idx = np.argwhere(class2label == label)[0, 0]
        number = p[idx]
        print('\nlabeled class %d:' % (label))
        number_str = np.array2string((number > 0.5).astype(int), separator=' ', max_line_width=28 * 2 + 1)
        print(' ' + number_str[1: -1])

    test_img_classes = classify_img_to_class(images_test, p, K)
    test_img_predict_labels = np.zeros(test_img_classes.shape[0])
    for i in range(test_img_classes.shape[0]):
        test_img_predict_labels[i] = class2label[test_img_classes[i]]
    
    print('-' * 30)
    calc_confusion_matrix(test_img_predict_labels, labels_test, K)