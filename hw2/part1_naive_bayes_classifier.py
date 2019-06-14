import os
import struct
import numpy as np
import math
import random
import time

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

def write_result(kind, posteriors, predictions, labels_test, imagination_numbers, computation_time, error_rate):
    filename = kind + '_naive_Bayes_result.txt'
    my_file = open(filename, 'w')

    for i in range(posteriors.shape[0]):
        my_file.write('Posterior (in log scale):\n')
        for j in range(posteriors.shape[1]):
            my_file.write('{0}: {1}\n'.format(j, posteriors[i, j]))
        my_file.write('Prediction: {0}, Ans: {1}\n\n'.format(int(predictions[i]), labels_test[i]))
    
    my_file.write('Imagination of numbers in Bayesian classifier:\n')
    for i, number in enumerate(imagination_numbers_discrete):
        my_file.write('\n%d:\n' % (i))
        number_str = np.array2string(number.astype(int), separator=' ', max_line_width=28 * 2 + 1)
        my_file.write(' ' + number_str[1: -1] + '\n')

    my_file.write('\nComputation time (minutes): {0}'.format(computation_time/60))
    my_file.write('\nError rate: {0}'.format(error_rate))

def classify_to_bins(x, n):
    '''
    n elements as a group
    '''
    return math.ceil((x+1)/n)

def naive_Bayes_discrete(images_train, labels_train, images_test, labels_test):
    # classify each pixel value from 0~255 to 1~32
    vectorized_classify_to_bins = np.vectorize(classify_to_bins)
    images_train_32 = vectorized_classify_to_bins(images_train, 8)
    images_test_32 = vectorized_classify_to_bins(images_test, 8)

    # build discrete distribution table
    # classify the images into 10 categories w.r.t. labels (0 ~ 9)
    images_train_labeled = []
    for i in range(10):
        # each element is an ndarray in the list
        images_train_labeled.append(images_train_32[labels_train == i])
    
    # for each label class, classify into 28*28 categories w.r.t. pixel order
    # and build the distribution table
    distribution_table = np.zeros((10, 28*28, 32))
    for i in range(len(images_train_labeled)):  # len(images_train_labeled) = 10
        target_images = images_train_labeled[i]
        for j in range(28*28):
            # unique(ndarray): 1 ~ 32
            # c(ndarray): #occurences of each unique
            unique, c = np.unique(target_images[:, j], return_counts=True)
            counts = np.zeros(32)
            for idx, value in enumerate(unique):
                counts[value - 1] = c[idx]
            distribution_table[i, j] = counts / np.sum(counts)

    n_error = 0
    posteriors = np.zeros((images_test.shape[0], 10))
    predictions = np.zeros(images_test.shape[0])
    # randomly pick 3 test images, and print the posteriors
    rnds = [random.randint(0, images_test_32.shape[0]) for _ in range(3)]
    for i in range(images_test_32.shape[0]):
        D = images_test_32[i]           # current test image
        ground_truth = labels_test[i]   # label of current test image
        counts = np.zeros(10)

        for target_label in range(10):
            # all images with label == target_label
            target_images = images_train_32[labels_train == target_label]
            n_target_images = target_images.shape[0]

            count = 0
            for j in range(D.shape[0]):
                pixel_value = D[j]
                col_target_images = target_images[:, j]
                n_pixel_value = col_target_images[col_target_images == pixel_value].shape[0]
                if n_pixel_value == 0:
                    n_pixel_value = 1
                count = count + math.log(n_pixel_value / n_target_images)
            counts[target_label] = count + math.log(labels_train[labels_train == target_label].shape[0] / images_train_32.shape[0])
        
        sum_counts = np.sum(counts)

        posterior = counts / sum_counts
        posteriors[i] = posterior

        prediction = np.argmin(posterior)
        predictions[i] = prediction

        if prediction != ground_truth:
            n_error += 1
        if i in rnds:
            print('Data #{0}'.format(i))
            print('Posterior (in log scale):')
            for i in range(posterior.shape[0]):
                print('{0}: {1}'.format(i, posterior[i]))
            print('Prediction: {0}, Ans: {1}\n'.format(prediction, ground_truth))
    
    error_rate = n_error / images_test_32.shape[0]

    return posteriors, predictions, error_rate, distribution_table

def naive_Bayes_discrete_2(images_train, labels_train, images_test, labels_test):
    # classify each pixel value from 0~255 to 1~32
    vectorized_classify_to_bins = np.vectorize(classify_to_bins)
    images_train_32 = vectorized_classify_to_bins(images_train, 8)
    images_test_32 = vectorized_classify_to_bins(images_test, 8)

    # build discrete distribution table
    # classify the images into 10 categories w.r.t. labels (0 ~ 9)
    images_train_labeled = []
    for i in range(10):
        # each element is an ndarray in the list
        images_train_labeled.append(images_train_32[labels_train == i])
    
    # for each label class, classify into 28*28 categories w.r.t. pixel order
    # and build the distribution table
    distribution_table = np.zeros((10, 28*28, 32))
    for i in range(len(images_train_labeled)):  # len(images_train_labeled) = 10
        target_images = images_train_labeled[i]
        for j in range(28*28):
            # unique(ndarray): 1 ~ 32
            # c(ndarray): #occurences of each unique
            unique, c = np.unique(target_images[:, j], return_counts=True)
            counts = np.zeros(32)
            for idx, value in enumerate(unique):
                counts[value - 1] = c[idx]
            distribution_table[i, j] = counts / np.sum(counts)

    n_error = 0
    posteriors = np.zeros((images_test.shape[0], 10))
    predictions = np.zeros(images_test.shape[0])
    # randomly pick 3 test images, and print the posteriors
    rnds = [random.randint(0, images_test_32.shape[0]) for _ in range(3)]
    for i in range(images_test_32.shape[0]):
        D = images_test_32[i]           # current test image
        ground_truth = labels_test[i]   # label of current test image
        counts = np.zeros(10)

        for target_label in range(10):
            count = 0
            for j in range(D.shape[0]):
                pixel_value = D[j]
                probability = distribution_table[target_label, j, pixel_value - 1]
                if probability != 0:
                    count = count + math.log(probability)
            counts[target_label] = count + math.log(labels_train[labels_train == target_label].shape[0] / images_train_32.shape[0])
        
        sum_counts = np.sum(counts)

        posterior = counts / sum_counts
        posteriors[i] = posterior

        prediction = np.argmin(posterior)
        predictions[i] = prediction

        if prediction != ground_truth:
            n_error += 1
        if i in rnds:
            print('Data #{0}'.format(i))
            print('Posterior (in log scale):')
            for i in range(posterior.shape[0]):
                print('{0}: {1}'.format(i, posterior[i]))
            print('Prediction: {0}, Ans: {1}\n'.format(prediction, ground_truth))
    
    error_rate = n_error / images_test_32.shape[0]

    return posteriors, predictions, error_rate, distribution_table

def cal_mean_variance(data):
    '''
    data is an ndarray
    '''
    # compute mean, variance
    m = np.sum(data) / (data.shape[0] * data.shape[1])
    var = np.sum((data - m) * (data - m)) / (data.shape[0] * data.shape[1])

    return m, var

def Gaussian_distribution(x, m, var):
    if var == 0:
        return -1
    
    p = (1 / math.sqrt(2 * math.pi * var)) * math.exp(-((x - m) ** 2) / (2 * var))
    if p == 0:
        return -1
    else:
        return p

def imagination_numbers_discrete_Bayes(distribution_table):
    imagination = np.zeros((10, 28 * 28))
    for label in range(distribution_table.shape[0]):
        for pixel in range(distribution_table.shape[1]):
            white = 0
            black = 0
            for i in range(2 ** 7):
                bin_class = math.ceil((i+1)/8) - 1
                white = white + distribution_table[label, pixel, bin_class]
                bin_class = math.ceil((i+2 ** 7+1)/8) - 1
                black = black + distribution_table[label, pixel, bin_class]
                if white > black:
                    imagination[label, pixel] = 0
                else:
                    imagination[label, pixel] = 1

    return imagination

def imagination_numbers_continuous_Bayes(means, variances):
    imagination = np.zeros((10, 28 * 28))
    for label in range(means.shape[0]):
        for pixel in range(means.shape[1]):
            white = 0
            black = 0 
            for i in range(2 ** 7):
                white = white + Gaussian_distribution(i, means[label, pixel], variances[label, pixel])
                black = black + Gaussian_distribution(i + 2 ** 7, means[label, pixel], variances[label, pixel])
            # print('white: {0}, black: {1}'.format(white, black))
            if white >= black:
                imagination[label, pixel] = 0
            else:
                imagination[label, pixel] = 1
    
    return imagination

def naive_Bayes_continuous(images_train, labels_train, images_test, labels_test):
    # classify the images into 10 categories w.r.t. labels (0 ~ 9)
    images_train_labeled = []
    for i in range(10):
        # each element is an ndarray in the list
        images_train_labeled.append(images_train[labels_train == i])
    
    # for each label class, classify into 28*28 categories w.r.t. pixel order
    # and calculate their means and variances
    all_mean = np.zeros((10, 28*28))
    all_var = np.zeros((10, 28*28))
    for i in range(len(images_train_labeled)):  # len(images_train_labeled) = 10
        target_images = images_train_labeled[i]
        for j in range(28*28):
            mean, variance = cal_mean_variance(target_images[:, j].reshape(-1, 1))
            all_mean[i, j] = mean       # mean of ith label, jth pixel
            all_var[i, j] = variance    # variance of ith label, jth pixel
    
    n_error = 0
    posteriors = np.zeros((images_test.shape[0], 10))
    predictions = np.zeros(images_test.shape[0])
    # randomly pick 3 test images, and print the posteriors in the terminal
    rnds = [random.randint(0, images_test.shape[0]) for _ in range(3)]
    for i in range(images_test.shape[0]):
        D = images_test[i]              # current test image
        ground_truth = labels_test[i]   # label of current test image
        counts = np.zeros(10)           # posterior for each label (0 ~ 9)

        for target_label in range(10):
            probability = 0
            for j in range(D.shape[0]):
                p = Gaussian_distribution(D[j], all_mean[target_label, j], all_var[target_label, j])
                if p != -1:
                    probability = probability + math.log(p)
            
            counts[target_label] = probability + math.log(labels_train[labels_train == target_label].shape[0] / images_train.shape[0])
        
        sum_counts = np.sum(counts)

        posterior = counts / sum_counts
        posteriors[i] = posterior

        prediction = np.argmin(posterior)
        predictions[i] = prediction

        if prediction != ground_truth:
            n_error += 1
        # print on the terminal
        if i in rnds:
            print('Data #{0}'.format(i))
            print('Posterior (in log scale):')
            for j in range(posterior.shape[0]):
                print('{0}: {1}'.format(j, posterior[j]))
            print('Prediction: {0}, Ans: {1}\n'.format(prediction, ground_truth))
    

    error_rate = n_error / images_test.shape[0]

    return posteriors, predictions, error_rate, all_mean, all_var
    

if __name__ == '__main__':
    # read data
    images_train, labels_train = load_mnist('./', 'train')
    images_test, labels_test = load_mnist('./', 't10k')

    print('Discrete naive Bayes\n')

    start_discrete = time.time()
    posteriors, predictions, error_rate, distribution_table = naive_Bayes_discrete(images_train, labels_train, images_test, labels_test)
    end_discrete = time.time()

    imagination_numbers_discrete = imagination_numbers_discrete_Bayes(distribution_table)

    # print imagination numbers
    print('Imagination of numbers in discrete Bayes classifier: ')
    for i, number in enumerate(imagination_numbers_discrete):
        print('\n%d:' % (i))
        number_str = np.array2string(number.astype(int), separator=' ', max_line_width=28 * 2 + 1)
        print(' ' + number_str[1: -1])

    print('\nComputation time(minutes): {0}'.format((end_discrete - start_discrete)/60))
    print('Error rate: {0}'.format(error_rate))
    write_result('discrete', posteriors, predictions, labels_test, imagination_numbers_discrete, (end_discrete - start_discrete), error_rate)

    print('\n' + '-' * 10)
    print('Discrete naive Bayes 2\n')

    start_discrete_2 = time.time()
    posteriors, predictions, error_rate, distribution_table = naive_Bayes_discrete_2(images_train, labels_train, images_test, labels_test)
    end_discrete_2 = time.time()

    imagination_numbers_discrete = imagination_numbers_discrete_Bayes(distribution_table)

    # print imagination numbers
    print('Imagination of numbers in discrete Bayes classifier: ')
    for i, number in enumerate(imagination_numbers_discrete):
        print('\n%d:' % (i))
        number_str = np.array2string(number.astype(int), separator=' ', max_line_width=28 * 2 + 1)
        print(' ' + number_str[1: -1])

    print('\nComputation time(minutes): {0}'.format((end_discrete_2 - start_discrete_2)/60))
    print('Error rate: {0}'.format(error_rate))
    write_result('discrete_2', posteriors, predictions, labels_test, imagination_numbers_discrete, (end_discrete_2 - start_discrete_2), error_rate)

    print('\n' + '-' * 10)
    print('Continuous naive Bayes\n')

    start_con = time.time()
    posteriors, predictions, error_rate, all_mean, all_var = naive_Bayes_continuous(images_train, labels_train, images_test, labels_test)
    end_con = time.time()

    imagination_numbers = imagination_numbers_continuous_Bayes(all_mean, all_var)

    # print imagination numbers
    print('Imagination of numbers in continuous Bayes classifier: ')
    for i, number in enumerate(imagination_numbers):
        print('\n%d:' % (i))
        number_str = np.array2string(number.astype(int), separator=' ', max_line_width=28 * 2 + 1)
        print(' ' + number_str[1: -1])
    
    print('\nComputation time(minutes): {0}'.format((end_con - start_con)/60))
    print('Error rate: {0}'.format(error_rate))
    write_result('continuous', posteriors, predictions, labels_test, imagination_numbers, (end_con - start_con), error_rate)