import numpy as np
import matplotlib.pyplot as plt


# to plot images and its predicted labels
# input
# row, col means how many images in a row/col
# when predicted labels are given, type should be specified and predicted label will be shown
# type = 'T' shows corrected labeled images
# type = 'F' shows corrected labeled images
def show_img(data, label, row, col, type=None, p_label=None):
    size = (28, 28)
    # first transform data type
    data = (data - data.min()) / data.max() * 255
    data = data.astype('uint8')
    # randomly choosing some images
    try:
        if type == 'T':
            idx = np.random.choice(np.where(p_label == label)[0], row * col, replace=False)
        elif type == 'F':
            idx = np.random.choice(np.where(p_label != label)[0], row * col, replace=False)
        else:
            idx = np.random.randint(0, data.shape[0], row * col)
            p_label = label
    except:
        print('**************less points than needed***************')
        return 0
    # create a plot
    plt.figure(figsize=(row * 2, col * 2))
    # show images and labels
    for i in range(row * col):
        plt.subplot(row, col, i + 1)
        tmp = data[idx[i],]
        # if not gray image, remove cmap = 'gray'
        plt.imshow(tmp.reshape(size), cmap='gray')
        plt.title('label: ' + str(p_label[idx[i]]))
        plt.xticks([])
        plt.yticks([])


# extract specific label (denoted by e, a list) and transform label
# transform binary classification label to -1/+1
# transform multi-class label to one-hot vector
def data_extract(data, label, e=None):
    # extract index for labels in e
    idx = [l in e for l in label]
    # extract them
    new_data = data[idx, ]
    tmp_label = label[idx, ]
    # transform to new label
    if len(e) == 2:
        new_label = np.zeros_like(tmp_label)
        new_label[tmp_label == e[0]] = -1
        new_label[tmp_label == e[1]] = 1
        mapping = {e[0]: -1, e[1]: 1}
    elif len(e) > 2:
        row = tmp_label.shape[0]
        col = len(e)
        new_label = np.zeros([row, col])
        mapping = {a: b for a, b in zip(e, np.arange(col))}
        new_label[np.arange(row), [mapping[i] for i in tmp_label.flatten()]] = 1
    else:
        new_label = tmp_label
        mapping = None
    # return
    return new_data, new_label, mapping


# transform binary classification label to -1/+1
# transform multi-class label to one-hot vector
def get_label(label, k):
    # get unique labels
    e = np.unique(label)
    if len(e) != k:
        raise Exception('number of labels K not match with data')
    # transform to new label
    if k == 2:
        new_label = np.zeros_like(label)
        new_label[label == e[0]] = -1
        new_label[label == e[1]] = 1
        label2tag = {e[0]: -1, e[1]: 1}
        tag2label = {-1: e[0], 1: e[1]}
    elif k > 2:
        row = label.shape[0]
        col = k
        new_label = np.zeros([row, col])
        label2tag = {a: b for a, b in zip(e, np.arange(col))}
        tag2label = {b: a for a, b in zip(e, np.arange(col))}
        new_label[np.arange(row), [label2tag[i] for i in label.flatten()]] = 1
    else:
        raise Exception('invalid K')
    # return
    return label2tag, tag2label, new_label


# this function will drop each element in data with prob p, p can be a list
# input
# keep: means the percentage of data you want to keep, an integer
# rep : means do rep times for the whole data, a real value from 0 to 1
# if the input data has n samples, the output will have rep*n*keep samples, each drop its feature with prob p.
def drop_data(data, label, p, memorycontrol):
    if type(p) != list:
        tmp_p = p
        t, l = drop_data_h(data, label, tmp_p, memorycontrol)
    else:
        tmp_p = p[0]
        t, l = drop_data_h(data, label, tmp_p, memorycontrol)
        if len(p) > 1:
            for tmp_p in p[1:]:
                tt, ll = drop_data_h(data, label, tmp_p, memorycontrol)
                t = np.concatenate((t, tt), axis=0)
                l = np.concatenate((l, ll), axis=0)
    return t, l


# the helper function for drop_data
def drop_data_h(data, label, p, memorycontrol):
    rep = int(memorycontrol/2 - 0.01) + 1
    keep = memorycontrol/2
    # initialize
    n_sample = data.shape[0]
    n_keep = min(n_sample, int(n_sample * keep))
    data_drop = np.zeros([rep * n_keep] + list(data.shape[1:]))
    label_drop = np.zeros([rep * n_keep] + list(label.shape[1:]))
    # drop
    for i in range(rep):
        tmp_idx = np.random.choice(np.arange(n_sample), n_keep, replace=False)
        tmp_data = data[tmp_idx,]
        tmp_mask = np.random.random(tmp_data.shape)
        data_drop[i * n_keep:(i + 1) * n_keep, ] = tmp_data * (tmp_mask < (1 - p))
        label_drop[i * n_keep:(i + 1) * n_keep, ] = label[tmp_idx,]
    # save
    return data_drop, label_drop
