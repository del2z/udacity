#!/usr/bin/env python
import pickle
from keras import backend as K
from keras.models import Model
from keras.regularizers import l1, l2
from keras.layers import Input, Conv2D, Add, AveragePooling2D, Dropout
from utils import *

def crop_norm_img(img):
    """ Crop the image and normalize pixel values of image
    :param img: The image to be processed
    :return: Returns the processed image
    """
    ## Chop off 1/3 from the top and cut bottom 150px(which contains the head of car)
    shape = img.shape
    img = img[int(shape[0]/3):shape[0]-150, 0:shape[1]]
    ## Resize the image
    img = cv2.resize(img, (params.FLAGS.img_w, params.FLAGS.img_h), interpolation=cv2.INTER_AREA)
    return img / 255.0


def load_epoch(ep_id):
    """ Load image and steering data for epoch 'ep_id' """
    steering_path = join_dir(params.data_dir, 'epoch{:0>2}_steering.csv'.format(ep_id))
    front_vid_path = join_dir(params.data_dir, 'epoch{:0>2}_front.mkv'.format(ep_id))
    assert os.path.isfile(steering_path) and os.path.isfile(front_vid_path), \
        "Steering and video file NOT exist!"

    steering = fetch_csv_data(steering_path)
    assert steering.shape[0] == frame_count(front_vid_path), \
        "Frame count NOT match!"

    front_cap = cv2.VideoCapture(front_vid_path)
    vid_size = video_resolution_to_size('720p', width_first=True)
    assert vid_size[0] == int(front_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) and \
        vid_size[1] == int(front_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), \
        "Video is NOT 720p, must rescale it!"

    images = []
    labels = steering.wheel.values
    while True:
        ret, img = front_cap.read()
        if not ret:
            break
        images.append(crop_norm_img(img))
    front_cap.release()
    return np.array(images), labels


def split_epoch(ep_path, test_eps=[10,]):
    """ Split training and testing datasets """
    df_ep = pd.read_table(ep_path, sep=',', header=0)
    df_ep.drop(df_ep[df_ep['epoch_id'].isin(test_eps)].index, inplace=True)
    train_eps = df_ep['epoch_id'].values
    return train_eps


def combine_data(epochs):
    """ Combine training, validation and testing images and labels """
    images_list = []
    labels_list = []
    for ep_id in epochs:
        ep_images, ep_labels = load_epoch(ep_id)
        images_list.append(ep_images)
        labels_list.append(ep_labels)
        print("Loading steering and video for epoch-{:0>2} sucessfully.".format(ep_id))
    return images_list, labels_list


def dump_dataset(test_eps, fp_data):
    """ Dump datasets into disk """
    train_eps = split_epoch(params.data_dir + '/epochs.csv', test_eps)

    print("Training & validation datasets: epoch {:s}.".format(', '.join([str(x) for x in train_eps])))
    train_images, train_labels = combine_data(train_eps)
    print("Training & validation images: {:s}, \tlabels: {:s}\n".format(
        str(sum([t.shape[0] for t in train_images])) + ' x ' + ' x '.join([str(t) for t in train_images[0].shape[1:]]),
        str(sum([t.shape[0] for t in train_labels]))))

    print("Testing dataset: epoch {:s}.".format(', '.join([str(x) for x in test_eps])))
    test_images, test_labels = combine_data(test_eps)
    print("Testing images: {:s}, \tlabels: {:s}".format(
        str(sum([t.shape[0] for t in test_images])) + ' x ' + ' x '.join([str(t) for t in test_images[0].shape[1:]]),
        str(sum([t.shape[0] for t in test_labels]))))

    ds_dict = {
        'train_epochs': train_eps,
        'train_images': train_images,
        'train_labels': train_labels,
        'test_epochs': test_eps,
        'test_images': test_images,
        'test_labels': test_labels,
    }
    pickle.dump(ds_dict, open(fp_data, 'wb'), protocol=4)
    return


def load_dataset(fp_data):
    """ Load datasets from disk into memory """
    ds_dict = pickle.load(open(fp_data, 'rb'))
    print("Training & validation datasets: epoch {:s}.".format(', '.join([str(x) for x in ds_dict['train_epochs']])))
    print("Training images: {:s}, \tlabels: {:s}".format(
        str(sum([t.shape[0] for t in ds_dict['train_images']])) + ' x ' + ' x '.join([str(t) for t in ds_dict['train_images'][0].shape[1:]]),
        str(sum([t.shape[0] for t in ds_dict['train_labels']]))))

    print("Testing dataset: epoch {:s}.".format(', '.join([str(x) for x in ds_dict['test_epochs']])))
    print("Testing images: {:s}, \tlabels: {:s}\n".format(
        str(sum([t.shape[0] for t in ds_dict['test_images']])) + ' x ' + ' x '.join([str(t) for t in ds_dict['test_images'][0].shape[1:]]),
        str(sum([t.shape[0] for t in ds_dict['test_labels']]))))
    return ds_dict


def prepare_data_cnn(ds_dict, vali_ratio=0.3):
    """ Combine, split and shuffle data for ConvNet models """
    train_images = np.vstack(ds_dict['train_images'])
    train_labels = np.hstack(ds_dict['train_labels'])

    np.random.seed(42)
    vali_indices = np.random.choice(np.arange(train_labels.shape[0]), int(vali_ratio * train_labels.shape[0]), replace=False)
    train_mask = np.ones(train_labels.shape[0], dtype=bool)
    train_mask[vali_indices] = False

    vali_images = train_images[vali_indices, :, :, :]
    vali_labels = train_labels[vali_indices]

    train_images = train_images[train_mask, :, :, :]
    train_labels = train_labels[train_mask]

    test_images = np.vstack(ds_dict['test_images'])
    test_labels = np.hstack(ds_dict['test_labels'])

    print("Training examples {:d}".format(train_labels.shape[0]))
    print("Validation examples {:d}".format(vali_labels.shape[0]))
    print("Testing examples {:d}".format(test_labels.shape[0]))
    return train_images, train_labels, vali_images, vali_labels, test_images, test_labels


class RnnBatchGenerator(object):
    def __init__(self):
        pass



def prepare_data_rnn(ds_dict, batch_size, num_unroll):
    """ Combine and generate batch data for Recurrent NN """
    return


def create_benmk(input_shape, num_output, num_hidden=512, activation='sigmoid'):
    """ 基准模型（三层全连接神经网络）
    """
    input = Input(shape=input_shape)
    flat_out = Flatten()(input)
    fc_out = Dense(num_hidden,
            kernel_initializer='he_normal',
            bias_initializer='zeros',
            kernel_regularizer=l1(0.0001),
            activation=activation)(flat_out)
    output = Dense(num_output,
            kernel_initializer='he_normal',
            bias_initializer='zeros',
            kernel_regularizer=l1(0.0001),
            activation='linear')(fc_out)
    model = Model(inputs=input, outputs=output)
    return model


def create_convnet(input_shape, num_output):
    """ 简单卷积模型
    """
    input = Input(shape=input_shape)
    conv1_out = Conv2D(64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding='valid',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(0.0001))(input)
    pool1_out = MaxPooling2D(pool_size=(2, 2))(conv1_out)
    conv2_out = Conv2D(128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(0.0001))(pool1_out)
    pool2_out = MaxPooling2D(pool_size=(2, 2))(conv2_out)
    conv3_out = Conv2D(256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(0.0001))(pool2_out)
    #pool3_out = MaxPooling2D(pool_size=(2, 2))(conv3_out)
    relu_out = Activation("relu")(conv3_out)
    flat_out = Flatten()(relu_out)
    output = Dense(num_output,
            kernel_initializer='he_normal',
            bias_initializer='zeros',
            kernel_regularizer=l1(0.0001),
            activation='linear')(flat_out)
    model = Model(inputs=input, outputs=output)
    return model


def res1_block(filters, activation='relu'):
    """ 类型1残差结构（< 34层）
    """
    def func_res1(input):
        conv1_out = Conv2D(filters,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(0.0001))(input)
        relu_out = Activation("relu")(conv1_out)
        conv2_out = Conv2D(filters,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(0.0001))(relu_out)
        sum_out = Add()([input, conv2_out])
        output = Activation("relu")(sum_out)
        return output
    return func_res1


def res2_block(filters, activation='relu'):
    """ 类型2残差结构
    """
    def func_res2(input):
        conv1_out = Conv2D(filters,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(0.0001))(input)
        relu_out = Activation("relu")(conv1_out)
        conv2_out = Conv2D(filters,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(0.0001))(relu_out)
        in_shape = K.int_shape(input)
        out_shape = K.int_shape(conv2_out)
        scut_out = Conv2D(filters=filters,
                kernel_size=(1, 1),
                strides=(int(round(in_shape[1] / out_shape[1])), int(round(in_shape[2] / out_shape[2]))),
                padding="valid",
                kernel_initializer="he_normal",
                kernel_regularizer=l2(0.0001))(input)
        sum_out = Add()([scut_out, conv2_out])
        output = Activation("relu")(sum_out)
        return output
    return func_res2


def res3_block(filters, activation='relu'):
    """ 类型3残差结构（>= 34层）
    """
    def func_res3(input):
        return
    return func_res3


def create_resnet(input_shape, num_output, res_layer=4, optimizer='adam'):
    """ 残差网络
    """
    input = Input(shape=input_shape)
    conv_out = Conv2D(64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding='valid',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(0.0001))(input)
    res_out = MaxPooling2D(pool_size=(2, 2))(conv_out)
    for k in range(3):
        res_out = res1_block(64)(res_out)
    res_out = res2_block(128)(res_out)
    for k in range(3):
        res_out = res1_block(128)(res_out)
    res_out = res2_block(256)(res_out)
    for k in range(3):
        res_out = res1_block(256)(res_out)
    pool_out = AveragePooling2D(pool_size=(2, 2))(res_out)
    flat_out = Flatten()(pool_out)
    flat_out = Dropout(0.2)(flat_out)
    fc_out = Dense(256,
            kernel_initializer='he_normal',
            kernel_regularizer=l2(0.0001),
            activation='relu')(flat_out)
    fc_out = Dropout(0.3)(fc_out)
    output = Dense(num_output,
            kernel_initializer='he_normal',
            kernel_regularizer=l2(0.0001),
            activation='linear')(fc_out)
    model = Model(inputs=input, outputs=output)
    return model


if __name__ == '__main__':
    test_eps = [ 10, ]
    dump_dataset(test_eps, 'datasets.pickle')

