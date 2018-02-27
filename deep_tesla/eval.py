#!/usr/bin/env python
import sys
import pickle
import time
from keras.optimizers import Adam
from model import load_dataset, create_benmk, create_convnet, create_resnet, save_model

def train(model, train_images=None, train_labels=None, 
        vali_images=None, vali_labels=None, batch_size=128, epochs=10):
    assert train_images is not None and train_labels is not None and \
            vali_images is not None and vali_labels is not None
    
    model.compile(loss='mean_squared_error',
            optimizer='adam',
            metrics=['mse'])
    ts_begin = time.time()
    history = model.fit(train_images, train_labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(vali_images, vali_labels))
    ts_end = time.time()
    ts_train = round(ts_end - ts_begin, 2)
    print('Training time: {:f}'.format(ts_train))
    return history, ts_train


def evaluate(model, test_images, test_labels, batch_size=128):
    history = model.evaluate(test_images, test_labels,
            batch_size=batch_size,
            verbose=1)
    return history


def predict(model, test_images, batch_size=128):
    prdt_out = model.predict(test_images, batch_size=batch_size, verbose=0)
    return prdt_out


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(1)

    VERS_MODEL = sys.argv[1]

    ds_dict = load_dataset('datasets_new.pickle')
    train_images = ds_dict['train_images']
    train_labels = ds_dict['train_labels']
    vali_images = ds_dict['vali_images']
    vali_labels = ds_dict['vali_labels']
    test_images = ds_dict['test_images']
    test_labels = ds_dict['test_labels']

    benmk = {
        'num_hidden': 512,
        'activation': 'sigmoid',
    }
    model_v0 = create_benmk(input_shape = train_images.shape[1:], num_output = 1, **benmk)
    model_v1 = create_convnet(input_shape = train_images.shape[1:], num_output = 1)
    model_v2 = create_resnet(input_shape = train_images.shape[1:], num_output = 1)

    batch_size = 128
    epochs = 10
    train_param = {
        'train_images': train_images,
        'train_labels': train_labels,
        'vali_images': vali_images,
        'vali_labels': vali_labels, 
        'batch_size': batch_size,
        'epochs': epochs,
    }

    ev_model = eval("model_" + VERS_MODEL)
    train_hist, train_time = train(ev_model, **train_param)
    pickle.dump([train_hist.history, train_time], open('models/{:s}_train.info'.format(VERS_MODEL), 'wb'))

    test_hist = evaluate(ev_model, test_images, test_labels, batch_size)
    print('Test MSE:', test_hist[1])

    pred_out = predict(ev_model, test_images, batch_size)
    pickle.dump([test_labels, pred_out], open('models/{:s}_pred.out'.format(VERS_MODEL), 'wb'))

    save_model(ev_model, VERS_MODEL)

