"""Train the model"""

import argparse
import os
import cv2
import tensorflow as tf
import numpy as np

from model.input_fn import train_input_fn
from model.input_fn import test_input_fn
from model.model_fn import model_fn
from model.utils import Params


parser = argparse.ArgumentParser()
# parser.add_argument('--model_dir', default='experiments/base_model',
parser.add_argument('--model_dir', default='experiments/batch_all',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default=r'F:\110105036-2126-5505_OK_fill',
                    help="Directory containing the dataset")
BATCH_SIZE = 64
EPOCHS = 5

def loadImageLabel(data_dir):
    images = []
    labels = []
    for i in os.listdir(data_dir):
        for j in os.listdir(data_dir + '/' + i):
            images.append(cv2.imread(data_dir + '/' + i + '/' + j))
            labels.append(i)

    return np.asarray(images), np.asarray(labels)


def input_fn(images, labels, epochs, batch_size):
    # Convert the inputs to a Dataset. (E)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    # Shuffle, repeat, and batch the examples. (T)
    SHUFFLE_SIZE = 5000
    dataset = dataset.shuffle(SHUFFLE_SIZE).repeat(epochs).batch(batch_size)
    dataset = dataset.prefetch(None)
    # Return the dataset. (L)
    return dataset


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Define the model
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    # Train the model
    tf.logging.info("Starting training for {} epoch(s).".format(params.num_epochs))
    # 'Train dateset'
    train_images, train_labels = loadImageLabel(args.data_dir)

    estimator.train(lambda: input_fn(train_images, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE))

    # Evaluate the model on the test set
    tf.logging.info("Evaluation on test set.")
    res = estimator.evaluate(lambda: test_input_fn(args.data_dir, params))
    for key in res:
        print("{}: {}".format(key, res[key]))
