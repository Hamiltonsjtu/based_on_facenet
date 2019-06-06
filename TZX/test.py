'''
Created on May 20, 2019

@author: jason
'''

import tensorflow as tf
from keras.utils import plot_model
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
## Import Packages
import os
import sys
import imgaug
import numpy as np
import cv2
import shutil

from mmrcnn.model import ImageMetaLayer, AnchorsLayer

from mmrcnn import model as modellib
import coco
from skimage import io, transform

from keras.models import load_model
from keras.models import Model

import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.models import Model
from keras.layers import BatchNormalization
import os
import numpy as np
import cv2
#from _testbuffer import ndarray

## Paths
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_DIR = os.path.join(ROOT_DIR, 'data/coco')
#COCO_DIR = os.path.join(ROOT_DIR, '/media/jason/Data/eclipse-workspace/yolact/data/coco')
WEIGHTS_DIR = os.path.join(ROOT_DIR, "weights")
DEFAULT_WEIGHTS = os.path.join(ROOT_DIR, "mask_rcnn_512_coco_0027.h5")

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, variable_names_whitelist=freeze_var_names)
        return frozen_graph

def export_frozen_pb(model, frozen_model_path, frozen_model_name):
    from tensorflow.python.framework import graph_io
    K.set_learning_phase(0)
    print(model.inputs)
    print(model.outputs)
    
    #with tf.Session(graph=tf.Graph()) as sess:
    #with tf.keras.backend.get_session() as sess:
    with K.get_session() as sess:
        frozen_graph = freeze_session(sess, output_names=[out.op.name for out in model.outputs])
        #frozen_graph = freeze_session(K.get_session(), output_names=["encoder_output"])
        if not os.path.isdir(frozen_model_path):
            os.mkdir(frozen_model_path)
        #tf.train.write_graph(frozen_graph, frozen_model_path, frozen_model_name, as_text=False)#True
        graph_io.write_graph(frozen_graph, frozen_model_path, frozen_model_name, as_text=False)#True
        print('Frozen graph saved!')
    

def convert_frozen_pb_to_savedmodel(model, frozen_model_path, frozen_model_name, savedmodel_path):
    from tensorflow.python.saved_model import signature_constants
    from tensorflow.python.saved_model import tag_constants
    from tensorflow.python.saved_model.signature_constants import PREDICT_INPUTS
    from tensorflow.python.saved_model.signature_constants import PREDICT_OUTPUTS

    '''graph_pb = frozen_model_path + frozen_model_name
    builder = tf.saved_model.builder.SavedModelBuilder(savedmodel_path)
    
    with tf.gfile.GFile(graph_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())'''
    builder = tf.saved_model.builder.SavedModelBuilder(savedmodel_path)
    signature = {}
        
    #with tf.Session(graph=tf.Graph()) as sess:
    with K.get_session() as sess:
    #with tf.keras.backend.get_session() as sess:
        # name="" is important to ensure we don't get spurious prefixing
        #K.set_learning_phase(0)
        g = tf.get_default_graph()

        input = g.get_tensor_by_name('input_8uc3:0')
        output1 = g.get_tensor_by_name('mask_rcnn/mrcnn_detection/Reshape_1:0')
        output2 = g.get_tensor_by_name('mask_rcnn/mrcnn_mask/Reshape_1:0')
        
        signature[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
                tf.saved_model.signature_def_utils.predict_signature_def(
                                                    {PREDICT_INPUTS: input},
                                                    {'detection': output1, 'mask': output2})
        
        builder.add_meta_graph_and_variables(sess,
                                            [tag_constants.SERVING],
                                            signature_def_map=signature
                                            )
        builder.save()

def freeze_model(model):
    layer_list = model.layers
    for layer in layer_list:
        try: 
            #print(layer.name)
            if type(layer) == Model:
                freeze_model(model)
            else:
                layer.trainable = False
        except Exception as e:
            print(e)

if __name__ == '__main__':
    K.set_learning_phase(0)
    ## Dataset
    class_names = None #['person']  # all classes: None
    dataset_train = coco.CocoDataset()
    dataset_train.load_coco(COCO_DIR, "train", class_names=class_names)
    dataset_train.prepare()
    dataset_val = coco.CocoDataset()
    dataset_val.load_coco(COCO_DIR, "val", class_names=class_names)
    dataset_val.prepare()
    
    ## Model
    config = coco.CocoConfig()
    #print(config.RPN_ANCHOR_SCALES)
    #print(config.RPN_ANCHOR_RATIOS)
    #config.display()
    model = modellib.MaskRCNN(mode="inference", model_dir = MODEL_DIR, config=config)
    model.load_weights(DEFAULT_WEIGHTS, by_name=True)

    image_shape=(512, 288, 3)

    inf_model = model.export_inference_model(image_shape=image_shape)
    freeze_model(inf_model)
    print("Model inputs", inf_model.inputs)
    print("Model outputs", inf_model.outputs)
    
    K.set_learning_phase(0)
    frozen_model_path = '/media/jason/Data/eclipse-workspace/Mobile_Mask_RCNN/mmrcnn_frozen_model/'
    frozen_model_name = 'mmrcnn.pb'    
    saved_model_path = '/media/jason/Data/eclipse-workspace/Mobile_Mask_RCNN/mmrcnn_saved_model/'
    
    if os.path.exists(frozen_model_path):
        shutil.rmtree(frozen_model_path)
    os.mkdir(frozen_model_path)
    if os.path.exists(saved_model_path):
        shutil.rmtree(saved_model_path)

    convert_frozen_pb_to_savedmodel(inf_model, frozen_model_path, frozen_model_name, saved_model_path)
