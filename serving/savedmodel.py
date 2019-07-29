#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 09:53:31 2018

@author: zlmo
"""
from __future__ import print_function
import os
import sys
import tensorflow as tf
import os
#import freeze_graph
from tensorflow.python.platform import gfile

model_version = 1

#export_path='/home/zlmo/serving/test'
export_path_base = '2018_serable'
export_path = os.path.join(
tf.compat.as_bytes(export_path_base),
tf.compat.as_bytes(str(model_version)))
builder = tf.saved_model.builder.SavedModelBuilder(export_path)


with tf.Session() as sess:
    
    saver = tf.train.import_meta_graph('./2017_raw/model-20170512-110547.meta')
    graph = tf.get_default_graph()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './2017/model-20170512-110547.ckpt-250000')
    """
    with gfile.FastGFile('./darknet19.pb','rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            output = tf.import_graph_def(graph_def, input_map=None, name='')
    #print(sess.run(output))
    #input_graph_def = sess.graph.as_graph_def()
    """
    
    """
    whitelist_names=[]
    for node in graph_def.node:
        if (node.name.startswith('InceptionResnet') or node.name.startswith('embeddings') or 
                node.name.startswith('image_batch') or node.name.startswith('label_batch') or
                node.name.startswith('phase_train') or node.name.startswith('Logits')):
            whitelist_names.append(node.name)
        print(node.name)
        #if (node.name.startswith('input') or node.name.startwith('avgpool')):
            #print(node.name)

        #if(node.name.startswith('input')): 
            #print(node)
    """

    #frozen_graph = freeze_graph.freeze_graph_def(sess, graph_def, "embeddings")
    #tf.import_graph_def(frozen_graph)
        
    images_placeholder = sess.graph.get_tensor_by_name("input:0")
    embeddings = sess.graph.get_tensor_by_name("embeddings:0")
    phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")

    tensor_info_images = tf.saved_model.utils.build_tensor_info(images_placeholder)
    tensor_info_embeddings = tf.saved_model.utils.build_tensor_info(embeddings)
    tensor_info_phase = tf.saved_model.utils.build_tensor_info(phase_train_placeholder)
        
    prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'images': tensor_info_images,'phase':tensor_info_phase},
                    outputs={'embeddings': tensor_info_embeddings},
                    method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    
    builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                    'calculate_embeddings': prediction_signature
                    #tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:classification_signature
                    },
            main_op=tf.tables_initializer(),
            strip_default_attrs=True)
builder.save()
