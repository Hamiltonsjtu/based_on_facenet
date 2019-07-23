
from __future__ import print_function
import os
import sys
import tensorflow as tf
import os
# import freeze_graph
from tensorflow.python.platform import gfile

model_version = 1

export_path_base = './MTCNN_severable_zs'
export_path = export_path_base + '/' + str(model_version)

if tf.gfile.Exists(export_path_base):
    tf.gfile.DeleteRecursively(export_path_base)


builder = tf.saved_model.builder.SavedModelBuilder(export_path)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./allinone/allinone.ckpt.meta')
    graph = tf.get_default_graph()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './allinone/allinone.ckpt')

    pnet_images_placeholder = sess.graph.get_tensor_by_name('pnet/input:0')
    pnet_face_placeholder = sess.graph.get_tensor_by_name('pnet/prob1:0')
    pnet_bbox_placeholder = sess.graph.get_tensor_by_name('pnet/conv4-2/BiasAdd:0')
    rnet_images_placeholder = sess.graph.get_tensor_by_name('rnet/input:0')
    rnet_face_placeholder = sess.graph.get_tensor_by_name('rnet/prob1:0')
    rnet_bbox_placeholder = sess.graph.get_tensor_by_name('rnet/conv5-2/conv5-2:0')
    onet_images_placeholder = sess.graph.get_tensor_by_name('onet/input:0')
    onet_face_placeholder = sess.graph.get_tensor_by_name('onet/prob1:0')
    onet_bbox_placeholder = sess.graph.get_tensor_by_name('onet/conv6-2/conv6-2:0')
    onet_landmark_placeholder = sess.graph.get_tensor_by_name('onet/conv6-3/conv6-3:0')


    tensor_info_pnet_images = tf.saved_model.utils.build_tensor_info(pnet_images_placeholder)
    tensor_info_pnet_face = tf.saved_model.utils.build_tensor_info(pnet_face_placeholder)
    tensor_info_pnet_bbox = tf.saved_model.utils.build_tensor_info(pnet_bbox_placeholder)
    tensor_info_rnet_images = tf.saved_model.utils.build_tensor_info(rnet_images_placeholder)
    tensor_info_rnet_face = tf.saved_model.utils.build_tensor_info(rnet_face_placeholder)
    tensor_info_rnet_bbox = tf.saved_model.utils.build_tensor_info(rnet_bbox_placeholder)
    tensor_info_onet_images = tf.saved_model.utils.build_tensor_info(onet_images_placeholder)
    tensor_info_onet_face = tf.saved_model.utils.build_tensor_info(onet_face_placeholder)
    tensor_info_onet_bbox = tf.saved_model.utils.build_tensor_info(onet_bbox_placeholder)
    tensor_info_onet_landmark = tf.saved_model.utils.build_tensor_info(onet_landmark_placeholder)


    prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'images': tensor_info_pnet_images},
                    outputs={'pnet_face': tensor_info_pnet_face,
                             'pnet_bbox': tensor_info_pnet_bbox,
                             'rnet_face': tensor_info_rnet_face,
                             'rnet_bbox': tensor_info_rnet_bbox,
                             'onet_face': tensor_info_onet_face,
                             'onet_bbox': tensor_info_onet_bbox,
                             'onet_landmark': tensor_info_onet_landmark,
                            },
                    method_name = tf.saved_model.signature_constants.REGRESS_METHOD_NAME))

    builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                    'all_ouput': prediction_signature
                    #tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:classification_signature
                    },
            main_op=tf.tables_initializer())


builder.save()

# THE FOLLOW COMMAND CAN HELP TO CHECK DETAILS FOR SAVEDMODEL, RUN IN TERMINAL
# python D:\Anaconda\envs\facenet\Lib\site-packages\tensorflow\python\tools\saved_model_cli.py show --dir E:\shuai\Face\faceNet\2017_servable\1 --all
