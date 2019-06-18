import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
import numpy as np
import os

graph_pb = './model/frozen_inference_graph_face.pb'

model_version = 1
export_path_base = 'savedModel'
export_path = os.path.join( tf.compat.as_bytes(export_path_base), tf.compat.as_bytes(str(model_version)))
builder = tf.saved_model.builder.SavedModelBuilder(export_path)


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(graph_pb, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(graph=detection_graph, config=config) as sess:

        images_placeholder = sess.graph.get_tensor_by_name('image_tensor:0')
        out_boxes_placeholder = sess.graph.get_tensor_by_name('detection_boxes:0')
        out_scores_placeholder = sess.graph.get_tensor_by_name('detection_scores:0')
        out_classes_placeholder = sess.graph.get_tensor_by_name('detection_classes:0')
        out_num_detections_placeholder = sess.graph.get_tensor_by_name('num_detections:0')

        tensor_info_images = tf.saved_model.utils.build_tensor_info(images_placeholder)
        tensor_info_boxes = tf.saved_model.utils.build_tensor_info(out_boxes_placeholder)
        tensor_info_scores = tf.saved_model.utils.build_tensor_info(out_scores_placeholder)
        tensor_info_classes = tf.saved_model.utils.build_tensor_info(out_classes_placeholder)
        tensor_info_num_detections = tf.saved_model.utils.build_tensor_info(out_num_detections_placeholder)


        prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                        inputs={'images': tensor_info_images},
                        outputs={'boxes': tensor_info_boxes,
                                 'scores': tensor_info_scores,
                                 'classes': tensor_info_classes,
                                 'num_detection': tensor_info_num_detections
                                 },
                        method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                        'calculate_BBox': prediction_signature
                        },
                main_op=tf.tables_initializer(),
                strip_default_attrs=True)

    builder.save()