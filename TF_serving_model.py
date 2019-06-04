import os, argparse

import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter


model_exporter = exporter.Exporter(2017)

with tf.Session() as sess:
    model_exporter.init(sess.graph.as_graph_def(),
    named_graph_signatures={
        'inputs': exporter.generic_signature({'x': x}),
        'outputs': exporter.generic_signature({'y': y_pred})})
    model_exporter.export(FLAGS.work_dir,
                      tf.constant(FLAGS.export_version), sess)