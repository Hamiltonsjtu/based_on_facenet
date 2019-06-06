#coding:utf-8
import sys
import os
import io
from tensorflow.python.platform import gfile
import tensorflow as tf
# sys.path.append('src/')
import re


def load_model(model, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

graph = tf.Graph()

with graph.as_default():
    with tf.Session() as sess:
        load_model('2017_raw')
        # print(graph.get_name_scope())


        # 建立签名映射，需要包括计算图中的placeholder（ChatInputs, SegInputs, Dropout）和我们需要的结果（project/logits,crf_loss/transitions）
        """
        build_tensor_info：建立一个基于提供的参数构造的TensorInfo protocol buffer，
        输入：tensorflow graph中的tensor；
        输出：基于提供的参数（tensor）构建的包含TensorInfo的protocol buffer
                    get_operation_by_name：通过name获取checkpoint中保存的变量，能够进行这一步的前提是在模型保存的时候给对应的变量赋予name
        """

        '''image_inputs =tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("input").outputs[0])
        phase_train =tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("phase_train").outputs[0])
        emb_out =tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("embeddings").outputs[0])'''

        print('Input tensor: ', tf.get_default_graph().get_tensor_by_name("input:0"))

        image_inputs = tf.saved_model.utils.build_tensor_info(tf.get_default_graph().get_tensor_by_name("input:0"))
        emb_out = tf.saved_model.utils.build_tensor_info(tf.get_default_graph().get_tensor_by_name("embeddings:0"))
        phase_train = tf.saved_model.utils.build_tensor_info(tf.get_default_graph().get_tensor_by_name("phase_train:0"))

        """
        signature_constants：SavedModel保存和恢复操作的签名常量。
        在序列标注的任务中，这里的method_name是"tensorflow/serving/predict"
        """
        # 定义模型的输入输出，建立调用接口与tensor签名之间的映射
        labeling_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    "imageinputs":
                        image_inputs,
                    "phase_train":
                        phase_train
                },
                outputs={
                    "embedding":
                        emb_out
                },
                method_name= tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        """
        tf.group : 创建一个将多个操作分组的操作，返回一个可以执行所有输入的操作
        """
        # legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')


        """
        add_meta_graph_and_variables：建立一个Saver来保存session中的变量，
                                      输出对应的原图的定义，这个函数假设保存的变量已经被初始化；
                                      对于一个SavedModelBuilder，这个API必须被调用一次来保存meta graph；
                                      对于后面添加的图结构，可以使用函数 add_meta_graph()来进行添加
        """

        export_path_base = '2017_servable'
        count = 1
        export_path = os.path.join(
            tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(count)))
        print('Exporting trained model to', export_path)
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        # 建立模型名称与模型签名之间的映射
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            # 保存模型的方法名，与客户端的request.model_spec.signature_name对应
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: labeling_signature}, strip_default_attrs=True
            )
        print('========================================')
        print(tf.trainable_variables())
        builder.save()
        print("Build Done")


#
# ### 测试模型转换
# tf.app.flags.DEFINE_string("ckpt_path",     "e:/shuai/Face/2017",             "path of source checkpoints")
# tf.app.flags.DEFINE_string("pb_path",       "e:shuai/Face/2017/servable-models",             "path of servable models")
# tf.app.flags.DEFINE_integer("version",      1,              "the number of model version")
# tf.app.flags.DEFINE_string("classes",       '20170512-110547',          "multi-models to be converted")
# FLAGS = tf.flags.FLAGS
#
# classes = FLAGS.classes
# input_checkpoint = FLAGS.ckpt_path  + "/" + classes
# model_path = FLAGS.pb_path + '/' + classes
# print('model path {}'.format(model_path))
#
# # 版本号控制
# count = FLAGS.version
# modify = False
#
#
# if not os.path.exists(model_path):
#     os.makedirs(model_path)
# else:
#     for v in os.listdir(model_path):
#         print(type(v), v)
#         if int(v) >= count:
#             count = int(v)
#             modify = True
#     if modify:
#         count += 1
#
# # 模型格式转换
# restore_and_save(input_checkpoint, model_path)