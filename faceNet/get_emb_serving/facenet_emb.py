
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

import tensorflow as tf

def img_to_emb_feature(img, channel):
    # print(img.shape)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()

    request.model_spec.name = 'facenet'
    request.model_spec.signature_name = 'calculate_embeddings'
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(img, dtype=tf.float32))
    request.inputs['phase'].CopyFrom(tf.contrib.util.make_tensor_proto(False))
    result_tmp = stub.Predict(request, 10.0)  # 10 secs timeout
    # print(result_tmp)
    result = result_tmp.outputs['embeddings'].float_val

    return result
