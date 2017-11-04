
from __future__ import print_function

# Communication to TensorFlow server via gRPC
from grpc.beta import implementations
import tensorflow as tf
# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from utils import TweetMapper

import tensorflow as tf

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')

FLAGS = tf.app.flags.FLAGS


def main(_):
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  tweet_mapper = TweetMapper()
  # Send request
  data = tweet_mapper.vectorize(["Hoy fue un dia de mierda","Me encanta jugar en el parque"])
  # See prediction_service.proto for gRPC request/response details.
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'sim'
  request.inputs['tweets'].CopyFrom(
      tf.contrib.util.make_tensor_proto(data))
  result = stub.Predict(request, 10.0)  # 10 secs timeout
  print(result)


if __name__ == '__main__':
  tf.app.run()
