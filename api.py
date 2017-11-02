
from __future__ import print_function

# Communication to TensorFlow server via gRPC
from grpc.beta import implementations
import tensorflow as tf
# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from utils import TweetMapper
import grpc
from concurrent import futures
import time

import api_pb2
import api_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


tweet_mapper = TweetMapper()

class Api(api_pb2_grpc.ApiServicer):

  def Predict(self, request, context):
    response = evaluate(request.tweets)
    return api_pb2.PredictReply(predictions=response.outputs['scores'].float_val)

def evaluate(tweets):

  channel = implementations.insecure_channel('localhost', int(9000))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  vec_data = tweet_mapper.vectorize(tweets)
    
  # See prediction_service.proto for gRPC request/response details.
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'sim'
  request.inputs['tweets'].CopyFrom(
      tf.contrib.util.make_tensor_proto(vec_data))
  result = stub.Predict(request, 10.0)  # 10 secs timeout
  return result

def serve():
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  api_pb2_grpc.add_ApiServicer_to_server(Api(), server)
  server.add_insecure_port('[::]:50051')
  server.start()
  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.stop(0)

if __name__ == '__main__':
  serve()
  
