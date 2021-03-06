# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import api_pb2 as api__pb2


class ApiStub(object):
  """The greeting service definition.
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Predict = channel.unary_unary(
        '/Api/Predict',
        request_serializer=api__pb2.PredictRequest.SerializeToString,
        response_deserializer=api__pb2.PredictReply.FromString,
        )


class ApiServicer(object):
  """The greeting service definition.
  """

  def Predict(self, request, context):
    """Sends an array
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_ApiServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Predict': grpc.unary_unary_rpc_method_handler(
          servicer.Predict,
          request_deserializer=api__pb2.PredictRequest.FromString,
          response_serializer=api__pb2.PredictReply.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'Api', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
