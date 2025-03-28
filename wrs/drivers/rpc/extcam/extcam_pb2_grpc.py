# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from . import extcam_pb2 as extcam_b2


class CamStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.getimg = channel.unary_unary(
        '/Cam/getimg',
        request_serializer=extcam_b2.Empty.SerializeToString,
        response_deserializer=extcam_b2.CamImg.FromString,
        )


class CamServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def getimg(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_CamServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'getimg': grpc.unary_unary_rpc_method_handler(
          servicer.getimg,
          request_deserializer=extcam_b2.Empty.FromString,
          response_serializer=extcam_b2.CamImg.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'Cam', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
