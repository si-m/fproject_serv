# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: api.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='api.proto',
  package='',
  syntax='proto3',
  serialized_pb=_b('\n\tapi.proto\" \n\x0ePredictRequest\x12\x0e\n\x06tweets\x18\x01 \x03(\t\"\'\n\x0cPredictReply\x12\x17\n\x0bpredictions\x18\x01 \x03(\x02\x42\x02\x10\x01\x32\x32\n\x03\x41pi\x12+\n\x07Predict\x12\x0f.PredictRequest\x1a\r.PredictReply\"\x00\x62\x06proto3')
)




_PREDICTREQUEST = _descriptor.Descriptor(
  name='PredictRequest',
  full_name='PredictRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='tweets', full_name='PredictRequest.tweets', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=13,
  serialized_end=45,
)


_PREDICTREPLY = _descriptor.Descriptor(
  name='PredictReply',
  full_name='PredictReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='predictions', full_name='PredictReply.predictions', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=47,
  serialized_end=86,
)

DESCRIPTOR.message_types_by_name['PredictRequest'] = _PREDICTREQUEST
DESCRIPTOR.message_types_by_name['PredictReply'] = _PREDICTREPLY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PredictRequest = _reflection.GeneratedProtocolMessageType('PredictRequest', (_message.Message,), dict(
  DESCRIPTOR = _PREDICTREQUEST,
  __module__ = 'api_pb2'
  # @@protoc_insertion_point(class_scope:PredictRequest)
  ))
_sym_db.RegisterMessage(PredictRequest)

PredictReply = _reflection.GeneratedProtocolMessageType('PredictReply', (_message.Message,), dict(
  DESCRIPTOR = _PREDICTREPLY,
  __module__ = 'api_pb2'
  # @@protoc_insertion_point(class_scope:PredictReply)
  ))
_sym_db.RegisterMessage(PredictReply)


_PREDICTREPLY.fields_by_name['predictions'].has_options = True
_PREDICTREPLY.fields_by_name['predictions']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))

_API = _descriptor.ServiceDescriptor(
  name='Api',
  full_name='Api',
  file=DESCRIPTOR,
  index=0,
  options=None,
  serialized_start=88,
  serialized_end=138,
  methods=[
  _descriptor.MethodDescriptor(
    name='Predict',
    full_name='Api.Predict',
    index=0,
    containing_service=None,
    input_type=_PREDICTREQUEST,
    output_type=_PREDICTREPLY,
    options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_API)

DESCRIPTOR.services_by_name['Api'] = _API

# @@protoc_insertion_point(module_scope)
