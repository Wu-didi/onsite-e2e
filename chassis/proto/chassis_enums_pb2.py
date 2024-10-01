# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: chassis/proto/chassis_enums.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='chassis/proto/chassis_enums.proto',
  package='vts.protocol.chassis',
  syntax='proto3',
  serialized_pb=_b('\n!chassis/proto/chassis_enums.proto\x12\x14vts.protocol.chassis*b\n\x07MsgType\x12\x14\n\x10VEHICLE_REGISTER\x10\x00\x12\x16\n\x12VEHICLE_UNREGISTER\x10\x01\x12\x13\n\x0fVEHICLE_CONTROL\x10\x02\x12\x14\n\x10VEHICLE_FEEDBACK\x10\x03*Y\n\x0b\x44rivingMode\x12\x07\n\x03OFF\x10\x00\x12\n\n\x06MANUAL\x10\x01\x12\x16\n\x12\x41UTONOMOUS_DRIVING\x10\x02\x12\t\n\x05\x46\x41ULT\x10\x03\x12\x12\n\x0eREMOTE_DRIVING\x10\x04*]\n\x0cVehicleError\x12\x06\n\x02OK\x10\x00\x12\x16\n\x12VEHICLE_REGISTERED\x10\x01\x12\x16\n\x12TESTCASE_NOT_READY\x10\x02\x12\x15\n\x11NO_MAP_FILE_FOUND\x10\x03*U\n\x13SteeringControlMode\x12\x1e\n\x1aTARGET_STEERING_ANGLE_MODE\x10\x00\x12\x1e\n\x1a\x41\x43TUAL_STEERING_ANGLE_MODE\x10\x01*@\n\x0fGearControlMode\x12\x17\n\x13\x41UTOMATIC_GEAR_MODE\x10\x00\x12\x14\n\x10MANUAL_GEAR_MODE\x10\x01*Z\n\x12\x44rivingControlMode\x12!\n\x1dTARGET_ACCELERATOR_PEDAL_MODE\x10\x00\x12!\n\x1d\x41\x43TUAL_ACCELERATOR_PEDAL_MODE\x10\x01*L\n\x10\x42rakeControlMode\x12\x1b\n\x17TARGET_BRAKE_PEDAL_MODE\x10\x00\x12\x1b\n\x17\x41\x43TUAL_BRAKE_PEDAL_MODE\x10\x01\x62\x06proto3')
)

_MSGTYPE = _descriptor.EnumDescriptor(
  name='MsgType',
  full_name='vts.protocol.chassis.MsgType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='VEHICLE_REGISTER', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VEHICLE_UNREGISTER', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VEHICLE_CONTROL', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VEHICLE_FEEDBACK', index=3, number=3,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=59,
  serialized_end=157,
)
_sym_db.RegisterEnumDescriptor(_MSGTYPE)

MsgType = enum_type_wrapper.EnumTypeWrapper(_MSGTYPE)
_DRIVINGMODE = _descriptor.EnumDescriptor(
  name='DrivingMode',
  full_name='vts.protocol.chassis.DrivingMode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='OFF', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MANUAL', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AUTONOMOUS_DRIVING', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FAULT', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='REMOTE_DRIVING', index=4, number=4,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=159,
  serialized_end=248,
)
_sym_db.RegisterEnumDescriptor(_DRIVINGMODE)

DrivingMode = enum_type_wrapper.EnumTypeWrapper(_DRIVINGMODE)
_VEHICLEERROR = _descriptor.EnumDescriptor(
  name='VehicleError',
  full_name='vts.protocol.chassis.VehicleError',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='OK', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VEHICLE_REGISTERED', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TESTCASE_NOT_READY', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NO_MAP_FILE_FOUND', index=3, number=3,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=250,
  serialized_end=343,
)
_sym_db.RegisterEnumDescriptor(_VEHICLEERROR)

VehicleError = enum_type_wrapper.EnumTypeWrapper(_VEHICLEERROR)
_STEERINGCONTROLMODE = _descriptor.EnumDescriptor(
  name='SteeringControlMode',
  full_name='vts.protocol.chassis.SteeringControlMode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='TARGET_STEERING_ANGLE_MODE', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ACTUAL_STEERING_ANGLE_MODE', index=1, number=1,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=345,
  serialized_end=430,
)
_sym_db.RegisterEnumDescriptor(_STEERINGCONTROLMODE)

SteeringControlMode = enum_type_wrapper.EnumTypeWrapper(_STEERINGCONTROLMODE)
_GEARCONTROLMODE = _descriptor.EnumDescriptor(
  name='GearControlMode',
  full_name='vts.protocol.chassis.GearControlMode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='AUTOMATIC_GEAR_MODE', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MANUAL_GEAR_MODE', index=1, number=1,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=432,
  serialized_end=496,
)
_sym_db.RegisterEnumDescriptor(_GEARCONTROLMODE)

GearControlMode = enum_type_wrapper.EnumTypeWrapper(_GEARCONTROLMODE)
_DRIVINGCONTROLMODE = _descriptor.EnumDescriptor(
  name='DrivingControlMode',
  full_name='vts.protocol.chassis.DrivingControlMode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='TARGET_ACCELERATOR_PEDAL_MODE', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ACTUAL_ACCELERATOR_PEDAL_MODE', index=1, number=1,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=498,
  serialized_end=588,
)
_sym_db.RegisterEnumDescriptor(_DRIVINGCONTROLMODE)

DrivingControlMode = enum_type_wrapper.EnumTypeWrapper(_DRIVINGCONTROLMODE)
_BRAKECONTROLMODE = _descriptor.EnumDescriptor(
  name='BrakeControlMode',
  full_name='vts.protocol.chassis.BrakeControlMode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='TARGET_BRAKE_PEDAL_MODE', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ACTUAL_BRAKE_PEDAL_MODE', index=1, number=1,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=590,
  serialized_end=666,
)
_sym_db.RegisterEnumDescriptor(_BRAKECONTROLMODE)

BrakeControlMode = enum_type_wrapper.EnumTypeWrapper(_BRAKECONTROLMODE)
VEHICLE_REGISTER = 0
VEHICLE_UNREGISTER = 1
VEHICLE_CONTROL = 2
VEHICLE_FEEDBACK = 3
OFF = 0
MANUAL = 1
AUTONOMOUS_DRIVING = 2
FAULT = 3
REMOTE_DRIVING = 4
OK = 0
VEHICLE_REGISTERED = 1
TESTCASE_NOT_READY = 2
NO_MAP_FILE_FOUND = 3
TARGET_STEERING_ANGLE_MODE = 0
ACTUAL_STEERING_ANGLE_MODE = 1
AUTOMATIC_GEAR_MODE = 0
MANUAL_GEAR_MODE = 1
TARGET_ACCELERATOR_PEDAL_MODE = 0
ACTUAL_ACCELERATOR_PEDAL_MODE = 1
TARGET_BRAKE_PEDAL_MODE = 0
ACTUAL_BRAKE_PEDAL_MODE = 1


DESCRIPTOR.enum_types_by_name['MsgType'] = _MSGTYPE
DESCRIPTOR.enum_types_by_name['DrivingMode'] = _DRIVINGMODE
DESCRIPTOR.enum_types_by_name['VehicleError'] = _VEHICLEERROR
DESCRIPTOR.enum_types_by_name['SteeringControlMode'] = _STEERINGCONTROLMODE
DESCRIPTOR.enum_types_by_name['GearControlMode'] = _GEARCONTROLMODE
DESCRIPTOR.enum_types_by_name['DrivingControlMode'] = _DRIVINGCONTROLMODE
DESCRIPTOR.enum_types_by_name['BrakeControlMode'] = _BRAKECONTROLMODE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)


# @@protoc_insertion_point(module_scope)
