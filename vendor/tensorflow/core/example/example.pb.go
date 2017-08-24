// Code generated by protoc-gen-go. DO NOT EDIT.
// source: tensorflow/core/example/example.proto

/*
Package tensorflow is a generated protocol buffer package.

It is generated from these files:
	tensorflow/core/example/example.proto
	tensorflow/core/example/feature.proto

It has these top-level messages:
	Example
	SequenceExample
	BytesList
	FloatList
	Int64List
	Feature
	Features
	FeatureList
	FeatureLists
*/
package tensorflow

import proto "github.com/golang/protobuf/proto"
import fmt "fmt"
import math "math"

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.ProtoPackageIsVersion2 // please upgrade the proto package

type Example struct {
	Features *Features `protobuf:"bytes,1,opt,name=features" json:"features,omitempty"`
}

func (m *Example) Reset()                    { *m = Example{} }
func (m *Example) String() string            { return proto.CompactTextString(m) }
func (*Example) ProtoMessage()               {}
func (*Example) Descriptor() ([]byte, []int) { return fileDescriptor0, []int{0} }

func (m *Example) GetFeatures() *Features {
	if m != nil {
		return m.Features
	}
	return nil
}

type SequenceExample struct {
	Context      *Features     `protobuf:"bytes,1,opt,name=context" json:"context,omitempty"`
	FeatureLists *FeatureLists `protobuf:"bytes,2,opt,name=feature_lists,json=featureLists" json:"feature_lists,omitempty"`
}

func (m *SequenceExample) Reset()                    { *m = SequenceExample{} }
func (m *SequenceExample) String() string            { return proto.CompactTextString(m) }
func (*SequenceExample) ProtoMessage()               {}
func (*SequenceExample) Descriptor() ([]byte, []int) { return fileDescriptor0, []int{1} }

func (m *SequenceExample) GetContext() *Features {
	if m != nil {
		return m.Context
	}
	return nil
}

func (m *SequenceExample) GetFeatureLists() *FeatureLists {
	if m != nil {
		return m.FeatureLists
	}
	return nil
}

func init() {
	proto.RegisterType((*Example)(nil), "tensorflow.Example")
	proto.RegisterType((*SequenceExample)(nil), "tensorflow.SequenceExample")
}

func init() { proto.RegisterFile("tensorflow/core/example/example.proto", fileDescriptor0) }

var fileDescriptor0 = []byte{
	// 190 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0x52, 0x2d, 0x49, 0xcd, 0x2b,
	0xce, 0x2f, 0x4a, 0xcb, 0xc9, 0x2f, 0xd7, 0x4f, 0xce, 0x2f, 0x4a, 0xd5, 0x4f, 0xad, 0x48, 0xcc,
	0x2d, 0xc8, 0x81, 0xd3, 0x7a, 0x05, 0x45, 0xf9, 0x25, 0xf9, 0x42, 0x5c, 0x08, 0x65, 0x52, 0x38,
	0xb5, 0xa4, 0xa5, 0x26, 0x96, 0x94, 0x16, 0x41, 0xb5, 0x28, 0x59, 0x73, 0xb1, 0xbb, 0x42, 0x24,
	0x84, 0x0c, 0xb8, 0x38, 0xa0, 0x72, 0xc5, 0x12, 0x8c, 0x0a, 0x8c, 0x1a, 0xdc, 0x46, 0x22, 0x7a,
	0x08, 0x43, 0xf4, 0xdc, 0xa0, 0x72, 0x41, 0x70, 0x55, 0x4a, 0x0d, 0x8c, 0x5c, 0xfc, 0xc1, 0xa9,
	0x85, 0xa5, 0xa9, 0x79, 0xc9, 0xa9, 0x30, 0x53, 0xf4, 0xb8, 0xd8, 0x93, 0xf3, 0xf3, 0x4a, 0x52,
	0x2b, 0x4a, 0xf0, 0x1a, 0x02, 0x53, 0x24, 0x64, 0xcb, 0xc5, 0x0b, 0x35, 0x2f, 0x3e, 0x27, 0xb3,
	0xb8, 0xa4, 0x58, 0x82, 0x09, 0xac, 0x4b, 0x02, 0x8b, 0x2e, 0x1f, 0x90, 0x7c, 0x10, 0x4f, 0x1a,
	0x12, 0xcf, 0x49, 0x87, 0x4b, 0x2c, 0xbf, 0x28, 0x1d, 0x59, 0x31, 0xd4, 0x9f, 0x4e, 0xbc, 0x50,
	0x17, 0x05, 0x80, 0xfc, 0x59, 0x1c, 0xc0, 0xf8, 0x83, 0x91, 0x31, 0x89, 0x0d, 0xec, 0x69, 0x63,
	0x40, 0x00, 0x00, 0x00, 0xff, 0xff, 0x98, 0x79, 0xef, 0x4a, 0x50, 0x01, 0x00, 0x00,
}
