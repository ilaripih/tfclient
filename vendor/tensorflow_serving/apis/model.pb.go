// Code generated by protoc-gen-go. DO NOT EDIT.
// source: tensorflow_serving/apis/model.proto

package tensorflow_serving

import proto "github.com/golang/protobuf/proto"
import fmt "fmt"
import math "math"
import google_protobuf "github.com/golang/protobuf/ptypes/wrappers"

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// Metadata for an inference request such as the model name and version.
type ModelSpec struct {
	// Required servable name.
	Name string `protobuf:"bytes,1,opt,name=name" json:"name,omitempty"`
	// Optional choice of which version of the model to use.
	//
	// Recommended to be left unset in the common case. Should be specified only
	// when there is a strong version consistency requirement.
	//
	// When left unspecified, the system will serve the best available version.
	// This is typically the latest version, though during version transitions,
	// notably when serving on a fleet of instances, may be either the previous or
	// new version.
	//
	// Types that are valid to be assigned to VersionChoice:
	//	*ModelSpec_Version
	//	*ModelSpec_VersionLabel
	VersionChoice isModelSpec_VersionChoice `protobuf_oneof:"version_choice"`
	// A named signature to evaluate. If unspecified, the default signature will
	// be used.
	SignatureName string `protobuf:"bytes,3,opt,name=signature_name,json=signatureName" json:"signature_name,omitempty"`
}

func (m *ModelSpec) Reset()                    { *m = ModelSpec{} }
func (m *ModelSpec) String() string            { return proto.CompactTextString(m) }
func (*ModelSpec) ProtoMessage()               {}
func (*ModelSpec) Descriptor() ([]byte, []int) { return fileDescriptor4, []int{0} }

type isModelSpec_VersionChoice interface {
	isModelSpec_VersionChoice()
}

type ModelSpec_Version struct {
	Version *google_protobuf.Int64Value `protobuf:"bytes,2,opt,name=version,oneof"`
}
type ModelSpec_VersionLabel struct {
	VersionLabel string `protobuf:"bytes,4,opt,name=version_label,json=versionLabel,oneof"`
}

func (*ModelSpec_Version) isModelSpec_VersionChoice()      {}
func (*ModelSpec_VersionLabel) isModelSpec_VersionChoice() {}

func (m *ModelSpec) GetVersionChoice() isModelSpec_VersionChoice {
	if m != nil {
		return m.VersionChoice
	}
	return nil
}

func (m *ModelSpec) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *ModelSpec) GetVersion() *google_protobuf.Int64Value {
	if x, ok := m.GetVersionChoice().(*ModelSpec_Version); ok {
		return x.Version
	}
	return nil
}

func (m *ModelSpec) GetVersionLabel() string {
	if x, ok := m.GetVersionChoice().(*ModelSpec_VersionLabel); ok {
		return x.VersionLabel
	}
	return ""
}

func (m *ModelSpec) GetSignatureName() string {
	if m != nil {
		return m.SignatureName
	}
	return ""
}

// XXX_OneofFuncs is for the internal use of the proto package.
func (*ModelSpec) XXX_OneofFuncs() (func(msg proto.Message, b *proto.Buffer) error, func(msg proto.Message, tag, wire int, b *proto.Buffer) (bool, error), func(msg proto.Message) (n int), []interface{}) {
	return _ModelSpec_OneofMarshaler, _ModelSpec_OneofUnmarshaler, _ModelSpec_OneofSizer, []interface{}{
		(*ModelSpec_Version)(nil),
		(*ModelSpec_VersionLabel)(nil),
	}
}

func _ModelSpec_OneofMarshaler(msg proto.Message, b *proto.Buffer) error {
	m := msg.(*ModelSpec)
	// version_choice
	switch x := m.VersionChoice.(type) {
	case *ModelSpec_Version:
		b.EncodeVarint(2<<3 | proto.WireBytes)
		if err := b.EncodeMessage(x.Version); err != nil {
			return err
		}
	case *ModelSpec_VersionLabel:
		b.EncodeVarint(4<<3 | proto.WireBytes)
		b.EncodeStringBytes(x.VersionLabel)
	case nil:
	default:
		return fmt.Errorf("ModelSpec.VersionChoice has unexpected type %T", x)
	}
	return nil
}

func _ModelSpec_OneofUnmarshaler(msg proto.Message, tag, wire int, b *proto.Buffer) (bool, error) {
	m := msg.(*ModelSpec)
	switch tag {
	case 2: // version_choice.version
		if wire != proto.WireBytes {
			return true, proto.ErrInternalBadWireType
		}
		msg := new(google_protobuf.Int64Value)
		err := b.DecodeMessage(msg)
		m.VersionChoice = &ModelSpec_Version{msg}
		return true, err
	case 4: // version_choice.version_label
		if wire != proto.WireBytes {
			return true, proto.ErrInternalBadWireType
		}
		x, err := b.DecodeStringBytes()
		m.VersionChoice = &ModelSpec_VersionLabel{x}
		return true, err
	default:
		return false, nil
	}
}

func _ModelSpec_OneofSizer(msg proto.Message) (n int) {
	m := msg.(*ModelSpec)
	// version_choice
	switch x := m.VersionChoice.(type) {
	case *ModelSpec_Version:
		s := proto.Size(x.Version)
		n += proto.SizeVarint(2<<3 | proto.WireBytes)
		n += proto.SizeVarint(uint64(s))
		n += s
	case *ModelSpec_VersionLabel:
		n += proto.SizeVarint(4<<3 | proto.WireBytes)
		n += proto.SizeVarint(uint64(len(x.VersionLabel)))
		n += len(x.VersionLabel)
	case nil:
	default:
		panic(fmt.Sprintf("proto: unexpected type %T in oneof", x))
	}
	return n
}

func init() {
	proto.RegisterType((*ModelSpec)(nil), "tensorflow.serving.ModelSpec")
}

func init() { proto.RegisterFile("tensorflow_serving/apis/model.proto", fileDescriptor4) }

var fileDescriptor4 = []byte{
	// 233 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x44, 0x8e, 0xc1, 0x4b, 0xc3, 0x30,
	0x14, 0xc6, 0x17, 0x37, 0x94, 0x45, 0x37, 0x24, 0xa7, 0xa2, 0x20, 0x43, 0x19, 0xec, 0x94, 0x80,
	0x8a, 0xde, 0x77, 0x9a, 0xa0, 0x1e, 0x2a, 0x78, 0x2d, 0x69, 0x7d, 0x8b, 0x81, 0x34, 0x2f, 0x24,
	0xe9, 0xfa, 0xaf, 0xf9, 0xa7, 0x79, 0x94, 0xa6, 0xad, 0xbb, 0x7d, 0x7c, 0xfc, 0xde, 0xef, 0x7b,
	0xf4, 0x2e, 0x82, 0x0d, 0xe8, 0xf7, 0x06, 0xdb, 0x22, 0x80, 0x3f, 0x68, 0xab, 0x84, 0x74, 0x3a,
	0x88, 0x1a, 0xbf, 0xc0, 0x70, 0xe7, 0x31, 0x22, 0x63, 0x47, 0x88, 0x0f, 0xd0, 0xd5, 0x8d, 0x42,
	0x54, 0x06, 0x44, 0x22, 0xca, 0x66, 0x2f, 0x5a, 0x2f, 0x9d, 0x03, 0x1f, 0xfa, 0x9b, 0xdb, 0x1f,
	0x42, 0xe7, 0x6f, 0x9d, 0xe3, 0xc3, 0x41, 0xc5, 0x18, 0x9d, 0x59, 0x59, 0x43, 0x46, 0x56, 0x64,
	0x33, 0xcf, 0x53, 0x66, 0xcf, 0xf4, 0xec, 0x00, 0x3e, 0x68, 0xb4, 0xd9, 0xc9, 0x8a, 0x6c, 0xce,
	0xef, 0xaf, 0x79, 0xef, 0xe4, 0xa3, 0x93, 0xbf, 0xd8, 0xf8, 0xf4, 0xf8, 0x29, 0x4d, 0x03, 0xbb,
	0x49, 0x3e, 0xd2, 0x6c, 0x4d, 0x17, 0x43, 0x2c, 0x8c, 0x2c, 0xc1, 0x64, 0xb3, 0xce, 0xba, 0x9b,
	0xe4, 0x17, 0x43, 0xfd, 0xda, 0xb5, 0x6c, 0x4d, 0x97, 0x41, 0x2b, 0x2b, 0x63, 0xe3, 0xa1, 0x48,
	0xeb, 0xd3, 0xb4, 0xbe, 0xf8, 0x6f, 0xdf, 0x65, 0x0d, 0xdb, 0x4b, 0xba, 0x1c, 0x6d, 0xd5, 0x37,
	0xea, 0x0a, 0xb6, 0xd3, 0x5f, 0x42, 0xca, 0xd3, 0xf4, 0xc4, 0xc3, 0x5f, 0x00, 0x00, 0x00, 0xff,
	0xff, 0xd5, 0x26, 0x44, 0xd1, 0x21, 0x01, 0x00, 0x00,
}
