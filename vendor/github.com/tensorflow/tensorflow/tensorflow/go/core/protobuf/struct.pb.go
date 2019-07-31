// Code generated by protoc-gen-go. DO NOT EDIT.
// source: tensorflow/core/protobuf/struct.proto

package protobuf

import proto "github.com/golang/protobuf/proto"
import fmt "fmt"
import math "math"
import tensorflow "github.com/tensorflow/tensorflow/tensorflow/go/core/framework"
import tensorflow1 "github.com/tensorflow/tensorflow/tensorflow/go/core/framework"

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

type TypeSpecProto_TypeSpecClass int32

const (
	TypeSpecProto_UNKNOWN             TypeSpecProto_TypeSpecClass = 0
	TypeSpecProto_SPARSE_TENSOR_SPEC  TypeSpecProto_TypeSpecClass = 1
	TypeSpecProto_INDEXED_SLICES_SPEC TypeSpecProto_TypeSpecClass = 2
	TypeSpecProto_RAGGED_TENSOR_SPEC  TypeSpecProto_TypeSpecClass = 3
	TypeSpecProto_TENSOR_ARRAY_SPEC   TypeSpecProto_TypeSpecClass = 4
	TypeSpecProto_DATA_DATASET_SPEC   TypeSpecProto_TypeSpecClass = 5
	TypeSpecProto_DATA_ITERATOR_SPEC  TypeSpecProto_TypeSpecClass = 6
	TypeSpecProto_OPTIONAL_SPEC       TypeSpecProto_TypeSpecClass = 7
	TypeSpecProto_PER_REPLICA_SPEC    TypeSpecProto_TypeSpecClass = 8
)

var TypeSpecProto_TypeSpecClass_name = map[int32]string{
	0: "UNKNOWN",
	1: "SPARSE_TENSOR_SPEC",
	2: "INDEXED_SLICES_SPEC",
	3: "RAGGED_TENSOR_SPEC",
	4: "TENSOR_ARRAY_SPEC",
	5: "DATA_DATASET_SPEC",
	6: "DATA_ITERATOR_SPEC",
	7: "OPTIONAL_SPEC",
	8: "PER_REPLICA_SPEC",
}
var TypeSpecProto_TypeSpecClass_value = map[string]int32{
	"UNKNOWN":             0,
	"SPARSE_TENSOR_SPEC":  1,
	"INDEXED_SLICES_SPEC": 2,
	"RAGGED_TENSOR_SPEC":  3,
	"TENSOR_ARRAY_SPEC":   4,
	"DATA_DATASET_SPEC":   5,
	"DATA_ITERATOR_SPEC":  6,
	"OPTIONAL_SPEC":       7,
	"PER_REPLICA_SPEC":    8,
}

func (x TypeSpecProto_TypeSpecClass) String() string {
	return proto.EnumName(TypeSpecProto_TypeSpecClass_name, int32(x))
}
func (TypeSpecProto_TypeSpecClass) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor3, []int{8, 0}
}

// `StructuredValue` represents a dynamically typed value representing various
// data structures that are inspired by Python data structures typically used in
// TensorFlow functions as inputs and outputs.
//
// For example when saving a Layer there may be a `training` argument. If the
// user passes a boolean True/False, that switches between two concrete
// TensorFlow functions. In order to switch between them in the same way after
// loading the SavedModel, we need to represent "True" and "False".
//
// A more advanced example might be a function which takes a list of
// dictionaries mapping from strings to Tensors. In order to map from
// user-specified arguments `[{"a": tf.constant(1.)}, {"q": tf.constant(3.)}]`
// after load to the right saved TensorFlow function, we need to represent the
// nested structure and the strings, recording that we have a trace for anything
// matching `[{"a": tf.TensorSpec(None, tf.float32)}, {"q": tf.TensorSpec([],
// tf.float64)}]` as an example.
//
// Likewise functions may return nested structures of Tensors, for example
// returning a dictionary mapping from strings to Tensors. In order for the
// loaded function to return the same structure we need to serialize it.
//
// This is an ergonomic aid for working with loaded SavedModels, not a promise
// to serialize all possible function signatures. For example we do not expect
// to pickle generic Python objects, and ideally we'd stay language-agnostic.
type StructuredValue struct {
	// The kind of value.
	//
	// Types that are valid to be assigned to Kind:
	//	*StructuredValue_NoneValue
	//	*StructuredValue_Float64Value
	//	*StructuredValue_Int64Value
	//	*StructuredValue_StringValue
	//	*StructuredValue_BoolValue
	//	*StructuredValue_TensorShapeValue
	//	*StructuredValue_TensorDtypeValue
	//	*StructuredValue_TensorSpecValue
	//	*StructuredValue_TypeSpecValue
	//	*StructuredValue_ListValue
	//	*StructuredValue_TupleValue
	//	*StructuredValue_DictValue
	//	*StructuredValue_NamedTupleValue
	Kind isStructuredValue_Kind `protobuf_oneof:"kind"`
}

func (m *StructuredValue) Reset()                    { *m = StructuredValue{} }
func (m *StructuredValue) String() string            { return proto.CompactTextString(m) }
func (*StructuredValue) ProtoMessage()               {}
func (*StructuredValue) Descriptor() ([]byte, []int) { return fileDescriptor3, []int{0} }

type isStructuredValue_Kind interface {
	isStructuredValue_Kind()
}

type StructuredValue_NoneValue struct {
	NoneValue *NoneValue `protobuf:"bytes,1,opt,name=none_value,json=noneValue,oneof"`
}
type StructuredValue_Float64Value struct {
	Float64Value float64 `protobuf:"fixed64,11,opt,name=float64_value,json=float64Value,oneof"`
}
type StructuredValue_Int64Value struct {
	Int64Value int64 `protobuf:"zigzag64,12,opt,name=int64_value,json=int64Value,oneof"`
}
type StructuredValue_StringValue struct {
	StringValue string `protobuf:"bytes,13,opt,name=string_value,json=stringValue,oneof"`
}
type StructuredValue_BoolValue struct {
	BoolValue bool `protobuf:"varint,14,opt,name=bool_value,json=boolValue,oneof"`
}
type StructuredValue_TensorShapeValue struct {
	TensorShapeValue *tensorflow.TensorShapeProto `protobuf:"bytes,31,opt,name=tensor_shape_value,json=tensorShapeValue,oneof"`
}
type StructuredValue_TensorDtypeValue struct {
	TensorDtypeValue tensorflow1.DataType `protobuf:"varint,32,opt,name=tensor_dtype_value,json=tensorDtypeValue,enum=tensorflow.DataType,oneof"`
}
type StructuredValue_TensorSpecValue struct {
	TensorSpecValue *TensorSpecProto `protobuf:"bytes,33,opt,name=tensor_spec_value,json=tensorSpecValue,oneof"`
}
type StructuredValue_TypeSpecValue struct {
	TypeSpecValue *TypeSpecProto `protobuf:"bytes,34,opt,name=type_spec_value,json=typeSpecValue,oneof"`
}
type StructuredValue_ListValue struct {
	ListValue *ListValue `protobuf:"bytes,51,opt,name=list_value,json=listValue,oneof"`
}
type StructuredValue_TupleValue struct {
	TupleValue *TupleValue `protobuf:"bytes,52,opt,name=tuple_value,json=tupleValue,oneof"`
}
type StructuredValue_DictValue struct {
	DictValue *DictValue `protobuf:"bytes,53,opt,name=dict_value,json=dictValue,oneof"`
}
type StructuredValue_NamedTupleValue struct {
	NamedTupleValue *NamedTupleValue `protobuf:"bytes,54,opt,name=named_tuple_value,json=namedTupleValue,oneof"`
}

func (*StructuredValue_NoneValue) isStructuredValue_Kind()        {}
func (*StructuredValue_Float64Value) isStructuredValue_Kind()     {}
func (*StructuredValue_Int64Value) isStructuredValue_Kind()       {}
func (*StructuredValue_StringValue) isStructuredValue_Kind()      {}
func (*StructuredValue_BoolValue) isStructuredValue_Kind()        {}
func (*StructuredValue_TensorShapeValue) isStructuredValue_Kind() {}
func (*StructuredValue_TensorDtypeValue) isStructuredValue_Kind() {}
func (*StructuredValue_TensorSpecValue) isStructuredValue_Kind()  {}
func (*StructuredValue_TypeSpecValue) isStructuredValue_Kind()    {}
func (*StructuredValue_ListValue) isStructuredValue_Kind()        {}
func (*StructuredValue_TupleValue) isStructuredValue_Kind()       {}
func (*StructuredValue_DictValue) isStructuredValue_Kind()        {}
func (*StructuredValue_NamedTupleValue) isStructuredValue_Kind()  {}

func (m *StructuredValue) GetKind() isStructuredValue_Kind {
	if m != nil {
		return m.Kind
	}
	return nil
}

func (m *StructuredValue) GetNoneValue() *NoneValue {
	if x, ok := m.GetKind().(*StructuredValue_NoneValue); ok {
		return x.NoneValue
	}
	return nil
}

func (m *StructuredValue) GetFloat64Value() float64 {
	if x, ok := m.GetKind().(*StructuredValue_Float64Value); ok {
		return x.Float64Value
	}
	return 0
}

func (m *StructuredValue) GetInt64Value() int64 {
	if x, ok := m.GetKind().(*StructuredValue_Int64Value); ok {
		return x.Int64Value
	}
	return 0
}

func (m *StructuredValue) GetStringValue() string {
	if x, ok := m.GetKind().(*StructuredValue_StringValue); ok {
		return x.StringValue
	}
	return ""
}

func (m *StructuredValue) GetBoolValue() bool {
	if x, ok := m.GetKind().(*StructuredValue_BoolValue); ok {
		return x.BoolValue
	}
	return false
}

func (m *StructuredValue) GetTensorShapeValue() *tensorflow.TensorShapeProto {
	if x, ok := m.GetKind().(*StructuredValue_TensorShapeValue); ok {
		return x.TensorShapeValue
	}
	return nil
}

func (m *StructuredValue) GetTensorDtypeValue() tensorflow1.DataType {
	if x, ok := m.GetKind().(*StructuredValue_TensorDtypeValue); ok {
		return x.TensorDtypeValue
	}
	return tensorflow1.DataType_DT_INVALID
}

func (m *StructuredValue) GetTensorSpecValue() *TensorSpecProto {
	if x, ok := m.GetKind().(*StructuredValue_TensorSpecValue); ok {
		return x.TensorSpecValue
	}
	return nil
}

func (m *StructuredValue) GetTypeSpecValue() *TypeSpecProto {
	if x, ok := m.GetKind().(*StructuredValue_TypeSpecValue); ok {
		return x.TypeSpecValue
	}
	return nil
}

func (m *StructuredValue) GetListValue() *ListValue {
	if x, ok := m.GetKind().(*StructuredValue_ListValue); ok {
		return x.ListValue
	}
	return nil
}

func (m *StructuredValue) GetTupleValue() *TupleValue {
	if x, ok := m.GetKind().(*StructuredValue_TupleValue); ok {
		return x.TupleValue
	}
	return nil
}

func (m *StructuredValue) GetDictValue() *DictValue {
	if x, ok := m.GetKind().(*StructuredValue_DictValue); ok {
		return x.DictValue
	}
	return nil
}

func (m *StructuredValue) GetNamedTupleValue() *NamedTupleValue {
	if x, ok := m.GetKind().(*StructuredValue_NamedTupleValue); ok {
		return x.NamedTupleValue
	}
	return nil
}

// XXX_OneofFuncs is for the internal use of the proto package.
func (*StructuredValue) XXX_OneofFuncs() (func(msg proto.Message, b *proto.Buffer) error, func(msg proto.Message, tag, wire int, b *proto.Buffer) (bool, error), func(msg proto.Message) (n int), []interface{}) {
	return _StructuredValue_OneofMarshaler, _StructuredValue_OneofUnmarshaler, _StructuredValue_OneofSizer, []interface{}{
		(*StructuredValue_NoneValue)(nil),
		(*StructuredValue_Float64Value)(nil),
		(*StructuredValue_Int64Value)(nil),
		(*StructuredValue_StringValue)(nil),
		(*StructuredValue_BoolValue)(nil),
		(*StructuredValue_TensorShapeValue)(nil),
		(*StructuredValue_TensorDtypeValue)(nil),
		(*StructuredValue_TensorSpecValue)(nil),
		(*StructuredValue_TypeSpecValue)(nil),
		(*StructuredValue_ListValue)(nil),
		(*StructuredValue_TupleValue)(nil),
		(*StructuredValue_DictValue)(nil),
		(*StructuredValue_NamedTupleValue)(nil),
	}
}

func _StructuredValue_OneofMarshaler(msg proto.Message, b *proto.Buffer) error {
	m := msg.(*StructuredValue)
	// kind
	switch x := m.Kind.(type) {
	case *StructuredValue_NoneValue:
		b.EncodeVarint(1<<3 | proto.WireBytes)
		if err := b.EncodeMessage(x.NoneValue); err != nil {
			return err
		}
	case *StructuredValue_Float64Value:
		b.EncodeVarint(11<<3 | proto.WireFixed64)
		b.EncodeFixed64(math.Float64bits(x.Float64Value))
	case *StructuredValue_Int64Value:
		b.EncodeVarint(12<<3 | proto.WireVarint)
		b.EncodeZigzag64(uint64(x.Int64Value))
	case *StructuredValue_StringValue:
		b.EncodeVarint(13<<3 | proto.WireBytes)
		b.EncodeStringBytes(x.StringValue)
	case *StructuredValue_BoolValue:
		t := uint64(0)
		if x.BoolValue {
			t = 1
		}
		b.EncodeVarint(14<<3 | proto.WireVarint)
		b.EncodeVarint(t)
	case *StructuredValue_TensorShapeValue:
		b.EncodeVarint(31<<3 | proto.WireBytes)
		if err := b.EncodeMessage(x.TensorShapeValue); err != nil {
			return err
		}
	case *StructuredValue_TensorDtypeValue:
		b.EncodeVarint(32<<3 | proto.WireVarint)
		b.EncodeVarint(uint64(x.TensorDtypeValue))
	case *StructuredValue_TensorSpecValue:
		b.EncodeVarint(33<<3 | proto.WireBytes)
		if err := b.EncodeMessage(x.TensorSpecValue); err != nil {
			return err
		}
	case *StructuredValue_TypeSpecValue:
		b.EncodeVarint(34<<3 | proto.WireBytes)
		if err := b.EncodeMessage(x.TypeSpecValue); err != nil {
			return err
		}
	case *StructuredValue_ListValue:
		b.EncodeVarint(51<<3 | proto.WireBytes)
		if err := b.EncodeMessage(x.ListValue); err != nil {
			return err
		}
	case *StructuredValue_TupleValue:
		b.EncodeVarint(52<<3 | proto.WireBytes)
		if err := b.EncodeMessage(x.TupleValue); err != nil {
			return err
		}
	case *StructuredValue_DictValue:
		b.EncodeVarint(53<<3 | proto.WireBytes)
		if err := b.EncodeMessage(x.DictValue); err != nil {
			return err
		}
	case *StructuredValue_NamedTupleValue:
		b.EncodeVarint(54<<3 | proto.WireBytes)
		if err := b.EncodeMessage(x.NamedTupleValue); err != nil {
			return err
		}
	case nil:
	default:
		return fmt.Errorf("StructuredValue.Kind has unexpected type %T", x)
	}
	return nil
}

func _StructuredValue_OneofUnmarshaler(msg proto.Message, tag, wire int, b *proto.Buffer) (bool, error) {
	m := msg.(*StructuredValue)
	switch tag {
	case 1: // kind.none_value
		if wire != proto.WireBytes {
			return true, proto.ErrInternalBadWireType
		}
		msg := new(NoneValue)
		err := b.DecodeMessage(msg)
		m.Kind = &StructuredValue_NoneValue{msg}
		return true, err
	case 11: // kind.float64_value
		if wire != proto.WireFixed64 {
			return true, proto.ErrInternalBadWireType
		}
		x, err := b.DecodeFixed64()
		m.Kind = &StructuredValue_Float64Value{math.Float64frombits(x)}
		return true, err
	case 12: // kind.int64_value
		if wire != proto.WireVarint {
			return true, proto.ErrInternalBadWireType
		}
		x, err := b.DecodeZigzag64()
		m.Kind = &StructuredValue_Int64Value{int64(x)}
		return true, err
	case 13: // kind.string_value
		if wire != proto.WireBytes {
			return true, proto.ErrInternalBadWireType
		}
		x, err := b.DecodeStringBytes()
		m.Kind = &StructuredValue_StringValue{x}
		return true, err
	case 14: // kind.bool_value
		if wire != proto.WireVarint {
			return true, proto.ErrInternalBadWireType
		}
		x, err := b.DecodeVarint()
		m.Kind = &StructuredValue_BoolValue{x != 0}
		return true, err
	case 31: // kind.tensor_shape_value
		if wire != proto.WireBytes {
			return true, proto.ErrInternalBadWireType
		}
		msg := new(tensorflow.TensorShapeProto)
		err := b.DecodeMessage(msg)
		m.Kind = &StructuredValue_TensorShapeValue{msg}
		return true, err
	case 32: // kind.tensor_dtype_value
		if wire != proto.WireVarint {
			return true, proto.ErrInternalBadWireType
		}
		x, err := b.DecodeVarint()
		m.Kind = &StructuredValue_TensorDtypeValue{tensorflow1.DataType(x)}
		return true, err
	case 33: // kind.tensor_spec_value
		if wire != proto.WireBytes {
			return true, proto.ErrInternalBadWireType
		}
		msg := new(TensorSpecProto)
		err := b.DecodeMessage(msg)
		m.Kind = &StructuredValue_TensorSpecValue{msg}
		return true, err
	case 34: // kind.type_spec_value
		if wire != proto.WireBytes {
			return true, proto.ErrInternalBadWireType
		}
		msg := new(TypeSpecProto)
		err := b.DecodeMessage(msg)
		m.Kind = &StructuredValue_TypeSpecValue{msg}
		return true, err
	case 51: // kind.list_value
		if wire != proto.WireBytes {
			return true, proto.ErrInternalBadWireType
		}
		msg := new(ListValue)
		err := b.DecodeMessage(msg)
		m.Kind = &StructuredValue_ListValue{msg}
		return true, err
	case 52: // kind.tuple_value
		if wire != proto.WireBytes {
			return true, proto.ErrInternalBadWireType
		}
		msg := new(TupleValue)
		err := b.DecodeMessage(msg)
		m.Kind = &StructuredValue_TupleValue{msg}
		return true, err
	case 53: // kind.dict_value
		if wire != proto.WireBytes {
			return true, proto.ErrInternalBadWireType
		}
		msg := new(DictValue)
		err := b.DecodeMessage(msg)
		m.Kind = &StructuredValue_DictValue{msg}
		return true, err
	case 54: // kind.named_tuple_value
		if wire != proto.WireBytes {
			return true, proto.ErrInternalBadWireType
		}
		msg := new(NamedTupleValue)
		err := b.DecodeMessage(msg)
		m.Kind = &StructuredValue_NamedTupleValue{msg}
		return true, err
	default:
		return false, nil
	}
}

func _StructuredValue_OneofSizer(msg proto.Message) (n int) {
	m := msg.(*StructuredValue)
	// kind
	switch x := m.Kind.(type) {
	case *StructuredValue_NoneValue:
		s := proto.Size(x.NoneValue)
		n += proto.SizeVarint(1<<3 | proto.WireBytes)
		n += proto.SizeVarint(uint64(s))
		n += s
	case *StructuredValue_Float64Value:
		n += proto.SizeVarint(11<<3 | proto.WireFixed64)
		n += 8
	case *StructuredValue_Int64Value:
		n += proto.SizeVarint(12<<3 | proto.WireVarint)
		n += proto.SizeVarint(uint64(uint64(x.Int64Value<<1) ^ uint64((int64(x.Int64Value) >> 63))))
	case *StructuredValue_StringValue:
		n += proto.SizeVarint(13<<3 | proto.WireBytes)
		n += proto.SizeVarint(uint64(len(x.StringValue)))
		n += len(x.StringValue)
	case *StructuredValue_BoolValue:
		n += proto.SizeVarint(14<<3 | proto.WireVarint)
		n += 1
	case *StructuredValue_TensorShapeValue:
		s := proto.Size(x.TensorShapeValue)
		n += proto.SizeVarint(31<<3 | proto.WireBytes)
		n += proto.SizeVarint(uint64(s))
		n += s
	case *StructuredValue_TensorDtypeValue:
		n += proto.SizeVarint(32<<3 | proto.WireVarint)
		n += proto.SizeVarint(uint64(x.TensorDtypeValue))
	case *StructuredValue_TensorSpecValue:
		s := proto.Size(x.TensorSpecValue)
		n += proto.SizeVarint(33<<3 | proto.WireBytes)
		n += proto.SizeVarint(uint64(s))
		n += s
	case *StructuredValue_TypeSpecValue:
		s := proto.Size(x.TypeSpecValue)
		n += proto.SizeVarint(34<<3 | proto.WireBytes)
		n += proto.SizeVarint(uint64(s))
		n += s
	case *StructuredValue_ListValue:
		s := proto.Size(x.ListValue)
		n += proto.SizeVarint(51<<3 | proto.WireBytes)
		n += proto.SizeVarint(uint64(s))
		n += s
	case *StructuredValue_TupleValue:
		s := proto.Size(x.TupleValue)
		n += proto.SizeVarint(52<<3 | proto.WireBytes)
		n += proto.SizeVarint(uint64(s))
		n += s
	case *StructuredValue_DictValue:
		s := proto.Size(x.DictValue)
		n += proto.SizeVarint(53<<3 | proto.WireBytes)
		n += proto.SizeVarint(uint64(s))
		n += s
	case *StructuredValue_NamedTupleValue:
		s := proto.Size(x.NamedTupleValue)
		n += proto.SizeVarint(54<<3 | proto.WireBytes)
		n += proto.SizeVarint(uint64(s))
		n += s
	case nil:
	default:
		panic(fmt.Sprintf("proto: unexpected type %T in oneof", x))
	}
	return n
}

// Represents None.
type NoneValue struct {
}

func (m *NoneValue) Reset()                    { *m = NoneValue{} }
func (m *NoneValue) String() string            { return proto.CompactTextString(m) }
func (*NoneValue) ProtoMessage()               {}
func (*NoneValue) Descriptor() ([]byte, []int) { return fileDescriptor3, []int{1} }

// Represents a Python list.
type ListValue struct {
	Values []*StructuredValue `protobuf:"bytes,1,rep,name=values" json:"values,omitempty"`
}

func (m *ListValue) Reset()                    { *m = ListValue{} }
func (m *ListValue) String() string            { return proto.CompactTextString(m) }
func (*ListValue) ProtoMessage()               {}
func (*ListValue) Descriptor() ([]byte, []int) { return fileDescriptor3, []int{2} }

func (m *ListValue) GetValues() []*StructuredValue {
	if m != nil {
		return m.Values
	}
	return nil
}

// Represents a Python tuple.
type TupleValue struct {
	Values []*StructuredValue `protobuf:"bytes,1,rep,name=values" json:"values,omitempty"`
}

func (m *TupleValue) Reset()                    { *m = TupleValue{} }
func (m *TupleValue) String() string            { return proto.CompactTextString(m) }
func (*TupleValue) ProtoMessage()               {}
func (*TupleValue) Descriptor() ([]byte, []int) { return fileDescriptor3, []int{3} }

func (m *TupleValue) GetValues() []*StructuredValue {
	if m != nil {
		return m.Values
	}
	return nil
}

// Represents a Python dict keyed by `str`.
// The comment on Unicode from Value.string_value applies analogously.
type DictValue struct {
	Fields map[string]*StructuredValue `protobuf:"bytes,1,rep,name=fields" json:"fields,omitempty" protobuf_key:"bytes,1,opt,name=key" protobuf_val:"bytes,2,opt,name=value"`
}

func (m *DictValue) Reset()                    { *m = DictValue{} }
func (m *DictValue) String() string            { return proto.CompactTextString(m) }
func (*DictValue) ProtoMessage()               {}
func (*DictValue) Descriptor() ([]byte, []int) { return fileDescriptor3, []int{4} }

func (m *DictValue) GetFields() map[string]*StructuredValue {
	if m != nil {
		return m.Fields
	}
	return nil
}

// Represents a (key, value) pair.
type PairValue struct {
	Key   string           `protobuf:"bytes,1,opt,name=key" json:"key,omitempty"`
	Value *StructuredValue `protobuf:"bytes,2,opt,name=value" json:"value,omitempty"`
}

func (m *PairValue) Reset()                    { *m = PairValue{} }
func (m *PairValue) String() string            { return proto.CompactTextString(m) }
func (*PairValue) ProtoMessage()               {}
func (*PairValue) Descriptor() ([]byte, []int) { return fileDescriptor3, []int{5} }

func (m *PairValue) GetKey() string {
	if m != nil {
		return m.Key
	}
	return ""
}

func (m *PairValue) GetValue() *StructuredValue {
	if m != nil {
		return m.Value
	}
	return nil
}

// Represents Python's namedtuple.
type NamedTupleValue struct {
	Name   string       `protobuf:"bytes,1,opt,name=name" json:"name,omitempty"`
	Values []*PairValue `protobuf:"bytes,2,rep,name=values" json:"values,omitempty"`
}

func (m *NamedTupleValue) Reset()                    { *m = NamedTupleValue{} }
func (m *NamedTupleValue) String() string            { return proto.CompactTextString(m) }
func (*NamedTupleValue) ProtoMessage()               {}
func (*NamedTupleValue) Descriptor() ([]byte, []int) { return fileDescriptor3, []int{6} }

func (m *NamedTupleValue) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *NamedTupleValue) GetValues() []*PairValue {
	if m != nil {
		return m.Values
	}
	return nil
}

// A protobuf to tf.TensorSpec.
type TensorSpecProto struct {
	Name  string                       `protobuf:"bytes,1,opt,name=name" json:"name,omitempty"`
	Shape *tensorflow.TensorShapeProto `protobuf:"bytes,2,opt,name=shape" json:"shape,omitempty"`
	Dtype tensorflow1.DataType         `protobuf:"varint,3,opt,name=dtype,enum=tensorflow.DataType" json:"dtype,omitempty"`
}

func (m *TensorSpecProto) Reset()                    { *m = TensorSpecProto{} }
func (m *TensorSpecProto) String() string            { return proto.CompactTextString(m) }
func (*TensorSpecProto) ProtoMessage()               {}
func (*TensorSpecProto) Descriptor() ([]byte, []int) { return fileDescriptor3, []int{7} }

func (m *TensorSpecProto) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *TensorSpecProto) GetShape() *tensorflow.TensorShapeProto {
	if m != nil {
		return m.Shape
	}
	return nil
}

func (m *TensorSpecProto) GetDtype() tensorflow1.DataType {
	if m != nil {
		return m.Dtype
	}
	return tensorflow1.DataType_DT_INVALID
}

// Represents a tf.TypeSpec
type TypeSpecProto struct {
	TypeSpecClass TypeSpecProto_TypeSpecClass `protobuf:"varint,1,opt,name=type_spec_class,json=typeSpecClass,enum=tensorflow.TypeSpecProto_TypeSpecClass" json:"type_spec_class,omitempty"`
	// The value returned by TypeSpec._serialize().
	TypeState *StructuredValue `protobuf:"bytes,2,opt,name=type_state,json=typeState" json:"type_state,omitempty"`
}

func (m *TypeSpecProto) Reset()                    { *m = TypeSpecProto{} }
func (m *TypeSpecProto) String() string            { return proto.CompactTextString(m) }
func (*TypeSpecProto) ProtoMessage()               {}
func (*TypeSpecProto) Descriptor() ([]byte, []int) { return fileDescriptor3, []int{8} }

func (m *TypeSpecProto) GetTypeSpecClass() TypeSpecProto_TypeSpecClass {
	if m != nil {
		return m.TypeSpecClass
	}
	return TypeSpecProto_UNKNOWN
}

func (m *TypeSpecProto) GetTypeState() *StructuredValue {
	if m != nil {
		return m.TypeState
	}
	return nil
}

func init() {
	proto.RegisterType((*StructuredValue)(nil), "tensorflow.StructuredValue")
	proto.RegisterType((*NoneValue)(nil), "tensorflow.NoneValue")
	proto.RegisterType((*ListValue)(nil), "tensorflow.ListValue")
	proto.RegisterType((*TupleValue)(nil), "tensorflow.TupleValue")
	proto.RegisterType((*DictValue)(nil), "tensorflow.DictValue")
	proto.RegisterType((*PairValue)(nil), "tensorflow.PairValue")
	proto.RegisterType((*NamedTupleValue)(nil), "tensorflow.NamedTupleValue")
	proto.RegisterType((*TensorSpecProto)(nil), "tensorflow.TensorSpecProto")
	proto.RegisterType((*TypeSpecProto)(nil), "tensorflow.TypeSpecProto")
	proto.RegisterEnum("tensorflow.TypeSpecProto_TypeSpecClass", TypeSpecProto_TypeSpecClass_name, TypeSpecProto_TypeSpecClass_value)
}

func init() { proto.RegisterFile("tensorflow/core/protobuf/struct.proto", fileDescriptor3) }

var fileDescriptor3 = []byte{
	// 785 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xac, 0x55, 0xdd, 0x6e, 0xe3, 0x44,
	0x14, 0x8e, 0x93, 0x26, 0xbb, 0x3e, 0x6e, 0x9a, 0x74, 0xd8, 0x5d, 0x4a, 0x41, 0x5a, 0xd7, 0xa8,
	0x22, 0x42, 0x90, 0x88, 0x74, 0xa9, 0xd8, 0xbd, 0xc2, 0xc4, 0x66, 0x13, 0x11, 0x39, 0xd6, 0xd8,
	0x2c, 0x70, 0x65, 0xb9, 0xc9, 0x04, 0xac, 0x7a, 0x6d, 0xcb, 0x9e, 0xb0, 0xca, 0x03, 0xf0, 0x1a,
	0x3c, 0x15, 0x6f, 0xc2, 0x25, 0x37, 0x68, 0x66, 0xfc, 0x97, 0xa4, 0xad, 0x10, 0xda, 0x9b, 0x68,
	0xce, 0x37, 0xdf, 0xf9, 0xce, 0x99, 0x6f, 0x7c, 0x26, 0x70, 0x49, 0x49, 0x94, 0xc5, 0xe9, 0x3a,
	0x8c, 0xdf, 0x8d, 0x96, 0x71, 0x4a, 0x46, 0x49, 0x1a, 0xd3, 0xf8, 0x66, 0xb3, 0x1e, 0x65, 0x34,
	0xdd, 0x2c, 0xe9, 0x90, 0xc7, 0x08, 0x2a, 0xda, 0xf9, 0x17, 0xfb, 0x29, 0xeb, 0xd4, 0x7f, 0x4b,
	0xde, 0xc5, 0xe9, 0xed, 0x48, 0xec, 0x78, 0xd9, 0x6f, 0x7e, 0x42, 0x44, 0xe6, 0xf9, 0xe5, 0x03,
	0xec, 0x6d, 0x42, 0x32, 0x41, 0xd3, 0xfe, 0x69, 0x43, 0xcf, 0xe1, 0x15, 0x37, 0x29, 0x59, 0xbd,
	0xf1, 0xc3, 0x0d, 0x41, 0xd7, 0x00, 0x51, 0x1c, 0x11, 0xef, 0x77, 0x16, 0x9d, 0x49, 0xaa, 0x34,
	0x50, 0xc6, 0x4f, 0x87, 0x95, 0xde, 0xd0, 0x8a, 0x23, 0xc2, 0xa9, 0xd3, 0x06, 0x96, 0xa3, 0x22,
	0x40, 0x97, 0xd0, 0x5d, 0x87, 0xb1, 0x4f, 0xaf, 0x5f, 0xe4, 0xa9, 0x8a, 0x2a, 0x0d, 0xa4, 0x69,
	0x03, 0x1f, 0xe7, 0xb0, 0xa0, 0x5d, 0x80, 0x12, 0x44, 0x15, 0xe9, 0x58, 0x95, 0x06, 0x68, 0xda,
	0xc0, 0xc0, 0x41, 0x41, 0xf9, 0x14, 0x8e, 0x33, 0x9a, 0x06, 0xd1, 0xaf, 0x39, 0xa7, 0xab, 0x4a,
	0x03, 0x79, 0xda, 0xc0, 0x8a, 0x40, 0x05, 0xe9, 0x39, 0xc0, 0x4d, 0x1c, 0x87, 0x39, 0xe5, 0x44,
	0x95, 0x06, 0x8f, 0x59, 0x3f, 0x0c, 0x13, 0x84, 0x39, 0xa0, 0xba, 0x31, 0x39, 0xf1, 0x39, 0x3f,
	0xcf, 0x27, 0xf5, 0xf3, 0xb8, 0x7c, 0xe9, 0x30, 0x92, 0xcd, 0x5c, 0x99, 0x36, 0x70, 0x9f, 0x56,
	0x98, 0x50, 0x33, 0x4a, 0xb5, 0x15, 0x33, 0x30, 0x57, 0x53, 0x55, 0x69, 0x70, 0x32, 0x7e, 0x52,
	0x57, 0x33, 0x7c, 0xea, 0xbb, 0xdb, 0x84, 0x54, 0x2a, 0x06, 0x4b, 0x10, 0x2a, 0x33, 0x38, 0x2d,
	0x7a, 0x4a, 0xc8, 0x32, 0x17, 0xb9, 0xe0, 0x2d, 0x7d, 0x7c, 0x47, 0x4b, 0x09, 0x59, 0x16, 0x1d,
	0xf5, 0x68, 0x09, 0x09, 0xa9, 0x09, 0xf4, 0x78, 0x23, 0x35, 0x21, 0x8d, 0x0b, 0x7d, 0xb4, 0x23,
	0xb4, 0x4d, 0x48, 0x5d, 0xa6, 0x4b, 0x73, 0xa0, 0xbc, 0xeb, 0x30, 0xc8, 0x68, 0x9e, 0x7f, 0x75,
	0x78, 0xd7, 0xf3, 0x20, 0xa3, 0xe5, 0x5d, 0x87, 0x45, 0x80, 0x5e, 0x82, 0x42, 0x37, 0x49, 0x58,
	0xd8, 0xf0, 0x82, 0x27, 0x3e, 0xdb, 0x29, 0xcc, 0xb6, 0x8b, 0x4c, 0xa0, 0x65, 0xc4, 0x4a, 0xae,
	0x82, 0x65, 0x51, 0xf2, 0xeb, 0xc3, 0x92, 0x46, 0xb0, 0xac, 0x4a, 0xae, 0x8a, 0x80, 0x59, 0x17,
	0xf9, 0x6f, 0xc9, 0xca, 0xab, 0x17, 0xbe, 0x3e, 0xb4, 0xce, 0x62, 0xa4, 0x9d, 0xea, 0xbd, 0x68,
	0x17, 0xfa, 0xae, 0x03, 0x47, 0xb7, 0x41, 0xb4, 0xd2, 0x14, 0x90, 0xcb, 0x6f, 0x59, 0xfb, 0x16,
	0xe4, 0xf2, 0xb0, 0xe8, 0x0a, 0x3a, 0xbc, 0x40, 0x76, 0x26, 0xa9, 0xad, 0xfd, 0x0a, 0x7b, 0x03,
	0x83, 0x73, 0xaa, 0xa6, 0x03, 0x54, 0x45, 0xfe, 0x9f, 0xc4, 0x9f, 0x12, 0xc8, 0xe5, 0xf9, 0xd1,
	0x4b, 0xe8, 0xac, 0x03, 0x12, 0xae, 0x0a, 0x89, 0x8b, 0x3b, 0x6d, 0x1a, 0x7e, 0xcf, 0x39, 0x66,
	0x44, 0xd3, 0x2d, 0xce, 0x13, 0xce, 0xdf, 0x80, 0x52, 0x83, 0x51, 0x1f, 0x5a, 0xb7, 0x64, 0xcb,
	0x87, 0x59, 0xc6, 0x6c, 0x89, 0xbe, 0x82, 0xb6, 0xb0, 0xb0, 0x79, 0x68, 0xe1, 0x7e, 0x77, 0x82,
	0xf9, 0xaa, 0xf9, 0x8d, 0xa4, 0xd9, 0x20, 0xdb, 0x7e, 0x90, 0x8a, 0xfe, 0xde, 0x87, 0xaa, 0xe6,
	0x42, 0x6f, 0xef, 0xca, 0x10, 0x82, 0x23, 0x76, 0x65, 0xb9, 0x30, 0x5f, 0xa3, 0x2f, 0x4b, 0x3b,
	0x9b, 0xdc, 0x8b, 0x9d, 0x4f, 0xa6, 0x6c, 0xa9, 0x34, 0xf2, 0x0f, 0x09, 0x7a, 0x7b, 0x43, 0x74,
	0xa7, 0xec, 0x18, 0xda, 0xfc, 0x75, 0xc8, 0x1b, 0x7e, 0xf0, 0x5d, 0xc0, 0x82, 0x8a, 0x3e, 0x87,
	0x36, 0x7f, 0x03, 0xce, 0x5a, 0xf7, 0x4f, 0x3f, 0x16, 0x14, 0xed, 0xef, 0x26, 0x74, 0x77, 0x66,
	0x10, 0x2d, 0xea, 0x73, 0xbb, 0x0c, 0xfd, 0x2c, 0xe3, 0x0d, 0x9d, 0x8c, 0x3f, 0xbb, 0x77, 0x6e,
	0xcb, 0x68, 0xc2, 0xe8, 0xd5, 0x0c, 0xf3, 0x10, 0xbd, 0x02, 0x10, 0x82, 0xd4, 0xa7, 0xff, 0xc9,
	0x78, 0x99, 0xe7, 0x33, 0xb6, 0xf6, 0x97, 0x54, 0xb5, 0x27, 0xd4, 0x14, 0x78, 0xf4, 0xa3, 0xf5,
	0x83, 0xb5, 0xf8, 0xc9, 0xea, 0x37, 0xd0, 0x33, 0x40, 0x8e, 0xad, 0x63, 0xc7, 0xf4, 0x5c, 0xd3,
	0x72, 0x16, 0xd8, 0x73, 0x6c, 0x73, 0xd2, 0x97, 0xd0, 0x87, 0xf0, 0xc1, 0xcc, 0x32, 0xcc, 0x9f,
	0x4d, 0xc3, 0x73, 0xe6, 0xb3, 0x89, 0xe9, 0x88, 0x8d, 0x26, 0x4b, 0xc0, 0xfa, 0xeb, 0xd7, 0xa6,
	0xb1, 0x93, 0xd0, 0x42, 0x4f, 0xe1, 0x34, 0x07, 0x74, 0x8c, 0xf5, 0x5f, 0x04, 0x7c, 0xc4, 0x60,
	0x43, 0x77, 0x75, 0x8f, 0xfd, 0x38, 0xa6, 0x2b, 0xe0, 0x36, 0x53, 0xe1, 0xf0, 0xcc, 0x35, 0xb1,
	0xee, 0x16, 0x2a, 0x1d, 0x74, 0x0a, 0xdd, 0x85, 0xed, 0xce, 0x16, 0x96, 0x3e, 0x17, 0xd0, 0x23,
	0xf4, 0x04, 0xfa, 0xb6, 0x89, 0x3d, 0x6c, 0xda, 0xf3, 0xd9, 0x44, 0x17, 0xe8, 0xe3, 0x9b, 0x0e,
	0xff, 0x77, 0xbb, 0xfa, 0x37, 0x00, 0x00, 0xff, 0xff, 0xcd, 0xee, 0xa4, 0xdf, 0x67, 0x07, 0x00,
	0x00,
}
