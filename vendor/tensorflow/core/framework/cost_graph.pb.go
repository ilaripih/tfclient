// Code generated by protoc-gen-go.
// source: tensorflow/core/framework/cost_graph.proto
// DO NOT EDIT!

package tensorflow

import proto "github.com/golang/protobuf/proto"
import fmt "fmt"
import math "math"

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

type CostGraphDef struct {
	Node []*CostGraphDef_Node `protobuf:"bytes,1,rep,name=node" json:"node,omitempty"`
}

func (m *CostGraphDef) Reset()                    { *m = CostGraphDef{} }
func (m *CostGraphDef) String() string            { return proto.CompactTextString(m) }
func (*CostGraphDef) ProtoMessage()               {}
func (*CostGraphDef) Descriptor() ([]byte, []int) { return fileDescriptor2, []int{0} }

func (m *CostGraphDef) GetNode() []*CostGraphDef_Node {
	if m != nil {
		return m.Node
	}
	return nil
}

type CostGraphDef_Node struct {
	// The name of the node. Names are globally unique.
	Name string `protobuf:"bytes,1,opt,name=name" json:"name,omitempty"`
	// The device of the node. Can be empty if the node is mapped to the
	// default partition or partitioning hasn't been run yet.
	Device string `protobuf:"bytes,2,opt,name=device" json:"device,omitempty"`
	// The id of the node. Node ids are only unique inside a partition.
	Id         int32                           `protobuf:"varint,3,opt,name=id" json:"id,omitempty"`
	InputInfo  []*CostGraphDef_Node_InputInfo  `protobuf:"bytes,4,rep,name=input_info,json=inputInfo" json:"input_info,omitempty"`
	OutputInfo []*CostGraphDef_Node_OutputInfo `protobuf:"bytes,5,rep,name=output_info,json=outputInfo" json:"output_info,omitempty"`
	// Temporary memory used by this node.
	TemporaryMemorySize  int64 `protobuf:"varint,6,opt,name=temporary_memory_size,json=temporaryMemorySize" json:"temporary_memory_size,omitempty"`
	HostPeakMemorySize   int64 `protobuf:"varint,10,opt,name=host_peak_memory_size,json=hostPeakMemorySize" json:"host_peak_memory_size,omitempty"`
	DevicePeakMemorySize int64 `protobuf:"varint,11,opt,name=device_peak_memory_size,json=devicePeakMemorySize" json:"device_peak_memory_size,omitempty"`
	PersistedMemorySize  int64 `protobuf:"varint,12,opt,name=persisted_memory_size,json=persistedMemorySize" json:"persisted_memory_size,omitempty"`
	AuxiliaryMemorySize  int64 `protobuf:"varint,13,opt,name=auxiliary_memory_size,json=auxiliaryMemorySize" json:"auxiliary_memory_size,omitempty"`
	// Estimate of the computational cost of this node, in microseconds.
	ComputeCost int64 `protobuf:"varint,9,opt,name=compute_cost,json=computeCost" json:"compute_cost,omitempty"`
	// If true, the output is permanent: it can't be discarded, because this
	// node is part of the "final output". Nodes may depend on final nodes.
	IsFinal bool `protobuf:"varint,7,opt,name=is_final,json=isFinal" json:"is_final,omitempty"`
	// Ids of the control inputs for this node.
	ControlInput []int32 `protobuf:"varint,8,rep,packed,name=control_input,json=controlInput" json:"control_input,omitempty"`
}

func (m *CostGraphDef_Node) Reset()                    { *m = CostGraphDef_Node{} }
func (m *CostGraphDef_Node) String() string            { return proto.CompactTextString(m) }
func (*CostGraphDef_Node) ProtoMessage()               {}
func (*CostGraphDef_Node) Descriptor() ([]byte, []int) { return fileDescriptor2, []int{0, 0} }

func (m *CostGraphDef_Node) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *CostGraphDef_Node) GetDevice() string {
	if m != nil {
		return m.Device
	}
	return ""
}

func (m *CostGraphDef_Node) GetId() int32 {
	if m != nil {
		return m.Id
	}
	return 0
}

func (m *CostGraphDef_Node) GetInputInfo() []*CostGraphDef_Node_InputInfo {
	if m != nil {
		return m.InputInfo
	}
	return nil
}

func (m *CostGraphDef_Node) GetOutputInfo() []*CostGraphDef_Node_OutputInfo {
	if m != nil {
		return m.OutputInfo
	}
	return nil
}

func (m *CostGraphDef_Node) GetTemporaryMemorySize() int64 {
	if m != nil {
		return m.TemporaryMemorySize
	}
	return 0
}

func (m *CostGraphDef_Node) GetHostPeakMemorySize() int64 {
	if m != nil {
		return m.HostPeakMemorySize
	}
	return 0
}

func (m *CostGraphDef_Node) GetDevicePeakMemorySize() int64 {
	if m != nil {
		return m.DevicePeakMemorySize
	}
	return 0
}

func (m *CostGraphDef_Node) GetPersistedMemorySize() int64 {
	if m != nil {
		return m.PersistedMemorySize
	}
	return 0
}

func (m *CostGraphDef_Node) GetAuxiliaryMemorySize() int64 {
	if m != nil {
		return m.AuxiliaryMemorySize
	}
	return 0
}

func (m *CostGraphDef_Node) GetComputeCost() int64 {
	if m != nil {
		return m.ComputeCost
	}
	return 0
}

func (m *CostGraphDef_Node) GetIsFinal() bool {
	if m != nil {
		return m.IsFinal
	}
	return false
}

func (m *CostGraphDef_Node) GetControlInput() []int32 {
	if m != nil {
		return m.ControlInput
	}
	return nil
}

// Inputs of this node. They must be executed before this node can be
// executed. An input is a particular output of another node, specified
// by the node id and the output index.
type CostGraphDef_Node_InputInfo struct {
	PrecedingNode int32 `protobuf:"varint,1,opt,name=preceding_node,json=precedingNode" json:"preceding_node,omitempty"`
	PrecedingPort int32 `protobuf:"varint,2,opt,name=preceding_port,json=precedingPort" json:"preceding_port,omitempty"`
}

func (m *CostGraphDef_Node_InputInfo) Reset()         { *m = CostGraphDef_Node_InputInfo{} }
func (m *CostGraphDef_Node_InputInfo) String() string { return proto.CompactTextString(m) }
func (*CostGraphDef_Node_InputInfo) ProtoMessage()    {}
func (*CostGraphDef_Node_InputInfo) Descriptor() ([]byte, []int) {
	return fileDescriptor2, []int{0, 0, 0}
}

func (m *CostGraphDef_Node_InputInfo) GetPrecedingNode() int32 {
	if m != nil {
		return m.PrecedingNode
	}
	return 0
}

func (m *CostGraphDef_Node_InputInfo) GetPrecedingPort() int32 {
	if m != nil {
		return m.PrecedingPort
	}
	return 0
}

// Outputs of this node.
type CostGraphDef_Node_OutputInfo struct {
	Size int64 `protobuf:"varint,1,opt,name=size" json:"size,omitempty"`
	// If >= 0, the output is an alias of an input. Note that an alias input
	// may itself be an alias. The algorithm will therefore need to follow
	// those pointers.
	AliasInputPort int64             `protobuf:"varint,2,opt,name=alias_input_port,json=aliasInputPort" json:"alias_input_port,omitempty"`
	Shape          *TensorShapeProto `protobuf:"bytes,3,opt,name=shape" json:"shape,omitempty"`
	Dtype          DataType          `protobuf:"varint,4,opt,name=dtype,enum=tensorflow.DataType" json:"dtype,omitempty"`
}

func (m *CostGraphDef_Node_OutputInfo) Reset()         { *m = CostGraphDef_Node_OutputInfo{} }
func (m *CostGraphDef_Node_OutputInfo) String() string { return proto.CompactTextString(m) }
func (*CostGraphDef_Node_OutputInfo) ProtoMessage()    {}
func (*CostGraphDef_Node_OutputInfo) Descriptor() ([]byte, []int) {
	return fileDescriptor2, []int{0, 0, 1}
}

func (m *CostGraphDef_Node_OutputInfo) GetSize() int64 {
	if m != nil {
		return m.Size
	}
	return 0
}

func (m *CostGraphDef_Node_OutputInfo) GetAliasInputPort() int64 {
	if m != nil {
		return m.AliasInputPort
	}
	return 0
}

func (m *CostGraphDef_Node_OutputInfo) GetShape() *TensorShapeProto {
	if m != nil {
		return m.Shape
	}
	return nil
}

func (m *CostGraphDef_Node_OutputInfo) GetDtype() DataType {
	if m != nil {
		return m.Dtype
	}
	return DataType_DT_INVALID
}

func init() {
	proto.RegisterType((*CostGraphDef)(nil), "tensorflow.CostGraphDef")
	proto.RegisterType((*CostGraphDef_Node)(nil), "tensorflow.CostGraphDef.Node")
	proto.RegisterType((*CostGraphDef_Node_InputInfo)(nil), "tensorflow.CostGraphDef.Node.InputInfo")
	proto.RegisterType((*CostGraphDef_Node_OutputInfo)(nil), "tensorflow.CostGraphDef.Node.OutputInfo")
}

func init() { proto.RegisterFile("tensorflow/core/framework/cost_graph.proto", fileDescriptor2) }

var fileDescriptor2 = []byte{
	// 556 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x09, 0x6e, 0x88, 0x02, 0xff, 0x7c, 0x53, 0xdb, 0x8e, 0xd3, 0x3c,
	0x10, 0x56, 0x7a, 0xee, 0xf4, 0xf0, 0xff, 0x32, 0x5b, 0x08, 0x15, 0x48, 0x01, 0xb4, 0x22, 0x5a,
	0xa1, 0x96, 0x0d, 0xe2, 0x05, 0x96, 0xd5, 0xa2, 0x5e, 0x00, 0x55, 0x76, 0x6f, 0xb8, 0x8a, 0x4c,
	0xe2, 0xb4, 0x56, 0x9b, 0xd8, 0xb2, 0x5d, 0x96, 0xee, 0x23, 0xf0, 0x28, 0x3c, 0x08, 0xcf, 0xc4,
	0x25, 0xf2, 0xa4, 0xa4, 0xe9, 0x22, 0x7a, 0x37, 0x99, 0xef, 0x60, 0x3b, 0xf3, 0x0d, 0x9c, 0x19,
	0x96, 0x6b, 0xa1, 0xd2, 0xb5, 0xb8, 0x9d, 0xc6, 0x42, 0xb1, 0x69, 0xaa, 0x68, 0xc6, 0x6e, 0x85,
	0x5a, 0x4d, 0x63, 0xa1, 0x4d, 0xb4, 0x50, 0x54, 0x2e, 0x27, 0x52, 0x09, 0x23, 0x08, 0xec, 0xb9,
	0xe3, 0x57, 0xff, 0xd6, 0x15, 0x48, 0xa4, 0x97, 0x54, 0xb2, 0x42, 0x39, 0x3e, 0x3d, 0xc2, 0xde,
	0x4a, 0xa6, 0x0b, 0xda, 0xf3, 0xef, 0x6d, 0xe8, 0xbf, 0x13, 0xda, 0xbc, 0xb7, 0x87, 0x5e, 0xb2,
	0x94, 0x9c, 0x43, 0x23, 0x17, 0x09, 0x73, 0x1d, 0xaf, 0xee, 0xf7, 0x82, 0xa7, 0x93, 0xbd, 0xcd,
	0xa4, 0xca, 0x9b, 0x7c, 0x14, 0x09, 0x0b, 0x91, 0x3a, 0xfe, 0xd9, 0x82, 0x86, 0xfd, 0x24, 0x04,
	0x1a, 0x39, 0xcd, 0xac, 0xd6, 0xf1, 0xbb, 0x21, 0xd6, 0xe4, 0x21, 0xb4, 0x12, 0xf6, 0x95, 0xc7,
	0xcc, 0xad, 0x61, 0x77, 0xf7, 0x45, 0x86, 0x50, 0xe3, 0x89, 0x5b, 0xf7, 0x1c, 0xbf, 0x19, 0xd6,
	0x78, 0x42, 0xae, 0x00, 0x78, 0x2e, 0x37, 0x26, 0xe2, 0x79, 0x2a, 0xdc, 0x06, 0x9e, 0xfe, 0xf2,
	0xe8, 0xe9, 0x93, 0x99, 0xe5, 0xcf, 0xf2, 0x54, 0x84, 0x5d, 0xfe, 0xa7, 0x24, 0x33, 0xe8, 0x89,
	0x8d, 0x29, 0x8d, 0x9a, 0x68, 0xe4, 0x1f, 0x37, 0xfa, 0x84, 0x02, 0x74, 0x02, 0x51, 0xd6, 0x24,
	0x80, 0x91, 0x61, 0x99, 0x14, 0x8a, 0xaa, 0x6d, 0x94, 0xb1, 0x4c, 0xa8, 0x6d, 0xa4, 0xf9, 0x1d,
	0x73, 0x5b, 0x9e, 0xe3, 0xd7, 0xc3, 0x07, 0x25, 0xf8, 0x01, 0xb1, 0x6b, 0x7e, 0xc7, 0xc8, 0x39,
	0x8c, 0x96, 0x76, 0x88, 0x92, 0xd1, 0xd5, 0x81, 0x06, 0x50, 0x43, 0x2c, 0x38, 0x67, 0x74, 0x55,
	0x91, 0xbc, 0x85, 0x47, 0xc5, 0x3f, 0xf9, 0x5b, 0xd4, 0x43, 0xd1, 0x49, 0x01, 0xdf, 0x93, 0x05,
	0x30, 0x92, 0x4c, 0x69, 0xae, 0x0d, 0x4b, 0x0e, 0x44, 0xfd, 0xe2, 0x76, 0x25, 0x78, 0xa8, 0xa1,
	0x9b, 0x6f, 0x7c, 0xcd, 0xef, 0xbf, 0x68, 0x50, 0x68, 0x4a, 0xb0, 0xa2, 0x79, 0x06, 0xfd, 0x58,
	0x64, 0x72, 0x63, 0x58, 0x64, 0xe3, 0xe9, 0x76, 0x91, 0xda, 0xdb, 0xf5, 0xec, 0xcf, 0x24, 0x8f,
	0xa1, 0xc3, 0x75, 0x94, 0xf2, 0x9c, 0xae, 0xdd, 0xb6, 0xe7, 0xf8, 0x9d, 0xb0, 0xcd, 0xf5, 0x95,
	0xfd, 0x24, 0x2f, 0x60, 0x10, 0x8b, 0xdc, 0x28, 0xb1, 0x8e, 0x70, 0x46, 0x6e, 0xc7, 0xab, 0xfb,
	0xcd, 0xb0, 0xbf, 0x6b, 0xe2, 0x08, 0xc7, 0x9f, 0xa1, 0x5b, 0xce, 0x92, 0x9c, 0xc2, 0x50, 0x2a,
	0x16, 0xb3, 0x84, 0xe7, 0x8b, 0x68, 0x17, 0x45, 0x1b, 0x92, 0x41, 0xd9, 0xc5, 0xac, 0x1d, 0xd0,
	0xa4, 0x50, 0x06, 0xf3, 0x55, 0xa5, 0xcd, 0x85, 0x32, 0xe3, 0x1f, 0x0e, 0xc0, 0x7e, 0xbc, 0x36,
	0xa1, 0xf8, 0x5e, 0x07, 0x1f, 0x81, 0x35, 0xf1, 0xe1, 0x7f, 0xba, 0xe6, 0x54, 0x17, 0x17, 0xdc,
	0x7b, 0xd5, 0xc3, 0x21, 0xf6, 0xf1, 0x6a, 0xd6, 0x8c, 0x04, 0xd0, 0xc4, 0x15, 0xc3, 0xd8, 0xf6,
	0x82, 0x27, 0xd5, 0x54, 0xdd, 0x60, 0x79, 0x6d, 0xe1, 0xb9, 0xdd, 0xac, 0xb0, 0xa0, 0x92, 0x33,
	0x68, 0x26, 0x76, 0xe1, 0xdc, 0x86, 0xe7, 0xf8, 0xc3, 0xe0, 0xa4, 0xaa, 0xb9, 0xa4, 0x86, 0xde,
	0x6c, 0x25, 0x0b, 0x0b, 0xca, 0xc5, 0x6b, 0x70, 0x85, 0x5a, 0x54, 0x19, 0xe5, 0xd2, 0x5e, 0xfc,
	0x57, 0xc6, 0x16, 0xed, 0xf5, 0xdc, 0xf9, 0xe5, 0x38, 0x5f, 0x5a, 0xb8, 0xc5, 0x6f, 0x7e, 0x07,
	0x00, 0x00, 0xff, 0xff, 0x7d, 0x8e, 0x66, 0x3c, 0x54, 0x04, 0x00, 0x00,
}
