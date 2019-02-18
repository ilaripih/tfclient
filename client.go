package tfclient

import (
	"bytes"
	"errors"
	"image"
	"log"
	"reflect"
	"sync"

	proto "github.com/golang/protobuf/proto"
	meta_graph "tensorflow/core/protobuf"
	tfcore "tensorflow/core/framework"
	tf "tensorflow_serving/apis"

	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"github.com/nfnt/resize"
	"golang.org/x/image/bmp"
)

type PredictionClient struct {
	mu      sync.RWMutex
	rpcConn *grpc.ClientConn
	svcConn tf.PredictionServiceClient
	debug   bool
	inputConf map[string]*InputConfig
	inputConfMutex sync.Mutex
}

type Prediction struct {
	Class string  `json:"class"`
	Score float32 `json:"score"`
}

type BoxPrediction struct {
	Class string  `json:"class"`
	Score float32 `json:"score"`
	Y1    float32 `json:"y1"`
	X1    float32 `json:"x1"`
	Y2    float32 `json:"y2"`
	X2    float32 `json:"x2"`
}

type InputConfig struct {
	SignatureName string
	InputName string
	Dtype tfcore.DataType
	Height int64
	Width int64
}

var classLabels = []string{
	"signs",
	"panels",
	"vehicles",
	"people",
	"license-plates",
	"traffic_light",
}

func NewClient(addr string) (*PredictionClient, error) {
	conn, err := grpc.Dial(addr, grpc.WithInsecure())
	if err != nil {
		return nil, err
	}
	c := tf.NewPredictionServiceClient(conn)
	return &PredictionClient{
		rpcConn: conn,
		svcConn: c,
		debug: false,
		inputConf: make(map[string]*InputConfig),
	}, nil
}

func (c *PredictionClient) SetDebugging(debug bool) {
	c.debug = debug
}

func (c *PredictionClient) Predict(modelName, inputsName, signatureName string, imgdata []byte) ([]Prediction, error) {
	resp, err := c.PredictRaw(modelName, inputsName, signatureName, imgdata)
	if err != nil {
		return nil, err
	}

	classesTensor, scoresTensor := resp["classes"], resp["scores"]
	if classesTensor == nil || scoresTensor == nil {
		return nil, errors.New("missing expected tensors in response")
	}

	classes := classesTensor.StringVal
	scores := scoresTensor.FloatVal
	var result []Prediction
	for i := 0; i < len(classes) && i < len(scores); i++ {
		result = append(result, Prediction{Class: string(classes[i]), Score: scores[i]})
	}
	return result, nil
}

func (c *PredictionClient) PredictBoxes(modelName, inputsName, signatureName string, imgdata []byte, minConfidence float32) ([]BoxPrediction, error) {
	resp, err := c.PredictRaw(modelName, inputsName, signatureName, imgdata)
	if err != nil {
		return nil, err
	}

	var result []BoxPrediction
	boxes := resp["detection_boxes"].FloatVal
	classes := resp["detection_classes"].FloatVal
	for i, score := range resp["detection_scores"].FloatVal {
		if score < minConfidence {
			break
		}
		classIndex := int(classes[i]) - 1

		coordIndex := i * 4
		p := BoxPrediction{
			Class: classLabels[classIndex],
			Score: score,
			Y1:    boxes[coordIndex],
			X1:    boxes[coordIndex+1],
			Y2:    boxes[coordIndex+2],
			X2:    boxes[coordIndex+3],
		}
		result = append(result, p)
	}

	return result, nil
}

func (c *PredictionClient) PredictRaw(modelName, inputsName, signatureName string, imgdata []byte) (map[string]*tfcore.TensorProto, error) {
	modelSpec := &tf.ModelSpec{
		Name: modelName,
	}
	if signatureName != "" {
		modelSpec.SignatureName = signatureName
	}

	resp, err := c.svcConn.Predict(context.Background(), &tf.PredictRequest{
		ModelSpec: modelSpec,
		Inputs: map[string]*tfcore.TensorProto{
			inputsName: &tfcore.TensorProto{
				Dtype:     tfcore.DataType_DT_STRING,
				StringVal: [][]byte{imgdata},
				TensorShape: &tfcore.TensorShapeProto{
					Dim: []*tfcore.TensorShapeProto_Dim{{Size: 1}},
				},
			},
		},
	})
	if err != nil {
		return nil, err
	}

	if c.debug {
		log.Println("Output format:", reflect.TypeOf(resp.Outputs))
		log.Println("Output:", resp.Outputs)
	}

	return resp.Outputs, nil
}

func (c *PredictionClient) GetInputConfig(modelName string) (*InputConfig, error) {
	c.inputConfMutex.Lock()
	conf, ok := c.inputConf[modelName]
	c.inputConfMutex.Unlock()
	if ok {
		return conf, nil
	}

	modelSpec := &tf.ModelSpec{
		Name: modelName,
	}
	resp, err := c.svcConn.GetModelMetadata(context.Background(), &tf.GetModelMetadataRequest{
		ModelSpec: modelSpec,
		MetadataField: []string{"signature_def"},
	})
	if err != nil {
		return nil, err
	}

	sgDefMap := tf.SignatureDefMap{}
	if err := proto.Unmarshal(resp.GetMetadata()["signature_def"].Value, &sgDefMap); err != nil {
		return nil, err
	}
	sgDef := sgDefMap.GetSignatureDef()
	var ret InputConfig
	var inputs map[string]*meta_graph.TensorInfo
	if _, ok := sgDef["predict_images"]; ok {
		ret.SignatureName = "predict_images"
		inputs = sgDef["predict_images"].Inputs
	} else {
		inputs = sgDef["serving_default"].Inputs
	}
	for key, val := range inputs {
		ret.InputName = key
		ret.Dtype = val.Dtype
		if val.Dtype != tfcore.DataType_DT_STRING {
			ret.Height = val.TensorShape.Dim[1].Size
			ret.Width = val.TensorShape.Dim[2].Size
		}
		break
	}
	c.inputConfMutex.Lock()
	c.inputConf[modelName] = &ret
	c.inputConfMutex.Unlock()

	return &ret, nil
}

func (c *PredictionClient) PredictImages(modelName, signatureName string, images []image.Image) (map[string]*tfcore.TensorProto, error) {
	inputConf, err := c.GetInputConfig(modelName)
	if err != nil {
		return nil, err
	}
	modelSpec := &tf.ModelSpec{
		Name: modelName,
	}
	if signatureName != "" {
		modelSpec.SignatureName = signatureName
	} else if inputConf.SignatureName != "" {
		modelSpec.SignatureName = inputConf.SignatureName
	}

	var inputProto *tfcore.TensorProto
	var tfshape *tfcore.TensorShapeProto
	if inputConf.Dtype == tfcore.DataType_DT_STRING {
		tfshape = &tfcore.TensorShapeProto{
			Dim: []*tfcore.TensorShapeProto_Dim{{Size: int64(len(images))}},
		}
		buf := new(bytes.Buffer)
		for _, img := range images {
			if err := bmp.Encode(buf, img); err != nil {
				return nil, err
			}
		}
		content := buf.Bytes()
		inputProto = &tfcore.TensorProto{
			Dtype: inputConf.Dtype,
			StringVal: [][]byte{content},
			TensorShape: tfshape,
		}
	} else {
		mustResize := true
		w := inputConf.Width
		h := inputConf.Height
		if w <= 0 || h <= 0 {
			bounds := images[0].Bounds()
			w = int64(bounds.Max.X)
			h = int64(bounds.Max.Y)
			mustResize = false
		}
		tfshape = &tfcore.TensorShapeProto{
			Dim: []*tfcore.TensorShapeProto_Dim{
				{Size: int64(len(images))},
				{Size: h},
				{Size: w},
				{Size: 3},
			},
		}
		content := make([]byte, int64(len(images)) * w * h * 3)
		i := 0
		for _, img := range images {
			if mustResize {
				img = resize.Resize(uint(inputConf.Width), uint(inputConf.Height), img, resize.NearestNeighbor)
			}
			for y := int64(0); y < h; y++ {
				for x := int64(0); x < w; x++ {
					c := img.At(int(x), int(y))
					r, g, b, _ := c.RGBA()
					content[i] = byte(r)
					i++
					content[i] = byte(g)
					i++
					content[i] = byte(b)
					i++
				}
			}
		}
		inputProto = &tfcore.TensorProto{
			Dtype: inputConf.Dtype,
			TensorContent: content,
			TensorShape: tfshape,
		}
	}
	resp, err := c.svcConn.Predict(context.Background(), &tf.PredictRequest{
		ModelSpec: modelSpec,
		Inputs: map[string]*tfcore.TensorProto{
			inputConf.InputName: inputProto,
		},
	})
	if err != nil {
		return nil, err
	}

	if c.debug {
		log.Println("Output format:", reflect.TypeOf(resp.Outputs))
		log.Println("Output:", resp.Outputs)
	}

	return resp.Outputs, nil
}

func (c *PredictionClient) GetOutput(modelName, signatureName string) (map[string]*tfcore.TensorProto, error) {
	modelSpec := &tf.ModelSpec{
		Name: modelName,
	}
	if signatureName != "" {
		modelSpec.SignatureName = signatureName
	}

	resp, err := c.svcConn.Predict(context.Background(), &tf.PredictRequest{
		ModelSpec: modelSpec,
		Inputs:    nil,
	})
	if err != nil {
		return nil, err
	}

	if c.debug {
		log.Println("Output format:", reflect.TypeOf(resp.Outputs))
		log.Println("Output:", resp.Outputs)
	}

	return resp.Outputs, nil
}

func (c *PredictionClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.svcConn = nil
	return c.rpcConn.Close()
}
