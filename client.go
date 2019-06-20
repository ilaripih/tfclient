package tfclient

import (
	"bytes"
	"image"
	"log"
	"reflect"
	"sync"

	tfcore "tensorflow/core/framework"
	meta_graph "tensorflow/core/protobuf"
	tf "tensorflow_serving/apis"

	proto "github.com/golang/protobuf/proto"

	"golang.org/x/image/bmp"
	"golang.org/x/image/draw"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

type TensorProto = tfcore.TensorProto
type ModelSpec = tf.ModelSpec

type PredictionClient struct {
	mu             sync.RWMutex
	rpcConn        *grpc.ClientConn
	svcConn        tf.PredictionServiceClient
	debug          bool
	inputConf      map[string]*InputConfig
	inputConfMutex sync.Mutex
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
	InputName     string
	Dtype         tfcore.DataType
	Height        int64
	Width         int64
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
		rpcConn:   conn,
		svcConn:   c,
		debug:     false,
		inputConf: make(map[string]*InputConfig),
	}, nil
}

func (c *PredictionClient) SetDebugging(debug bool) {
	c.debug = debug
}

func (c *PredictionClient) FormatBoxes(resp map[string]*tfcore.TensorProto, minConfidence float32) []BoxPrediction {
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

	return result
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
		ModelSpec:     modelSpec,
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

func (c *PredictionClient) FormatInputImages(images []image.Image, inputConf *InputConfig) (*tfcore.TensorProto, error) {
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
			Dtype:       inputConf.Dtype,
			StringVal:   [][]byte{content},
			TensorShape: tfshape,
		}
	} else {
		mustResize := true
		w := inputConf.Width
		h := inputConf.Height
		bounds := images[0].Bounds()
		if w <= 0 || h <= 0 {
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
		content := make([]byte, int64(len(images))*w*h*3)
		i := 0
		for _, img := range images {
			if mustResize {
				rect := image.Rect(0, 0, int(inputConf.Width), int(inputConf.Height))
				dst := image.NewRGBA(rect)
				draw.NearestNeighbor.Scale(dst, rect, img, bounds, draw.Over, nil)
				img = dst
			}

			var pix []uint8
			switch v := img.(type) {
			case *image.RGBA:
				pix = v.Pix
			case *image.NRGBA:
				pix = v.Pix
			}

			if pix == nil {
				// very slow fallback for non-RGBA images
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
			} else {
				nPix := len(pix)
				for j := 0; j < nPix; j += 4 {
					content[i] = pix[j]
					content[i+1] = pix[j+1]
					content[i+2] = pix[j+2]
					i += 3
				}
			}
		}
		inputProto = &tfcore.TensorProto{
			Dtype:         inputConf.Dtype,
			TensorContent: content,
			TensorShape:   tfshape,
		}
	}

	return inputProto, nil
}

func (c *PredictionClient) GetModelSpec(modelName, signatureName string, inputConf *InputConfig) *tf.ModelSpec {
	modelSpec := &tf.ModelSpec{
		Name: modelName,
	}
	if signatureName != "" {
		modelSpec.SignatureName = signatureName
	} else if inputConf.SignatureName != "" {
		modelSpec.SignatureName = inputConf.SignatureName
	}

	return modelSpec
}

func (c *PredictionClient) PredictRaw(modelSpec *tf.ModelSpec, inputConf *InputConfig, inputProto *tfcore.TensorProto) (map[string]*tfcore.TensorProto, error) {
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

	inputProto, err := c.FormatInputImages(images, inputConf)
	if err != nil {
		return nil, err
	}

	return c.PredictRaw(modelSpec, inputConf, inputProto)
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
