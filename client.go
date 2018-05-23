package tfclient

import (
	"errors"
	"log"
	"reflect"
	"sync"

	tfcore "tensorflow/core/framework"
	tf "tensorflow_serving/apis"

	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

type PredictionClient struct {
	mu      sync.RWMutex
	rpcConn *grpc.ClientConn
	svcConn tf.PredictionServiceClient
	debug   bool
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
	return &PredictionClient{rpcConn: conn, svcConn: c, debug: false}, nil
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

func (c *PredictionClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.svcConn = nil
	return c.rpcConn.Close()
}
