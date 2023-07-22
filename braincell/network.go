package braincell

import (
	"fmt"
	"log"
	"math"
)

type Network struct {
	LayerCount     int
	Layout         []int
	Activation     []Mat
	Weights        []Mat
	Biases         []Mat
	ActivationFunc []func(float64) float64
}

type Layer struct {
	NeuronCount    int
	ActivationFunc func(x float64) float64
}

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func ReLU(x float64) float64 {
	return math.Max(0.0, x)
}

func GELU(x float64) float64 {
	return 0.5 * x * (1 + math.Erf(x/math.Sqrt(2)))
}

func (n Network) validate() {
	if len(n.Weights) != len(n.Biases) {
		log.Fatal("Error:")
	}
}

/*
Initialise a new neural network

layers indicates the amiunt of neurons in each layer
*/
func NetworkNew(layers []Layer, biasInitialiser func() float64, weightInitialiser func() float64) Network {
	layerCount := len(layers)
	newNetwork := Network{LayerCount: layerCount}
	newNetwork.Layout = make([]int, layerCount)
	newNetwork.ActivationFunc = make([]func(float64) float64, layerCount)
	newNetwork.Activation = make([]Mat, layerCount)
	newNetwork.Weights = make([]Mat, layerCount-1)
	newNetwork.Biases = make([]Mat, layerCount-1)
	//Initialise activation layers
	for layer := 0; layer < layerCount; layer++ {
		newNetwork.Layout[layer] = layers[layer].NeuronCount
		newNetwork.ActivationFunc[layer] = layers[layer].ActivationFunc
		newNetwork.Activation[layer] = MatNew(1, newNetwork.Layout[layer], Zero)
	}
	//Initialise weights and biases
	for layer := 0; layer < (layerCount - 1); layer++ {
		prevActivationLayer := &newNetwork.Activation[layer]
		nextActivationLayer := &newNetwork.Activation[layer+1]
		newNetwork.Weights[layer] = MatNew(prevActivationLayer.Cols, nextActivationLayer.Cols, weightInitialiser)
		newNetwork.Biases[layer] = MatNew(1, nextActivationLayer.Cols, biasInitialiser)
	}

	return newNetwork
}

func (m *Network) Forward(a0 Mat) Mat {
	//TODO:Assert size of a0/input layer

	m.Activation[0] = a0

	for layer := 0; layer < (len(m.Activation) - 1); layer++ {
		a := m.Activation[layer]
		w := m.Weights[layer]
		b := m.Biases[layer]

		m.Activation[layer+1] = MatDot(a, w)
		m.Activation[layer+1] = MatSum(m.Activation[layer+1], b)
		MatApply(&m.Activation[layer+1], GELU)
	}

	outputLayer := m.Activation[len(m.Activation)-1]
	return outputLayer
}

func (network *Network) Cost(trainIn Mat, trainOut Mat) float64 {
	if trainIn.Rows != trainOut.Rows {
		log.Fatal("Rows of training input is not equivalent to rows of training output")
	}

	var cost float64 = 0
	for sample := 0; sample < trainIn.Rows; sample++ {
		network.Forward(trainIn.Row(sample))
		var expected Mat = trainOut.Row(sample)
		outputLayer := network.Activation[network.LayerCount-1]
		var got Mat = outputLayer
		MatApply(&got, func(x float64) float64 { return (x * -1) })
		residualMat := MatSum(expected, got)
		var residual float64 = 0
		for entry := 0; entry < residualMat.Cols; entry++ {
			residual += residualMat.Data[0][entry]
		}
		cost += residual * residual
	}
	return cost
}

func (network *Network) Adjust(gradient *Network, rate float64) {
	for layer := 0; layer < (network.LayerCount - 1); layer++ {
		for i := 0; i < gradient.Weights[layer].Rows; i++ {
			for j := 0; j < gradient.Weights[layer].Cols; j++ {
				gradient.Weights[layer].Data[i][j] *= rate * -1
			}
		}
		for i := 0; i < gradient.Biases[layer].Rows; i++ {
			for j := 0; j < gradient.Biases[layer].Cols; j++ {
				gradient.Biases[layer].Data[i][j] *= rate * -1
			}
		}
		network.Weights[layer] = MatSum(network.Weights[layer], gradient.Weights[layer])
		network.Biases[layer] = MatSum(network.Biases[layer], gradient.Biases[layer])
	}
}

func (network *Network) Backprop(trainIn Mat, trainOut Mat, rate float64) {
	if trainIn.Rows != trainOut.Rows {
		log.Fatal("Error!: Invalid training data")
	}
	sampleCount := trainIn.Rows
	gradientLayout := make([]Layer, network.LayerCount)
	for layer := 0; layer < network.LayerCount; layer++ {
		gradientLayout[layer] = Layer{
			NeuronCount:    network.Layout[layer],
			ActivationFunc: network.ActivationFunc[layer],
		}
	}
	gradient := NetworkNew(gradientLayout, Zero, Zero)

	for layer := 0; layer < (gradient.LayerCount - 1); layer++ {
		MatFill(&gradient.Biases[layer], 0)
		MatFill(&gradient.Weights[layer], 0)
	}
	for sample := 0; sample < int(sampleCount); sample++ {
		for layer := 0; layer < (gradient.LayerCount); layer++ {
			MatFill(&network.Activation[layer], 0)
			MatFill(&gradient.Activation[layer], 0)
		}
		network.Forward(trainIn.Row(sample))
		for i := 0; i < trainOut.Cols; i++ {
			got := network.Activation[gradient.LayerCount-1].Data[0][i]
			expected := trainOut.Data[sample][i]
			gradient.Activation[gradient.LayerCount-1].Data[0][i] = got - expected
		}

		for layer := (network.LayerCount - 1); layer > 0; layer-- {
			for j := 0; j < network.Activation[layer].Cols; j++ {
				var a float64 = network.Activation[layer].Data[0][j]
				var da float64 = gradient.Activation[layer].Data[0][j]
				gradient.Biases[layer-1].Data[0][j] += 2 * da * a * (1 - a)
				for k := 0; k < network.Activation[layer-1].Cols; k++ {
					var pa float64 = network.Activation[layer-1].Data[0][k]
					var w float64 = network.Weights[layer-1].Data[k][j]
					gradient.Weights[layer-1].Data[k][j] += 2 * da * a * (1 - a) * pa
					gradient.Activation[layer-1].Data[0][k] += 2 * da * a * (1 - a) * w
				}
			}
		}
	}

	for i := 0; i < (gradient.LayerCount - 1); i++ {
		for j := 0; j < gradient.Weights[i].Rows; j++ {
			for k := 0; k < gradient.Weights[i].Cols; k++ {
				gradient.Weights[i].Data[j][k] /= float64(sampleCount)
			}
		}
		for j := 0; j < gradient.Biases[i].Rows; j++ {
			for k := 0; k < gradient.Biases[i].Cols; k++ {
				gradient.Biases[i].Data[j][k] /= float64(sampleCount)
			}
		}
	}
	network.Adjust(&gradient, rate)
}

func NetworkPrint(network *Network) {
	for layer := 0; layer < (network.LayerCount - 1); layer++ {
		MatPrint(network.Activation[layer], fmt.Sprintf("Activationlayer %d", (layer+1)))
		MatPrint(network.Weights[layer], fmt.Sprintf("Weightlayer %d", (layer+1)))
		MatPrint(network.Biases[layer], fmt.Sprintf("Biaslayer %d", (layer+1)))
	}
	MatPrint(network.Activation[len(network.Activation)-1], "Output layer")
}
