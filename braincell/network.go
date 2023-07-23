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
	ActivationFunc []Act
}

type Layer struct {
	NeuronCount    int
	ActivationFunc Act
}

type Act int

const (
	Sigmoid Act = 0
	ReLU        = 1
	GELU        = 2
	TanH        = 3
)
const RELU_MIN = 0.01

func activateFn(act Act) func(x float64) float64 {
	var fn func(x float64) float64 = nil
	switch act {
	case Sigmoid:
		fn = func(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }
		break
	case ReLU:
		fn = func(x float64) float64 {
			if x > 0 {
				return x
			} else {
				return x * RELU_MIN
			}
		}
		break
	case GELU:
		fn = func(x float64) float64 { return 0.5 * x * (1 + math.Erf(x/math.Sqrt(2))) }
		break
	case TanH:
		fn = func(x float64) float64 { return ((math.Exp(x)-math.Exp(-x))/(math.Exp(x)+math.Exp(-x)) + 1) / 2 }
		break
	default:
		log.Fatal("Error!: Invalid activation function")
	}
	return fn
}

func dActivateFn(act Act) func(x float64) float64 {
	var fn func(x float64) float64 = nil
	switch act {
	case Sigmoid:
		fn = func(x float64) float64 { return x * (1 - x) }
		break
	case ReLU:
		fn = func(x float64) float64 {
			if x >= 0 {
				return 1
			} else {
				return RELU_MIN
			}
		}
		break
	case GELU:
		fn = func(x float64) float64 {
			if x >= 0 {
				return 1
			} else {
				return RELU_MIN
			}
		}
		break
	case TanH:
		fn = func(x float64) float64 { return 1 - x*x }
		break
	default:
		log.Fatal("Error!: Invalid activation function")
	}
	return fn
}

func dTanH(x float64) float64 {
	return 1 - x*x
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
	newNetwork.ActivationFunc = make([]Act, layerCount)
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

func (network *Network) Forward(a0 Mat) Mat {
	//TODO:Assert size of a0/input layer

	network.Activation[0] = a0

	for layer := 0; layer < (len(network.Activation) - 1); layer++ {
		a := network.Activation[layer]
		w := network.Weights[layer]
		b := network.Biases[layer]

		network.Activation[layer+1] = MatDot(a, w)
		network.Activation[layer+1] = MatSum(network.Activation[layer+1], b)
		MatApply(&network.Activation[layer+1], activateFn(network.ActivationFunc[layer+1]))
	}

	outputLayer := network.Activation[len(network.Activation)-1]
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
				network.Weights[layer].Data[i][j] -= gradient.Weights[layer].Data[i][j] * rate
			}
		}
		for i := 0; i < gradient.Biases[layer].Rows; i++ {
			for j := 0; j < gradient.Biases[layer].Cols; j++ {
				network.Biases[layer].Data[i][j] -= gradient.Biases[layer].Data[i][j] * rate
			}
		}
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
				var qa float64 = dActivateFn(network.ActivationFunc[layer])(a)
				//float qa = dactf(a, NN_ACT);
				gradient.Biases[layer-1].Data[0][j] += 2 * da * qa
				for k := 0; k < network.Activation[layer-1].Cols; k++ {
					var pa float64 = network.Activation[layer-1].Data[0][k]
					var w float64 = network.Weights[layer-1].Data[k][j]
					gradient.Weights[layer-1].Data[k][j] += 2 * da * qa * pa
					gradient.Activation[layer-1].Data[0][k] += 2 * da * qa * w
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
		MatPrint(&network.Activation[layer], fmt.Sprintf("Activationlayer %d", (layer+1)))
		MatPrint(&network.Weights[layer], fmt.Sprintf("Weightlayer %d", (layer+1)))
		MatPrint(&network.Biases[layer], fmt.Sprintf("Biaslayer %d", (layer+1)))
	}
	MatPrint(&network.Activation[len(network.Activation)-1], "Output layer")
}
