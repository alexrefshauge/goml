package braincell

import (
	"log"
	"math"
	"math/rand"
)

type Network struct {
	LayerCount     int
	Layout         []int
	Activation     []Mat
	Weights        []Mat
	Biases         []Mat
	activationFunc func(float64) float64
}

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
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
func NetworkNew(layers []int) Network {
	layerCount := len(layers)
	newNetwork := Network{LayerCount: layerCount, Layout: layers}
	newNetwork.Activation = make([]Mat, layerCount)
	newNetwork.Weights = make([]Mat, layerCount-1)
	newNetwork.Biases = make([]Mat, layerCount-1)
	//Initialise activation layers
	for layer := 0; layer < layerCount; layer++ {
		newNetwork.Activation[layer] = MatNew(1, layers[layer], Zero)
	}
	//Initialise weights and biases
	for layer := 0; layer < (layerCount - 1); layer++ {
		prevActivationLayer := &newNetwork.Activation[layer]
		nextActivationLayer := &newNetwork.Activation[layer+1]
		newNetwork.Weights[layer] = MatNew(prevActivationLayer.Cols, nextActivationLayer.Cols, rand.Float64)
		newNetwork.Biases[layer] = MatNew(1, nextActivationLayer.Cols, rand.Float64)
	}
	return newNetwork
}
func NetworkNewZero(layers []int) Network {
	layerCount := len(layers)
	newNetwork := Network{LayerCount: layerCount, Layout: layers}
	newNetwork.Activation = make([]Mat, layerCount)
	newNetwork.Weights = make([]Mat, layerCount-1)
	newNetwork.Biases = make([]Mat, layerCount-1)
	//Initialise activation layers
	for layer := 0; layer < layerCount; layer++ {
		newNetwork.Activation[layer] = MatNew(1, layers[layer], Zero)
	}
	//Initialise weights and biases
	for layer := 0; layer < (layerCount - 1); layer++ {
		prevActivationLayer := &newNetwork.Activation[layer]
		nextActivationLayer := &newNetwork.Activation[layer+1]
		newNetwork.Weights[layer] = MatNew(prevActivationLayer.Cols, nextActivationLayer.Cols, Zero)
		newNetwork.Biases[layer] = MatNew(1, nextActivationLayer.Cols, Zero)
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
		MatApply(&m.Activation[layer+1], Sigmoid)
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

func (network *Network) Adjust(gradient Network, rate float64) {
	for layer := 0; layer < (network.LayerCount - 1); layer++ {
		for i := 0; i < gradient.Weights[layer].Rows; i++ {
			for j := 0; j < gradient.Weights[layer].Cols; j++ {
				gradient.Weights[layer].Data[i][j] *= -1 * rate
			}
		}
		for i := 0; i < gradient.Biases[layer].Rows; i++ {
			for j := 0; j < gradient.Biases[layer].Cols; j++ {
				gradient.Biases[layer].Data[i][j] *= -1 * rate
			}
		}
		network.Weights[layer] = MatSum(network.Weights[layer], gradient.Weights[layer])
		network.Biases[layer] = MatSum(network.Biases[layer], gradient.Biases[layer])
	}
}

func (network *Network) FiniteDiff(trainIn Mat, trainOut Mat, eps float64, rate float64) {
	var saved float64
	gradient := NetworkNew(network.Layout)
	oldCost := network.Cost(trainIn, trainOut)

	for layer := 0; layer < (network.LayerCount - 1); layer++ {
		for i := 0; i < network.Weights[layer].Rows; i++ {
			for j := 0; j < network.Weights[layer].Cols; j++ {
				saved = network.Weights[layer].Data[i][j]
				network.Weights[layer].Data[i][j] += eps
				gradient.Weights[layer].Data[i][j] = (network.Cost(trainIn, trainOut) - oldCost) / eps
				network.Weights[layer].Data[i][j] = saved
			}
		}
		for i := 0; i < network.Biases[layer].Rows; i++ {
			for j := 0; j < network.Biases[layer].Cols; j++ {
				saved = network.Biases[layer].Data[i][j]
				network.Biases[layer].Data[i][j] += eps
				gradient.Biases[layer].Data[i][j] = (network.Cost(trainIn, trainOut) - oldCost) / eps
				network.Biases[layer].Data[i][j] = saved
			}
		}
	}

	for layer := 0; layer < (network.LayerCount - 1); layer++ {
		for i := 0; i < network.Weights[layer].Rows; i++ {
			for j := 0; j < network.Weights[layer].Cols; j++ {
				network.Weights[layer].Data[i][j] -= rate * gradient.Weights[layer].Data[i][j]
			}
		}
		for i := 0; i < network.Biases[layer].Rows; i++ {
			for j := 0; j < network.Biases[layer].Cols; j++ {
				network.Biases[layer].Data[i][j] -= rate * gradient.Biases[layer].Data[i][j]
			}
		}
	}

}

func (network *Network) Backprop(trainIn Mat, trainOut Mat, rate float64) {
	sampleCount := trainIn.Rows
	gradient := NetworkNewZero(network.Layout)
	//TODO: initialise to zero

	for sample := 0; sample < int(sampleCount); sample++ {
		network.Forward(trainIn.Row(sample))
		for i := 0; i < trainOut.Cols; i++ {
			gradient.Activation[len(gradient.Activation)-1].Data[0][i] = network.Activation[len(gradient.Activation)-1].Data[0][i] - trainOut.Data[sample][i]
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
	network.Adjust(gradient, rate)
}
