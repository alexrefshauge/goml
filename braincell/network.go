package braincell

import (
	"fmt"
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
		return
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
	fmt.Println("Weight/bias layers initialised!")
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
