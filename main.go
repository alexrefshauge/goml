package main

import (
	"fmt"
	. "goml/braincell"
	"math/rand"
)

var trainData = [][2]int{
	{0, 0},
	{1, 2},
	{2, 4},
	{3, 6},
	{4, 8},
}
var trainCount = len(trainData)

const eps = 1e-3
const learningRate = 1e-3

func model(x float64, w float64) float64 {
	return x * w
}

// func cost(w float64) float64 {
// 	var error float64 = 0
// 	for i := 0; i < trainCount; i++ {
// 		var x float64 = float64(trainData[i][0])
// 		var y float64 = model(x, w)
// 		var residual float64 = y - float64(trainData[i][1])
// 		error += residual * residual
// 		// fmt.Printf("got: %v - expected: %v\n", y, trainData[i][1])
// 	}

// 	var avgError float64 = error / float64(trainCount)
// 	return avgError
// }

type modelXOR struct {
	a0         Mat
	w1, b1, a1 Mat
	w2, b2, a2 Mat
}

func forwardXOR(m *modelXOR) {
	m.a1 = MatDot(m.a0, m.w1)
	m.a1 = MatSum(m.a1, m.b1)
	MatApply(m.a1, Sigmoid)

	m.a2 = MatDot(m.a1, m.w2)
	m.a2 = MatSum(m.a2, m.b2)
	MatApply(m.a2, Sigmoid)
}

func finiteDiff(m *modelXOR, g *modelXOR, traininIn Mat, trainingOut Mat, eps float64) {
	var saved float64
	oldCost := cost(m, traininIn, trainingOut)

	for i := 0; i < m.w1.Rows; i++ {
		for j := 0; j < m.w1.Cols; j++ {
			saved = m.w1.Data[i][j]
			m.w1.Data[i][j] += eps
			g.w1.Data[i][j] = (cost(m, traininIn, trainingOut) - oldCost) / eps
			m.w1.Data[i][j] = saved
		}
	}
	for i := 0; i < m.w2.Rows; i++ {
		for j := 0; j < m.w2.Cols; j++ {
			saved = m.w2.Data[i][j]
			m.w2.Data[i][j] += eps
			g.w2.Data[i][j] = (cost(m, traininIn, trainingOut) - oldCost) / eps
			m.w2.Data[i][j] = saved
		}
	}
	for i := 0; i < m.b1.Rows; i++ {
		for j := 0; j < m.b1.Cols; j++ {
			saved = m.b1.Data[i][j]
			m.b1.Data[i][j] += eps
			g.b1.Data[i][j] = (cost(m, traininIn, trainingOut) - oldCost) / eps
			m.b1.Data[i][j] = saved
		}
	}
	for i := 0; i < m.b2.Rows; i++ {
		for j := 0; j < m.b2.Cols; j++ {
			saved = m.b2.Data[i][j]
			m.b2.Data[i][j] += eps
			g.b2.Data[i][j] = (cost(m, traininIn, trainingOut) - oldCost) / eps
			m.b2.Data[i][j] = saved
		}
	}
}

func learn(m *modelXOR, g *modelXOR, rate float64) {
	for i := 0; i < m.w1.Rows; i++ {
		for j := 0; j < m.w1.Cols; j++ {
			m.w1.Data[i][j] -= rate * g.w1.Data[i][j]
		}
	}
	for i := 0; i < m.w2.Rows; i++ {
		for j := 0; j < m.w2.Cols; j++ {
			m.w2.Data[i][j] -= rate * g.w2.Data[i][j]
		}
	}
	for i := 0; i < m.b1.Rows; i++ {
		for j := 0; j < m.b1.Cols; j++ {
			m.b1.Data[i][j] -= rate * g.b1.Data[i][j]
		}
	}
	for i := 0; i < m.b2.Rows; i++ {
		for j := 0; j < m.b2.Cols; j++ {
			m.b2.Data[i][j] -= rate * g.b2.Data[i][j]
		}
	}
}

func cost(m *modelXOR, traininIn Mat, trainingOut Mat) float64 {
	if traininIn.Cols == trainingOut.Cols {
		panic(0)
	}

	var cost float64 = 0

	for sample := 0; sample < traininIn.Rows; sample++ {
		m.a0 = traininIn.Row(sample)
		forwardXOR(m)
		var expected float64 = trainingOut.Data[sample][0]
		var got float64 = m.a2.Data[0][0]
		residual := expected - got
		cost += residual * residual
	}

	return cost
}

func main() {
	var eps float64 = 1e-3
	var learningRate float64 = 1e-1

	rand.Seed(69420)

	var traininDataIn Mat = Mat{Rows: 4, Cols: 2, Data: [][]float64{{0, 0}, {1, 0}, {0, 1}, {1, 1}}}
	var traininDataOut Mat = Mat{Rows: 4, Cols: 1, Data: [][]float64{{0}, {1}, {1}, {0}}}

	//input
	x := MatNew(1, 2, Zero)
	x.Data = [][]float64{{1, 1}}

	var model *modelXOR = new(modelXOR)
	var gradient *modelXOR = new(modelXOR)

	//layers
	model.a0 = MatNew(1, 2, Zero)
	model.w1 = MatNew(2, 2, rand.Float64)
	model.b1 = MatNew(1, 2, rand.Float64)
	model.a1 = MatNew(1, 2, Zero)
	model.w2 = MatNew(2, 1, rand.Float64)
	model.b2 = MatNew(1, 1, rand.Float64)
	model.a2 = MatNew(1, 1, Zero)
	//a2 := MatNew(1, 1, zero)

	//arbitary gradient matrix
	gradient.a0 = MatNew(1, 2, Zero)
	gradient.w1 = MatNew(2, 2, Zero)
	gradient.b1 = MatNew(1, 2, Zero)
	gradient.a1 = MatNew(1, 2, Zero)
	gradient.w2 = MatNew(2, 1, Zero)
	gradient.b2 = MatNew(1, 1, Zero)
	gradient.a2 = MatNew(1, 1, Zero)

	fmt.Println(cost(model, traininDataIn, traininDataOut))
	cycles := 1000
	width := 100
	for i := 0; i < width-1; i++ {
		if i%(width/10) == 0 {
			fmt.Print("I")
		} else {
			fmt.Print(".")
		}
	}
	fmt.Print("I\n")
	for i := 0; i < cycles; i++ {
		finiteDiff(model, gradient, traininDataIn, traininDataOut, eps)
		learn(model, gradient, learningRate)
		if i%(cycles/width) == 0 {
			fmt.Print("█")
		}
	}
	fmt.Print("\n")
	fmt.Println(cost(model, traininDataIn, traininDataOut))

	//test
	for i := 0; i < traininDataIn.Rows; i++ {
		model.a0 = traininDataIn.Row(i)
		forwardXOR(model)
		fmt.Printf("TEST: %v ^ %v => %v\n", model.a0.Data[0][0], model.a0.Data[0][1], model.a2.Data[0][0])
	}

	network := NetworkNew([]int{4, 5, 2})
	fmt.Printf("Network with %v layers initialised!\n", network.LayerCount)
	MatPrint(network.Activation[0], "a0")
	MatPrint(network.Weights[0], "w1")
	MatPrint(network.Biases[0], "b1")
	MatPrint(network.Activation[1], "a1")
	MatPrint(network.Weights[1], "w2")
	MatPrint(network.Biases[1], "b2")
	MatPrint(network.Activation[2], "a2")
}
