package main

import (
	"fmt"
	. "goml/braincell"
	_ "image/png"
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

func main() {
	// var eps float64 = 1e-3
	// var learningRate float64 = 1e-1

	rand.Seed(69)

	// appleTrain64, err := os.Open("./datasets/apple_12.png")
	// trainingData, err := png.Decode(appleTrain64)
	// if err != nil {
	// 	log.Fatal("Error!: Failed to read image")
	// }

	cycles := 100
	//network := NetworkNew([]int{2, 3, 3})

	//TEST
	networkXor := NetworkNew([]int{2, 2, 1})
	trainingDataIn := Mat{Rows: 4, Cols: 2, Data: [][]float64{{0, 0}, {1, 0}, {0, 1}, {1, 1}}}
	trainingDataOut := Mat{Rows: 4, Cols: 1, Data: [][]float64{{0}, {1}, {1}, {0}}}

	gradient := NetworkNewZero(networkXor.Layout)
	// networkXor.Backprop(&gradient, trainingDataIn, trainingDataOut, 1)

	for i := 0; i < cycles; i++ {
		networkXor.Backprop(&gradient, trainingDataIn, trainingDataOut, 0.02)
		fmt.Println("\t\t", i, "\t", networkXor.Cost(trainingDataIn, trainingDataOut))
	}
	fmt.Println("0 ^ 0 : ", networkXor.Forward(trainingDataIn.Row(0)).Data[0][0])
	fmt.Println("1 ^ 0 : ", networkXor.Forward(trainingDataIn.Row(1)).Data[0][0])
	fmt.Println("0 ^ 1 : ", networkXor.Forward(trainingDataIn.Row(2)).Data[0][0])
	fmt.Println("1 ^ 1 : ", networkXor.Forward(trainingDataIn.Row(3)).Data[0][0])

	//TESTEND

	// trainingDataIn := MatNew(12*12, 2, func() float64 { return 0 })
	// trainingDataOut := MatNew(12*12, 3, func() float64 { return 0 })
	// for y := 0; y < 12; y++ {
	// 	for x := 0; x < 12; x++ {
	// 		trainingDataIn.Data[y*x+x][0] = float64(x)
	// 		trainingDataIn.Data[y*x+x][1] = float64(y)

	// 		pixel := trainingData.At(x, y)
	// 		r, g, b, _ := pixel.RGBA()
	// 		col := []float64{float64(r>>8) / 255, float64(g>>8) / 255, float64(b>>8) / 255}
	// 		trainingDataOut.Data[y*x+x] = col
	// 		trainingDataIn.Data[y*x+x] = []float64{float64(x) / 64, float64(y) / 64}
	// 	}
	// }

	// for i := 0; i < cycles; i++ {
	// 	fmt.Println("cycle", i)
	// 	network.FiniteDiff(trainingDataIn, trainingDataOut, eps, learningRate)
	// }

	// resWidth := 12
	// resHeight := 12
	// file, _ := os.Create("result.png")
	// upLeft := image.Point{0, 0}
	// lowRight := image.Point{resWidth, resHeight}
	// img := image.NewRGBA(image.Rectangle{upLeft, lowRight})

	// for y := 0; y < resHeight; y++ {
	// 	for x := 0; x < resWidth; x++ {
	// 		input := MatNew(1, 2, Zero)
	// 		input.Data[0][0] = float64(x) / float64(resWidth)
	// 		input.Data[0][1] = float64(y) / float64(resHeight)
	// 		network.Forward(input)
	// 		output := network.Activation[len(network.Activation)-1]
	// 		col := output.Data[0]
	// 		fmt.Println(input.Data[0][0], input.Data[0][1])
	// 		fmt.Println(output.Data[0][0], output.Data[0][1], output.Data[0][2])
	// 		r := uint8(col[0] * 255)
	// 		g := uint8(col[1] * 255)
	// 		b := uint8(col[2] * 255)
	// 		fmt.Println(r, g, b)
	// 		img.Set(x, y, color.RGBA{r, g, b, 255})
	// 	}
	// }

	// png.Encode(file, img)
	// fmt.Println(network.Cost(traininDataIn, traininDataOut))
}
