package main

import (
	"fmt"
	. "goml/braincell"
	"image"
	"image/color"
	"image/png"
	_ "image/png"
	"log"
	"math/rand"
	"os"
	"time"
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
const learningRate float64 = 0.1

func rgbToGrey(r float64, g float64, b float64) float64 {
	return 0.299*r + 0.587*g + 0.114*b
}

func saveImage(path string, size int, net Network) {
	file, _ := os.Create(path)
	upLeft := image.Point{0, 0}
	lowRight := image.Point{size, size}
	img := image.NewRGBA(image.Rectangle{upLeft, lowRight})

	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			input := MatNew(1, 2, Zero)
			input.Data[0][0] = float64(x) / float64(size)
			input.Data[0][1] = float64(y) / float64(size)
			net.Forward(input)
			output := net.Activation[net.LayerCount-1]
			col := output.Data[0]
			r := uint8(col[0] * 255)
			img.Set(x, y, color.Gray{r})
		}
	}

	png.Encode(file, img)
}

func main() {
	// var eps float64 = 1e-3
	// var learningRate float64 = 1e-1

	seed := time.Now().UnixNano()
	rand.Seed(seed)

	appleTrain64, err := os.Open("./datasets/3.png")
	resWidth := 32
	resHeight := 32
	trainingData, err := png.Decode(appleTrain64)
	if err != nil {
		log.Fatal("Error!: Failed to read image")
	}

	cycles := 50000
	network := NetworkNew([]Layer{
		{NeuronCount: 2, ActivationFunc: Sigmoid},
		{NeuronCount: 16, ActivationFunc: Sigmoid},
		{NeuronCount: 16, ActivationFunc: Sigmoid},
		{NeuronCount: 1, ActivationFunc: Sigmoid},
	}, func() float64 { return rand.Float64()*0.4 - 0.2 },
		func() float64 { return rand.Float64()*2 - 1 },
	)

	trainingDataIn := MatNew(resWidth*resHeight, 2, func() float64 { return 0 })
	trainingDataOut := MatNew(resWidth*resHeight, 1, func() float64 { return 0 })
	for y := 0; y < resHeight; y++ {
		for x := 0; x < resWidth; x++ {
			trainingDataIn.Data[y*x+x][0] = (float64(x) / float64(resWidth))
			trainingDataIn.Data[y*x+x][1] = (float64(y) / float64(resHeight))

			pixel := trainingData.At(x, y)
			r, g, b, _ := pixel.RGBA()
			trainingDataOut.Data[y*x+x] = []float64{rgbToGrey(float64(r>>8), float64(g>>8), float64(b>>8)) / 255}
		}
	}

	for i := 0; i < cycles; i++ {
		var rate float64 = learningRate
		network.Backprop(trainingDataIn, trainingDataOut, rate)
		fmt.Printf("epoch: %07d rate: %0.02f cost: %.03f \n", i, rate, network.Cost(trainingDataIn, trainingDataOut))

		if i%10 == 0 {
			saveImage("progress.png", 12, network)
		}
		if i%1000 == 0 {
			saveImage("result.png", 128, network)
		}
	}
	saveImage("progress.png", 1024, network)
	fmt.Println("seed: ", seed)
}
