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

const learningRate float64 = 0.1

func rgbToGrey(r float64, g float64, b float64) float64 {
	return 0.299*r + 0.587*g + 0.114*b
}

func generateImage(path string, size int, net Network) {
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
			g := uint8(col[1] * 255)
			b := uint8(col[2] * 255)
			img.Set(x, y, color.RGBA{r, g, b, 255})
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
	trainingImage, err := png.Decode(appleTrain64)
	if err != nil {
		log.Fatal("Error!: Failed to read image")
	}

	cycles := 50000
	network := NetworkNew([]Layer{
		{NeuronCount: 2, ActivationFunc: TanH},
		{NeuronCount: 8, ActivationFunc: ReLU},
		{NeuronCount: 3, ActivationFunc: TanH},
	}, func() float64 { return rand.Float64()*0.4 - 0.2 },
		func() float64 { return rand.Float64()*2 - 1 },
	)

	trainingData := MatNew(resWidth*resHeight, 5, Zero)
	trainingDataIn := MatNew(resWidth*resHeight, 2, Zero)
	trainingDataOut := MatNew(resWidth*resHeight, 3, Zero)
	for y := 0; y < resHeight; y++ {
		for x := 0; x < resWidth; x++ {
			pixel := trainingImage.At(x, y)
			r, g, b, _ := pixel.RGBA()
			xx := float64(x+1) / float64(resWidth)
			yy := float64(y+1) / float64(resHeight)
			//brightness := rgbToGrey(float64(r>>8), float64(g>>8), float64(b>>8)) / float64(255)
			trainingData.Data[y*resWidth+x] = []float64{
				xx,
				yy,
				float64(r>>8) / 255,
				float64(g>>8) / 255,
				float64(b>>8) / 255,
			}
		}
	}

	var rate float64 = learningRate
	var volatility float64 = 0
	var cost float64
	for i := 0; i < cycles; i++ {
		MatRowShuffle(&trainingData)
		for row := 0; row < trainingData.Rows; row++ {
			trainingDataIn.Data[row][0] = trainingData.Data[row][0]
			trainingDataIn.Data[row][1] = trainingData.Data[row][1]
			trainingDataOut.Data[row][0] = trainingData.Data[row][2]
			trainingDataOut.Data[row][1] = trainingData.Data[row][3]
			trainingDataOut.Data[row][2] = trainingData.Data[row][4]
		}
		network.Backprop(trainingDataIn, trainingDataOut, rate)
		oldCost := cost
		cost = network.Cost(trainingDataIn, trainingDataOut)
		fmt.Printf("epoch: %07d rate: %0.02f cost: %.03f vol: %0.02f\n", i, rate, cost, volatility)

		if i%10 == 0 {
			generateImage("progress.png", 32, network)
		}
		if i%1000 == 0 {
			generateImage("result.png", 128, network)
		}

		//adjust rate according to volatility
		if i > 300 {
			rate = 0.05
			deltaCost := cost - oldCost
			if volatility > 0 {
				volatility--
			}
			if deltaCost > 0 {
				volatility += 2
			}

			if volatility > 2 {
				rate /= 1.001
			}
		}
	}
	generateImage("progress.png", 1024, network)
	fmt.Println("seed: ", seed)
}
