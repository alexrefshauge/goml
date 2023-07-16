package braincell

import (
	"fmt"
)

type Mat struct {
	Rows, Cols int
	Data       [][]float64
}

func Zero() float64 {
	return 0
}

// Transpose of matrix
func (m Mat) Transpose() Mat {
	var resMat Mat = MatNew(m.Cols, m.Rows, func() float64 {
		return 0
	})

	for i := 0; i < resMat.Rows; i++ {
		for j := 0; j < resMat.Cols; j++ {
			resMat.Data[i][j] = m.Data[j][i]
		}
	}
	return resMat
}

func (m Mat) Row(rowIndex int) Mat {
	if rowIndex > m.Rows {
		fmt.Println("row index out of bounds!")
	}
	var rowMat Mat = MatNew(1, m.Cols, func() float64 { return 0 })
	for i := 0; i < rowMat.Cols; i++ {
		rowMat.Data[0][i] = m.Data[rowIndex][i]
	}
	return rowMat
}

// Product of two matrices
func MatDot(a Mat, b Mat) Mat {
	if a.Cols != b.Rows {
		fmt.Println("ERROR!: a.Cols not matching b.Rows")
	}

	var m Mat = MatNew(a.Rows, b.Cols, func() float64 {
		return 0
	})
	n := a.Cols
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			for k := 0; k < n; k++ {
				m.Data[i][j] += a.Data[i][k] * b.Data[k][j]
			}
		}
	}

	return m
}

// Sum of 2 matrices
func MatSum(a Mat, b Mat) Mat {
	if a.Rows != b.Rows {
		fmt.Println("ERROR!: a.Rows != b.Rows")
	}
	if a.Cols != b.Cols {
		fmt.Println("ERROR!: a.Cols != b.Cols")
	}
	var sum Mat = MatNew(a.Rows, a.Cols, func() float64 { return 0 })
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Cols; j++ {
			sum.Data[i][j] = a.Data[i][j] + b.Data[i][j]
		}
	}
	return sum
}

func MatApply(m *Mat, fn func(x float64) float64) {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Data[i][j] = fn(m.Data[i][j])
		}
	}
}

// Prints a matrix
func MatPrint(m Mat, label string) {
	fmt.Printf("[ %s\n", label)
	for row := 0; row < m.Rows; row++ {
		for col := 0; col < m.Cols; col++ {
			fmt.Printf("%5.2f ", m.Data[row][col])
		}
		fmt.Printf("\n")
	}
	fmt.Println("]")
}

// Initialise new matrix
func MatNew(rows, cols int, fn func() float64) Mat {
	var newMat Mat = Mat{
		Rows: rows,
		Cols: cols,
		Data: make([][]float64, rows),
	}
	for r := 0; r < rows; r++ {
		newMat.Data[r] = make([]float64, cols)
		for c := 0; c < cols; c++ {
			newMat.Data[r][c] = fn()
		}
	}
	return newMat
}
