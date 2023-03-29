/* Tunisian Fraud Detection Challenge
 * author: kogni7
 * date: autumn 2022 / winter 2022/23
 * 
 * This program trains a regression model using mlpack for the Tunisian Fraud Detection Challenge on zindi.africa.
 * The task is to predict the amount of the tax liability of a taxpayer.
 * 
 * The structure of the project is:
 * - FraudDetectionAI.go
 * - Data/
 * 	- data files
 * - Submission/
 * 	- directories with the submission files
 * 
 * Start the program within a terminal from the directory of FraudDetectionAI.go with the command:
 * go run FraudDetectionAI.go
 */


package main
import (
	"mlpack.org/v1/mlpack"
	"gonum.org/v1/gonum/mat"
	"fmt"
	"strconv"
	"os"
	"encoding/csv"
	"sort"
	"math"
)

// The Root Mean Squared Error!
func RMSE(y_pred *mat.Dense, y_true *mat.Dense) float64 {
	var N, _ = y_pred.Dims()
	var sum float64
	for i:=0; i<N; i++ {
		sum += math.Pow(y_pred.At(i, 0) - y_true.At(i, 0), 2)
	}
	sum /= float64(N)
	return math.Sqrt(sum)
}


func prepareData(file *os.File, preprocessing int, mode string) (*mat.Dense, *mat.Dense) {
	// read file
	csvReader := csv.NewReader(file)
	records, _ := csvReader.ReadAll()

	// column 2 is categorical, so convert to integer
	var categoricals = make(map[string]string)
	c := 0
	for i:=1; i<len(records); i++ {
		key := records[i][2]
		_, ok := categoricals[key]
		if !ok {
			categoricals[key] = strconv.Itoa(c)
			c++
		}
	}
	for i:=1; i<len(records); i++ {
		for j:=0; j<len(records[0]); j++ {
			if j == 2 {
				records[i][j] = categoricals[records[i][j]]
			}
		}
	}
	// imputation of NAs
	var columnsWithoutNAs = make(map[int]int) 
	// remove columns with NAs
	if preprocessing == 0 {
		columnsWithNAs := make(map[int]int)
		for j:=0; j<len(records[0]); j++ {
			for i:=1; i<len(records); i++ {
				if records[i][j] == "" {
					columnsWithNAs[j] = j
					break
				}
			}
		}
		for j:=0; j<len(records[0]); j++ {
			_, ok := columnsWithNAs[j]
			if !ok {
				columnsWithoutNAs[j] = j
			}
		}
	// impute with 0
	} else if preprocessing == 1 { 
		for i:=1; i<len(records); i++ {
			for j:=0; j<len(records[0]); j++ {
				if records[i][j] == "" {
					records[i][j] = "0.0"
				} else {
					continue
				}
			}
		}
	// impute with the median
	} else if preprocessing == 2 {
		medians := []float64{}
		for j:=0; j<len(records[0]); j++ {
			column := []float64{}
			for i:=1; i<len(records); i++ {
				var c, _ = strconv.ParseFloat(records[i][j], 64)
				column = append(column, c)
			}
			sort.Float64s(column)
			if len(column) % 2 == 0 {
				upperMedian := column[int64(len(column)/2)]
				lowerMedian := column[int64((len(column)/2)-1)]
				median := (upperMedian + lowerMedian) / 2
				medians = append(medians, median)
			} else {
				medians = append(medians, column[int64(len(column)/2)])
			}
		}
		for i:=1; i<len(records); i++ {
			for j:=0; j<len(records[0]); j++ {
				if records[i][j] == "" {
					records[i][j] = fmt.Sprintf("%f", medians[j])
				} else {
					continue
				}
			}
		}
	// impute with the mean
	} else if preprocessing == 3 {
		means := []float64{}
		for j:=0; j<len(records[0]); j++ {
			mean := 0.0
			for i:=1; i<len(records); i++ {
				var m, _ = strconv.ParseFloat(records[i][j], 64)
				mean += m
			}
			means = append(means, mean/(float64((len(records)-1))))
		}
		for i:=1; i<len(records); i++ {
			for j:=0; j<len(records[0]); j++ {
				if records[i][j] == "" {
					records[i][j] = fmt.Sprintf("%f", means[j])
				} else {
					continue
				}
			}
		}
	}

	// prepare data and target
	var mapOfData = make(map[int]string)
	var k = 0
	for i:=1; i<len(records); i++ {
		for j:=0; j<len(records[0])-1; j++ {
			if mode == "train" && j == 111 {
				k -= 1
			} else {
				if preprocessing == 0 {
					_, ok := columnsWithoutNAs[j]
					if ok {
						mapOfData[i*(len(records[0])-2) + k] = records[i][j]
					} else {
						k -= 1
					}
				} else {
					mapOfData[i*(len(records[0])-2) + k] = records[i][j]
				}
			}
			k++
		}
	}
	
	if mode == "train" {
		data := []float64{}
		for i:=0; i<(len(records)-1)*(len(records[0])-2); i++ {
			d, _ := strconv.ParseFloat(mapOfData[i], 64)
			data = append(data, d)
		}
		X := mat.NewDense(len(records)-1, len(records[0])-2, data)
		
		target := []float64{}
		for i:=0; i<len(records)-1; i++ {
			t, _ := strconv.ParseFloat(records[i][111], 64)
			target = append(target, t)
		}
		y := mat.NewDense(len(records)-1, 1, target)
		return X, y
	} else {
		data := []float64{}
		for i:=0; i<(len(records)-1)*(len(records[0])-1); i++ {
			d, _ := strconv.ParseFloat(mapOfData[i], 64)
			data = append(data, d)
		}
		X := mat.NewDense(len(records)-1, len(records[0])-1, data)
		y := mat.NewDense(len(records), 1, nil)
		return X, y
	}
}


func main() {
	// Welcome!
	fmt.Println("Welcome! This program trains a regression model for the Tunisian Fraud Detection Challenge on zindi.africa (https://zindi.africa/competitions/tunisian-fraud-detection)!")

	var possiblePreprocessing = [4]string{"remove columns with NAs", "replace NAs with 0", "replace NAs with the median", "replace NAs with the mean"}
	fmt.Println("Please choose the preprocessing: ")
	for i:=0; i<len(possiblePreprocessing); i++ {
		fmt.Println(strconv.Itoa(i) + " for " + possiblePreprocessing[i] + ".")
	}
	fmt.Println("Please enter the number of the desired preprocessing: ")
	var chosenPreprocessingString string
	fmt.Scanln(&chosenPreprocessingString)
	var chosenPreprocessing, _ = strconv.Atoi(chosenPreprocessingString)

	var possibleModels = [4]string{"Bayesian Linear Regression", "Lars", "Linear Regression", "Ridge Regression"}
	fmt.Println("Please choose a regression model for training: ")
	for i:=0; i < len(possibleModels); i++ {
		fmt.Println(strconv.Itoa(i) + " for " + possibleModels[i] + ".")
	}
	fmt.Println("Please enter the number of the desired model: ")
	var chosenModelString string
	fmt.Scanln(&chosenModelString)
	var chosenModel, _ = strconv.Atoi(chosenModelString)

	fmt.Println("A " + possibleModels[chosenModel] + " model will be trained using " + possiblePreprocessing[chosenPreprocessing] + " as preprocessing method. The model will be trained on 75% of the data and evaluated on 25% of the data.")
	
	// Load data!
	fileTrain, _ := os.Open("Data/SUPCOM_Train.csv")
	fileTest, _ := os.Open("Data/SUPCOM_Test.csv")

	// Prepare data!
	var X, y  = prepareData(fileTrain, chosenPreprocessing, "train")
	var X_test, _ = prepareData(fileTest, chosenPreprocessing, "test")

	fileTrain.Close()
	fileTest.Close()

	// Train model!
	var y_test *mat.Dense
	var rmse float64
	var settings string =  "\n## Settings:\n* preprocessing: " + possiblePreprocessing[chosenPreprocessing]
		
	// data split
	paramsSplit := mlpack.PreprocessSplitOptions()
	paramsSplit.NoShuffle = false
	paramsSplit.Seed = 42
	paramsSplit.TestRatio = 0.25
	paramsSplit.InputLabels = y
	X_val, y_val, X, y := mlpack.PreprocessSplit(X, paramsSplit)

	if chosenModel == 0 {
		// training
		param := mlpack.BayesianLinearRegressionOptions()
		fmt.Println("Shall the data be centered? [true/false]")
		var center string
		fmt.Scanln(&center)
		param.Center, _ = strconv.ParseBool(center)
		settings += "\n* Center: " + center
		param.Input = X
		param.InputModel = nil
		param.Responses = y
		fmt.Println("Shall the data be scaled? [true/false]")
		var scale string
		fmt.Scanln(&scale)
		param.Scale, _ = strconv.ParseBool(scale)
		settings += "\n* Scale: " + scale
		param.Test = X_val
		model, y_pred, _ := mlpack.BayesianLinearRegression(param)
		rmse = RMSE(y_pred, y_val)
		// prediction
		paramPrediction := mlpack.BayesianLinearRegressionOptions()
		paramPrediction.Test = X_test
		paramPrediction.InputModel = &model
		_, y_test, _ = mlpack.BayesianLinearRegression(paramPrediction)
	} else if chosenModel == 1 {
		// training
		param := mlpack.LarsOptions()
		param.Input = mat.DenseCopyOf(X.T())
		param.InputModel = nil
		fmt.Println("Lambda1? ")
		var lambda1 string
		fmt.Scanln(&lambda1)
		param.Lambda1, _ = strconv.ParseFloat(lambda1, 64)
		settings += "\n* lambda1: " + lambda1
		fmt.Println("Lambda2? ")
		var lambda2 string
		fmt.Scanln(&lambda2)	
		param.Lambda2, _ = strconv.ParseFloat(lambda2, 64)
		settings += "\n* lambda2: " + lambda2
		param.UseCholesky = false
		param.Responses = y
		param.Test = mat.DenseCopyOf(X_val.T())
		model, y_pred := mlpack.Lars(param)
		rmse = RMSE(mat.DenseCopyOf(y_pred.T()), y_val)
		// prediction
		paramPrediction := mlpack.LarsOptions()
		paramPrediction.Test = mat.DenseCopyOf(X_test.T())
		paramPrediction.InputModel = &model
		_, y_test = mlpack.Lars(paramPrediction)
	} else if chosenModel == 2 {
		// training
		param := mlpack.LinearRegressionOptions()
		param.InputModel = nil
		param.Lambda = 0
		param.Training = X
		param.TrainingResponses = y
		param.Test = X_val
		model, y_pred := mlpack.LinearRegression(param)
		rmse = RMSE(y_pred, y_val)
		// prediction
		paramPrediction := mlpack.LinearRegressionOptions()
		paramPrediction.Test = X_test
		paramPrediction.InputModel = &model
		_, y_test = mlpack.LinearRegression(paramPrediction)
		settings = ""
	} else if chosenModel == 3 {
		// training
		param := mlpack.LinearRegressionOptions()
		param.InputModel = nil
		fmt.Println("Lambda? ")
		var lambda string
		fmt.Scanln(&lambda)
		param.Lambda, _ = strconv.ParseFloat(lambda, 64)
		settings += "\n* lambda: " + lambda
		param.Training = X
		param.TrainingResponses = y
		param.Test = X_val
		model, y_pred := mlpack.LinearRegression(param)
		rmse = RMSE(y_pred, y_val)
		// prediction
		paramPrediction := mlpack.LinearRegressionOptions()
		paramPrediction.Test = X_test
		paramPrediction.InputModel = &model
		_, y_test = mlpack.LinearRegression(paramPrediction)
	}

	fmt.Println("Experiment successful!")

	// Make submission!
	file, _ := os.Open("Data/SUPCOM_Test.csv")
	csvReader := csv.NewReader(file)
	records, _ := csvReader.ReadAll()
	id := []string{}
	for i:=1; i<len(records); i++ {
		id = append(id, records[i][len(records[0])-1])
	}
	file.Close()

	var submission string = "id,target\n"
	for i:=0; i<len(id); i++ {
		var item float64
		if chosenModel == 1 {
			item = y_test.At(0, i)
		} else {
			item = y_test.At(i, 0)
		}		
		submission += id[i] + "," + fmt.Sprintf("%f", item) + "\n"
	}	
	
	var counter int = 1
	for i:=1; i < 10000; i++ {
		if _, err := os.Stat("Submission/" + strconv.Itoa(counter)); os.IsNotExist(err) {
			break
		} else {
			counter++
		}
	}

	var output string = "# Experiment Number " + strconv.Itoa(counter) + "\n## Model:\n" + possibleModels[chosenModel] + settings + "\n## Results:\n * RMSE: " + fmt.Sprintf("%f", rmse)
	
	os.Mkdir("Submission/"+ strconv.Itoa(counter), 0700)
	_ = os.WriteFile("Submission/" + strconv.Itoa(counter) + "/submission.csv", []byte(submission), 0644)
	_ = os.WriteFile("Submission/" + strconv.Itoa(counter) + "/output.md", []byte(output), 0644)
	fmt.Println("submission.csv and output.md are saved under Submission/" + strconv.Itoa(counter))
}
