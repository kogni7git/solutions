import scala.language.postfixOps
import smile.data.`type`.{DataTypes, StructField}


object main {
  def main(args: Array[String]): Unit = {
    // Welcome!
    println("Welcome! This program trains a regression model for the Flight Delay Prediction Challenge on zindi.africa " +
      "(https://zindi.africa/competitions/flight-delay-prediction-challenge)!")
    val possibleModels = List("Ordinary Least Squares", "Ridge Regression", "Lasso Regression", "Regression Tree", "Random Forest")

    println("Please choose a regression model for training: ")
    var counter = 0
    for (i <- possibleModels) {
      println(s"${counter} for ${i}.")
      counter += 1
    }
    val chosenModel = scala.io.StdIn.readLine("Please enter the number of the desired model: ")
    val cv: String = scala.io.StdIn.readLine("On how many folds for cross-validation shall the model be trained? ")

    println(s"A ${possibleModels(chosenModel.toInt)} will be trained on $cv folds.")

    // Load data!
    val trainSchema = DataTypes.struct(
      new StructField("ID", DataTypes.StringType),
      new StructField("DATOP", DataTypes.StringType),
      new StructField("FLTID", DataTypes.StringType),
      new StructField("DEPSTN", DataTypes.StringType),
      new StructField("ARRSTN", DataTypes.StringType),
      new StructField("STD", DataTypes.StringType),
      new StructField("STA", DataTypes.StringType),
      new StructField("STATUS", DataTypes.StringType),
      new StructField("AC", DataTypes.StringType),
      new StructField("target", DataTypes.FloatType)
    )

    val testSchema = DataTypes.struct(
      new StructField("ID", DataTypes.StringType),
      new StructField("DATOP", DataTypes.StringType),
      new StructField("FLTID", DataTypes.StringType),
      new StructField("DEPSTN", DataTypes.StringType),
      new StructField("ARRSTN", DataTypes.StringType),
      new StructField("STD", DataTypes.StringType),
      new StructField("STA", DataTypes.StringType),
      new StructField("STATUS", DataTypes.StringType),
      new StructField("AC", DataTypes.StringType)
    )

    val sampleSubmissionSchema = DataTypes.struct(
      new StructField("ID", DataTypes.StringType),
      new StructField("target", DataTypes.FloatType)
    )

    val trainCSV = smile.read.csv("Data/Train.csv", delimiter = ",", header = true, schema = trainSchema)
    val testCSV = smile.read.csv("Data/Test.csv", delimiter = ",", header = true, schema = testSchema)
    val sampleSubmissionCSV = smile.read.csv("Data/SampleSubmission.csv", delimiter = ",", header = true, schema = sampleSubmissionSchema)

    // Prepare data!
    val (trainDF, testDF) = preparing.prepareData(trainCSV, testCSV)

    // Train model!
    val model = new modelling.Modelling(cv.toInt)
    val (y_test, rmse, settings) = model.trainModel(trainDF, testDF, possibleModels(chosenModel.toInt))

    println("Experiment successful!")

    // Make Submission!
    val submission = preparing.prepareSubmission(sampleSubmissionCSV, y_test)

    counter = 1
    while (new java.io.File("Submission/".concat(counter.toString)).exists()) {
      counter += 1
    }

    val output = "# Experiment Number " + counter + "\n## Model:\n" + possibleModels(chosenModel.toInt) + settings + "\n## Results:\n" + rmse

    new java.io.File("Submission/".concat(counter.toString)).mkdir()

    new java.io.PrintWriter("Submission/".concat(counter.toString).concat("/submission.csv")) {
        write(submission)
        close()
    }
    new java.io.PrintWriter("Submission/".concat(counter.toString).concat("/output.md")) {
        write(output)
        close()
    }

    println(s"submission.csv and output.md are saved under Submission/$counter")
  }
}