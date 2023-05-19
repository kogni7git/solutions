import java.nio.file.Files
import java.nio.file.Paths
import java.io.File

import org.apache.commons.csv.CSVFormat
import org.apache.commons.csv.CSVParser

fun main(args: Array<String>) {

    val seed: Long = 42

    // Welcome!
    println("Welcome! This program trains an Artificial Neural Network for the Richter's Predictor: Modeling Earthquake Damage competition on drivendata.org (https://www.drivendata.org/competitions/57/nepal-earthquake/)!")
    val settings = mutableMapOf<String, String>()
    println("Please enter the number of epochs: ")
    settings["Epochs"] = readLine().toString()
    println("Please enter the learning rate: ")
    settings["Learning Rate"] = readLine().toString()
    println("Please enter the batch size: ")
    settings["Batch Size"] = readLine().toString()
    println("Please enter the hidden size: ")
    settings["Hidden Size"] = readLine().toString()
    println("Please enter the dropout rate: ")
    settings["Dropout"] = readLine().toString()

    println("An Artificial Neural Network with " +
            settings["Hidden Size"].toString() + " hidden neurons and Dropout of " +
            settings["Dropout"].toString() + " will be trained " +
            settings["Epochs"].toString() + " epochs with the batch size " +
            settings["Batch Size"].toString() + " and the learning rate " +
            settings["Learning Rate"].toString() + " on 80% of the data and evaluated on 20% of the data.")

    // Load data!
    println("Data will be loaded and prepared!")
    val readerTrainValues = Files.newBufferedReader(Paths.get("Data/train_values.csv"))
    val parserTrainValues = CSVParser(readerTrainValues, CSVFormat.DEFAULT.withFirstRecordAsHeader().withIgnoreHeaderCase().withTrim().withDelimiter(','))
    val readerTrainValuesHelper = Files.newBufferedReader(Paths.get("Data/train_values.csv"))
    val parserTrainValuesHelper = CSVParser(readerTrainValuesHelper, CSVFormat.DEFAULT.withFirstRecordAsHeader().withIgnoreHeaderCase().withTrim().withDelimiter(','))

    val readerTrainLabels = Files.newBufferedReader(Paths.get("Data/train_labels.csv"))
    val parserTrainLabels = CSVParser(readerTrainLabels, CSVFormat.DEFAULT.withFirstRecordAsHeader().withIgnoreHeaderCase().withTrim().withDelimiter(','))

    val readerTestValues = Files.newBufferedReader(Paths.get("Data/test_values.csv"))
    val parserTestValues = CSVParser(readerTestValues, CSVFormat.DEFAULT.withFirstRecordAsHeader().withIgnoreHeaderCase().withTrim().withDelimiter(','))

    val readerSubmissionFormat = Files.newBufferedReader(Paths.get("Data/submission_format.csv"))
    val parserSubmissionFormat = CSVParser(readerSubmissionFormat, CSVFormat.DEFAULT.withFirstRecordAsHeader().withIgnoreHeaderCase().withTrim().withDelimiter(','))

    // prepare data
    val (trainValuesPrepared, trainLabels, testValuesPrepared) = prepareData(
        parserTrainValues,
        parserTrainValuesHelper,
        parserTrainLabels,
        parserTestValues
    )

    // clustering
    println("Shall clustering be used for variance reduction? [y/n]")
    val cl = readLine().toString()
    var algorithm = ""
    val algorithms = arrayOf("G-Means", "K-Means", "X-Means")

    val trainValues: Array<DoubleArray>
    val testValues: Array<DoubleArray>

    if (cl == "y") {
        println("Which clustering algorithm shall be used?")
        for ((c, a) in algorithms.withIndex()) {
            println("$c for $a")
        }
        algorithm = readLine().toString()
        val (t1, t2) = clusterData(trainValuesPrepared, testValuesPrepared, algorithm)
        trainValues = t1
        testValues = t2
    } else {
        trainValues = trainValuesPrepared
        testValues = testValuesPrepared
    }

    // create data
    val (trainingData, evalData, testData) = createData(trainValues, trainLabels, testValues, seed)

    // Make net
    val model = makeModel(
        seed,
        38,
        settings["Hidden Size"].toString().toInt(),
        settings["Dropout"].toString().toDouble(),
        settings["Learning Rate"].toString().toDouble()
    )
    model.init()

    // Train model!
    val (predictions, results) = trainModel(
        settings["Epochs"].toString().toInt(),
        settings["Batch Size"].toString().toInt(),
        model,
        trainingData,
        evalData,
        testData
    )

    println("\nExperiment successful!")

    // Make Submission!
    val submission = createSubmission(parserSubmissionFormat, predictions)

    var counter = 1
    while (File("Submission/".plus(counter)).exists()) {
        counter += 1
    }

    var output = "# Experiment Number $counter\n## Settings:"
    if (cl == "y") {
        output += "\n* Clustering: " + algorithms[algorithm.toInt()]
    }
    output += "\n* Epochs: " + settings["Epochs"].toString() +
            "\n* Dropout: " + settings["Dropout"].toString() +
            "\n* Learning Rate: " + settings["Learning Rate"].toString() +
            "\n* Hidden Size: " + settings["Hidden Size"].toString() +
            "\n* Batch Size: " + settings["Batch Size"].toString() +
            "\n## Results:" + results

    File("Submission/".plus(counter)).mkdirs()
    File("Submission/".plus(counter).plus("/submission.csv")).writeText(submission)
    File("Submission/".plus(counter).plus("/output.md")).writeText(output)
    println("submission.csv and output.md are saved under " + "Submission/".plus(counter))
}