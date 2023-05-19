import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.GeneralizedLinearRegression
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql._

import java.nio.file.Files
import java.nio.file.Paths

import org.apache.log4j.{Level, Logger}

object Main {
  def main(args: Array[String]): Unit = {
    val spark = {
        SparkSession.builder()
          .master("local")
          .getOrCreate()
    }

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    // DATA
    val train_csv = spark.read.format("csv").option("header", value = true).load("./Data/Train.csv")
    val test_csv = spark.read.format("csv").option("header", value = true).load("./Data/Test.csv")
    // val sample_submission_csv = spark.read.format("csv").option("header", true).load("./Data/SampleSubmission.csv")

    // PREPARATION
    val indexer_DATOP = new StringIndexer()
      .setInputCol("DATOP")
      .setOutputCol("DATOP_2")

    val indexer_FLTID = new StringIndexer()
      .setInputCol("FLTID")
      .setOutputCol("FLTID_2")

    val indexer_DEPSTN = new StringIndexer()
      .setInputCol("DEPSTN")
      .setOutputCol("DEPSTN_2")

    val indexer_ARRSTN = new StringIndexer()
      .setInputCol("ARRSTN")
      .setOutputCol("ARRSTN_2")

    val indexer_STD = new StringIndexer()
      .setInputCol("STD")
      .setOutputCol("STD_2")

    val indexer_STA = new StringIndexer()
      .setInputCol("STA")
      .setOutputCol("STA_2")

    val indexer_STATUS = new StringIndexer()
      .setInputCol("STATUS")
      .setOutputCol("STATUS_2")

    val indexer_AC = new StringIndexer()
      .setInputCol("AC")
      .setOutputCol("AC_2")

    val indexer_target = new StringIndexer()
      .setInputCol("target")
      .setOutputCol("label")

    val pipeline_train = new Pipeline()
      .setStages(Array(indexer_DATOP, indexer_FLTID, indexer_DEPSTN, indexer_ARRSTN, indexer_STD, indexer_STA, indexer_STATUS, indexer_AC, indexer_target))

    val pipeline_test = new Pipeline()
      .setStages(Array(indexer_DATOP, indexer_FLTID, indexer_DEPSTN, indexer_ARRSTN, indexer_STD, indexer_STA, indexer_STATUS, indexer_AC))

    val assembler = new VectorAssembler()
      .setInputCols(Array("DATOP_2", "FLTID_2", "DEPSTN_2", "ARRSTN_2", "STD_2", "STA_2", "STATUS_2", "AC_2"))
      .setOutputCol("features")

    val train_pipe = pipeline_train.fit(train_csv).transform(train_csv)
    val test_pipe = pipeline_test.fit(test_csv).transform(test_csv)

    val train_data = assembler.transform(train_pipe)
    val test_data = assembler.transform(test_pipe)

    // TRAINING
    val glr = new GeneralizedLinearRegression()
      .setFamily("gaussian")
      .setLink("identity")
      .setFeaturesCol("features")
      .setLabelCol("label")

    val paramGrid = new ParamGridBuilder()
      .addGrid(glr.regParam, Array(0.0, 1.0, 5.0, 10.0, 50.0))
      .addGrid(glr.maxIter, Array(10, 20, 50, 100, 200))
      .build()

    val cv = new CrossValidator()
      .setEstimator(glr)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)
      .setSeed(42)

    val cvModel = cv.fit(train_data)
    // cvModel.avgMetrics
    // val train_predictions = cvModel.transform(train_data)

    // PREDICTION AND SUBMISSION
    val predictions = cvModel.transform(test_data)
    val submission = predictions.select("ID","prediction")

    var counter = 1
    while(Files.exists(Paths.get("./Submission/" + counter))) {
      counter += 1
    }

    submission.withColumnRenamed("prediction", "target").repartition(1)
      .write.format("com.databricks.spark.csv")
      .option("header", "true")
      .save("./Submission/" + counter + "/submission")
  }
}