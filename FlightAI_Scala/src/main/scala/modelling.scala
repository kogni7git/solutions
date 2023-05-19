import smile.data.DataFrame
import smile.data.formula.Formula
import smile.regression.{cart, lasso, lm, randomForest, ridge}
import smile.validation.CrossValidation


object modelling {
  class Modelling(var cv: Int) {
    private val nFolds: Int = cv

    private def trainOLS(trainDF: DataFrame, testDF: DataFrame): (Array[Double], String, String) = {
      val settings = "\n## Settings:\n* nFolds: " + this.nFolds
      val modelCV = CrossValidation.regression(this.nFolds, Formula.lhs("target"), trainDF, (formula: Formula, data: DataFrame) => lm(formula, data))
      val model = lm(Formula.lhs("target"), trainDF)
      (model.predict(testDF), modelCV.toString, settings)
    }

    private def trainRidge(trainDF: DataFrame, testDF: DataFrame): (Array[Double], String, String) = {
      val lambda = scala.io.StdIn.readLine("lambda? ")
      val settings = "\n## Settings:\n* nFolds: " + this.nFolds + "\n* lambda: " + lambda
      val modelCV = CrossValidation.regression(this.nFolds, Formula.lhs("target"), trainDF, (formula: Formula, data: DataFrame) => ridge(formula, data, lambda.toDouble))
      val model = ridge(Formula.lhs("target"), trainDF, lambda.toDouble)
      (model.predict(testDF), modelCV.toString, settings)
    }

    private def trainLasso(trainDF: DataFrame, testDF: DataFrame): (Array[Double], String, String) = {
      val lambda = scala.io.StdIn.readLine("lambda? ")
      val settings = "\n## Settings:\n* nFolds: " + this.nFolds + "\n* lambda: " + lambda
      val modelCV = CrossValidation.regression(this.nFolds, Formula.lhs("target"), trainDF, (formula: Formula, data: DataFrame) => lasso(formula, data, lambda.toDouble))
      val model = lasso(Formula.lhs("target"), trainDF, lambda.toDouble)
      (model.predict(testDF), modelCV.toString, settings)
    }

    private def trainTree(trainDF: DataFrame, testDF: DataFrame): (Array[Double], String, String) = {
      val maxDepth = scala.io.StdIn.readLine("Depth? ")
      val maxNodes = scala.io.StdIn.readLine("Nodes? ")
      val nodeSize = scala.io.StdIn.readLine("Node size? ")
      val settings = "\n## Settings:\n* nFolds: " + this.nFolds + "\n* maxDepth: " + maxDepth + "\n* maxNodes: " + maxNodes + "\n* nodeSize: " + nodeSize
      val modelCV = CrossValidation.regression(this.nFolds, Formula.lhs("target"), trainDF,
        (formula: Formula, data: DataFrame) => cart(formula, data, maxDepth = maxDepth.toInt, maxNodes = maxNodes.toInt, nodeSize = nodeSize.toInt))
      val model = cart(Formula.lhs("target"), trainDF, maxDepth.toInt, maxNodes.toInt, nodeSize.toInt)
      (model.predict(testDF), modelCV.toString, settings)
    }

    private def trainForest(trainDF: DataFrame, testDF: DataFrame): (Array[Double], String, String) = {
      val ntrees = scala.io.StdIn.readLine("Trees? ")
      val maxDepth = scala.io.StdIn.readLine("Depth? ")
      val maxNodes = scala.io.StdIn.readLine("Nodes? ")
      val nodeSize = scala.io.StdIn.readLine("Node size? ")
      val settings = "\n## Settings:\n* nFolds: " + this.nFolds + "\n* ntrees: " + ntrees + "\n* maxDepth: " + maxDepth + "\n* maxNodes: " + maxNodes + "\n* nodeSize: " + nodeSize
      val modelCV = CrossValidation.regression(this.nFolds, Formula.lhs("target"), trainDF,
        (formula: Formula, data: DataFrame) => randomForest(formula, data, ntrees = ntrees.toInt, maxDepth = maxDepth.toInt, maxNodes = maxNodes.toInt, nodeSize = nodeSize.toInt))
      val model = randomForest(Formula.lhs("target"), trainDF, ntrees = ntrees.toInt, maxDepth = maxDepth.toInt, maxNodes = maxNodes.toInt, nodeSize = nodeSize.toInt)
      (model.predict(testDF), modelCV.toString, settings)
    }

    def trainModel(trainDF: DataFrame, testDF: DataFrame, model: String): (Array[Double], String, String) = {
      if (model == "Ordinary Least Squares") this.trainOLS(trainDF, testDF)
      else if (model == "Ridge Regression") this.trainRidge(trainDF, testDF)
      else if (model == "Lasso Regression") this.trainLasso(trainDF, testDF)
      else if (model == "Regression Tree") this.trainTree(trainDF, testDF)
      else if (model == "Random Forest") this.trainForest(trainDF, testDF)
      else (new Array[Double](9332), " ", " ")
    }
  }
}