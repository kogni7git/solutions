import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.evaluation.EvaluationAveraging
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import kotlin.math.round


fun makeModel(seed: Long, inSize: Int, hiddenSize: Int, dropOut: Double, learningRate: Double): MultiLayerNetwork {
    val configuration = NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(Adam(learningRate))
                .list()
                .layer(DenseLayer.Builder().nIn(inSize).nOut(hiddenSize).activation(Activation.RELU).build())
                .layer(DenseLayer.Builder().nIn(hiddenSize).nOut(hiddenSize).dropOut(dropOut).activation(Activation.RELU).build())
                .layer(DenseLayer.Builder().nIn(hiddenSize).nOut(hiddenSize).dropOut(dropOut).activation(Activation.RELU).build())
                .layer(OutputLayer.Builder(LossFunctions.LossFunction.XENT).nIn(hiddenSize).nOut(3).activation(Activation.SIGMOID).build())
                .build()
    return MultiLayerNetwork(configuration)
}


fun trainModel(epochs: Int, batchSize: Int, model: MultiLayerNetwork, trainingData: DataSet, evalData: DataSet, testData: DataSet): Pair<MutableList<Int>, String> {
    val eval = Evaluation()
    var evalOutput: INDArray
    var results = " "

    // Train!
    val trainingDataIterator = IteratorDataSetIterator(trainingData.iterator(), batchSize).next()

    for (e in 0 until epochs) {
        model.fit(trainingDataIterator)
        evalOutput = model.output(evalData.features)
        eval.eval(evalData.labels, evalOutput)
        val result = "\n* Epoch: " + e.toString() + "; micro F1: " + (round(eval.f1(EvaluationAveraging.Micro) * 1000) / 1000).toString()
        print(result)
        results += result
    }

    val testDataIterator = IteratorDataSetIterator(testData.iterator(), 1)
    val predictions = mutableListOf<Int>()

    while (testDataIterator.hasNext()) {
        val next = testDataIterator.next()
        val prediction: INDArray = model.output(next.features)
        predictions.add(prediction.getRow(0).argMax().add(1).toString().toDouble().toString()[0].digitToInt())
    }
    return Pair(predictions, results)
}