import org.apache.commons.csv.CSVParser
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.factory.Nd4j
import smile.clustering.gmeans
import smile.clustering.kmeans
import smile.clustering.xmeans


fun prepareData(parserTrainValues: CSVParser, parserTrainValuesHelper: CSVParser, parserTrainLabels: CSVParser, parserTestValues: CSVParser): Triple<Array<DoubleArray>, Array<DoubleArray>, Array<DoubleArray>> {
    // Label encoding!
    val landSurfaceConditionValues = mutableListOf<String>()
    val foundationTypeValues = mutableListOf<String>()
    val roofTypeValues = mutableListOf<String>()
    val groundFloorTypeValues = mutableListOf<String>()
    val otherFloorTypeValues = mutableListOf<String>()
    val positionValues = mutableListOf<String>()
    val planConfigurationValues = mutableListOf<String>()
    val legalOwnershipStatusValues = mutableListOf<String>()

    for (record in parserTrainValuesHelper) {
        landSurfaceConditionValues.add(record.get("land_surface_condition").toString())
        foundationTypeValues.add(record.get("foundation_type").toString())
        roofTypeValues.add(record.get("roof_type").toString())
        groundFloorTypeValues.add(record.get("ground_floor_type").toString())
        otherFloorTypeValues.add(record.get("other_floor_type").toString())
        positionValues.add(record.get("position").toString())
        planConfigurationValues.add(record.get("plan_configuration").toString())
        legalOwnershipStatusValues.add(record.get("legal_ownership_status").toString())
    }

    val landSurfaceConditionValuesSet = landSurfaceConditionValues.distinct()
    val foundationTypeValuesSet = foundationTypeValues.distinct()
    val roofTypeValuesSet = roofTypeValues.distinct()
    val groundFloorTypeValuesSet = groundFloorTypeValues.distinct()
    val otherFloorTypeValuesSet = otherFloorTypeValues.distinct()
    val positionValuesSet = positionValues.distinct()
    val planConfigurationValuesSet = planConfigurationValues.distinct()
    val legalOwnershipStatusValuesSet = legalOwnershipStatusValues.distinct()

    // prepare train data
    val trainValues = Array(260601) { DoubleArray(38) }
    val trainLabels = Array(260601) { DoubleArray(3) }
    val testValues = Array(86868) { DoubleArray(38) }

    val columns = listOf(
        "geo_level_1_id",
        "geo_level_2_id",
        "geo_level_3_id",
        "count_floors_pre_eq",
        "age",
        "area_percentage",
        "height_percentage",
        "land_surface_condition",
        "foundation_type",
        "roof_type",
        "ground_floor_type",
        "other_floor_type",
        "position",
        "plan_configuration",
        "has_superstructure_adobe_mud",
        "has_superstructure_mud_mortar_stone",
        "has_superstructure_stone_flag",
        "has_superstructure_cement_mortar_stone",
        "has_superstructure_mud_mortar_brick",
        "has_superstructure_cement_mortar_brick",
        "has_superstructure_timber",
        "has_superstructure_bamboo",
        "has_superstructure_rc_non_engineered",
        "has_superstructure_rc_engineered",
        "has_superstructure_other",
        "legal_ownership_status",
        "count_families",
        "has_secondary_use",
        "has_secondary_use_agriculture",
        "has_secondary_use_hotel",
        "has_secondary_use_rental",
        "has_secondary_use_institution",
        "has_secondary_use_school",
        "has_secondary_use_industry",
        "has_secondary_use_health_post",
        "has_secondary_use_gov_office",
        "has_secondary_use_use_police",
        "has_secondary_use_other"
    )

    var row = 0
    for (record in parserTrainValues) {
        for ((col, column) in columns.withIndex()) {
            var trainValue: String
            when (column) {
                "land_surface_condition" -> {
                    trainValue = landSurfaceConditionValuesSet.indexOf(record.get(column).toString()).toString()
                }
                "foundation_type" -> {
                    trainValue = foundationTypeValuesSet.indexOf(record.get(column).toString()).toString()
                }
                "roof_type" -> {
                    trainValue = roofTypeValuesSet.indexOf(record.get(column).toString()).toString()
                }
                "ground_floor_type" -> {
                    trainValue = groundFloorTypeValuesSet.indexOf(record.get(column).toString()).toString()
                }
                "other_floor_type" -> {
                    trainValue = otherFloorTypeValuesSet.indexOf(record.get(column).toString()).toString()
                }
                "position" -> {
                    trainValue = positionValuesSet.indexOf(record.get(column).toString()).toString()
                }
                "plan_configuration" -> {
                    trainValue = planConfigurationValuesSet.indexOf(record.get(column).toString()).toString()
                }
                "legal_ownership_status" -> {
                    trainValue = legalOwnershipStatusValuesSet.indexOf(record.get(column).toString()).toString()
                }
                else -> {
                    trainValue = record.get(column).toString()
                }
            }
            trainValues[row][col] = trainValue.toDouble()
        }
        row++
    }

    row = 0
    for (record in parserTrainLabels) {
        when (record.get("damage_grade").toString().toInt()) {
            1 -> {
                trainLabels[row][0] = 1.0
                trainLabels[row][1] = 0.0
                trainLabels[row][2] = 0.0
            }
            2 -> {
                trainLabels[row][0] = 0.0
                trainLabels[row][1] = 1.0
                trainLabels[row][2] = 0.0
            }
            3 -> {
                trainLabels[row][0] = 0.0
                trainLabels[row][1] = 0.0
                trainLabels[row][2] = 1.0
            }
        }
        row++
    }

    // prepare test data
    row = 0
    for (record in parserTestValues) {
        for ((col, column) in columns.withIndex()) {
            var testValue: String
            when (column) {
                "land_surface_condition" -> {
                    testValue = landSurfaceConditionValuesSet.indexOf(record.get(column).toString()).toString()
                }
                "foundation_type" -> {
                    testValue = foundationTypeValuesSet.indexOf(record.get(column).toString()).toString()
                }
                "roof_type" -> {
                    testValue = roofTypeValuesSet.indexOf(record.get(column).toString()).toString()
                }
                "ground_floor_type" -> {
                    testValue = groundFloorTypeValuesSet.indexOf(record.get(column).toString()).toString()
                }
                "other_floor_type" -> {
                    testValue = otherFloorTypeValuesSet.indexOf(record.get(column).toString()).toString()
                }
                "position" -> {
                    testValue = positionValuesSet.indexOf(record.get(column).toString()).toString()
                }
                "plan_configuration" -> {
                    testValue = planConfigurationValuesSet.indexOf(record.get(column).toString()).toString()
                }
                "legal_ownership_status" -> {
                    testValue = legalOwnershipStatusValuesSet.indexOf(record.get(column).toString()).toString()
                }
                else -> {
                    testValue = record.get(column).toString()
                }
            }
            testValues[row][col] = testValue.toDouble()
        }
        row++
    }

    return Triple(trainValues, trainLabels, testValues)
}

fun clusterData(trainValues: Array<DoubleArray>, testValues: Array<DoubleArray>, algorithm: String): Pair<Array<DoubleArray>, Array<DoubleArray>> {

    val trainValuesForClustering= Array(260601) { DoubleArray(12) }
    val testValuesForClustering = Array(86868) { DoubleArray(12) }

    val columns= arrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

    for (row in 0..260600) {
        for (col in columns) {
            trainValuesForClustering[row][col-1] = trainValues[row][col]
        }
    }
    for (row in 0..86867) {
        for (col in columns) {
            testValuesForClustering[row][col-1] = testValues[row][col]
        }
    }

    var centroidsTrain= Array(10) { DoubleArray(12) }
    var yTrain = IntArray(260601)

    var centroidsTest= Array(10) { DoubleArray(12) }
    var yTest = IntArray(86868)

    if (algorithm == "0") {
        val clusterTrain = gmeans(trainValuesForClustering, 10)
        val clusterTest = gmeans(testValuesForClustering, 10)

        centroidsTrain = clusterTrain.centroids
        yTrain = clusterTrain.y
        centroidsTest = clusterTest.centroids
        yTest = clusterTest.y
    } else if (algorithm == "1") {
        val clusterTrain = kmeans(trainValuesForClustering, 10)
        val clusterTest = kmeans(testValuesForClustering, 10)

        centroidsTrain = clusterTrain.centroids
        yTrain = clusterTrain.y
        centroidsTest = clusterTest.centroids
        yTest = clusterTest.y
    } else if (algorithm == "2") {
        val clusterTrain = xmeans(trainValuesForClustering, 10)
        val clusterTest = xmeans(testValuesForClustering, 10)

        centroidsTrain = clusterTrain.centroids
        yTrain = clusterTrain.y
        centroidsTest = clusterTest.centroids
        yTest = clusterTest.y
    }

    for (row in 0..260600) {
        for ((col2, col) in columns.withIndex()) {
            trainValues[row][col] = centroidsTrain[yTrain[row]][col2]
        }
    }
    for (row in 0..86867) {
        for ((col2, col) in columns.withIndex()) {
            testValues[row][col] = centroidsTest[yTest[row]][col2]
        }

    }

    return Pair(trainValues, testValues)
}


fun createData(trainValues: Array<DoubleArray>, trainLabels: Array<DoubleArray>, testValues: Array<DoubleArray>, seed: Long): Triple<DataSet, DataSet, DataSet> {
    val features = Nd4j.create(trainValues)
    val labels = Nd4j.create(trainLabels)
    val testFeatures = Nd4j.create(testValues)

    val data = DataSet(features, labels)
    data.shuffle(seed)

    // build evaluation set
    val splittedData = data.splitTestAndTrain(0.8)
    val trainingData = splittedData.train
    val evalData = splittedData.test

    // normalize data
    val normalizer: DataNormalization = NormalizerStandardize()
    normalizer.fit(trainingData)
    normalizer.transform(trainingData)
    normalizer.transform(evalData)
    normalizer.transform(testFeatures)

    val testData = DataSet(testFeatures, Nd4j.zeros(86868, 1))

    return Triple(trainingData, evalData, testData)
}


fun createSubmission(parserSubmissionFormat: CSVParser,  predictions: MutableList<Int> ): String {
    val building_id = mutableListOf<Int>()
    for (record in parserSubmissionFormat) {
        building_id.add(record.get("building_id").toString().toInt())
    }
    var submission: String = "building_id,damage_grade\n"

    var counter = 0
    for (i in building_id) {
        submission += i.toString() + "," + predictions[counter] + "\n"
        counter++
    }
    return submission
}