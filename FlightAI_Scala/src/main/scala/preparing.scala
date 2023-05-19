import smile.data.{DataFrame, pimpDataFrame}


object preparing {
  def prepareData(trainCSV: DataFrame, testCSV: DataFrame): (DataFrame, DataFrame) = {
    // categorical data: DATOP, FLTID, DEPSTN, ARRSTN, STATUS, AC
    val datopValues = trainCSV.column("DATOP").toStringArray().toList.distinct
    val fltidValues = trainCSV.column("FLTID").toStringArray().toList.distinct
    val depstnValues = trainCSV.column("DEPSTN").toStringArray().toList.distinct
    val arrstnValues = trainCSV.column("ARRSTN").toStringArray().toList.distinct
    val statusValues = trainCSV.column("STATUS").toStringArray().toList.distinct
    val acValues = trainCSV.column("AC").toStringArray().toList.distinct

    val datop = trainCSV.select("DATOP").map { row => datopValues.indexOf(row.getString(0)) }.toArray
    val fltid = trainCSV.select("FLTID").map { row => fltidValues.indexOf(row.getString(0)) }.toArray
    val depstn = trainCSV.select("DEPSTN").map { row => depstnValues.indexOf(row.getString(0)) }.toArray
    val arrstn = trainCSV.select("ARRSTN").map { row => arrstnValues.indexOf(row.getString(0)) }.toArray
    val status = trainCSV.select("STATUS").map { row => statusValues.indexOf(row.getString(0)) }.toArray
    val ac = trainCSV.select("AC").map { row => acValues.indexOf(row.getString(0)) }.toArray

    // Feature engineering: to minutes
    val std = trainCSV.select("STD").map { row => row.getString(0).subSequence(11, 16).toString.subSequence(0, 2).toString.toInt * 60 +
      row.getString(0).subSequence(11, 16).toString.subSequence(3, 5).toString.toInt }.toArray
    val sta = trainCSV.select("STA").map { row => row.getString(0).subSequence(11, 16).toString.subSequence(0, 2).toString.toInt * 60 +
      row.getString(0).subSequence(11, 16).toString.subSequence(3, 5).toString.toInt }.toArray

    val duration = Array.ofDim[Double](sta.length, 1)
    for (i <- sta.indices) {
      duration(i)(0) = sta(i) - std(i)
    }

    val target = trainCSV.select("target").map { row => row.getString(0).toDouble }.toArray

    val data = Array.ofDim[Double](target.length, 10)
    for (i <- datop.indices) data(i)(0) = datop(i)
    for (i <- fltid.indices) data(i)(1) = fltid(i)
    for (i <- depstn.indices) data(i)(2) = depstn(i)
    for (i <- arrstn.indices) data(i)(3) = arrstn(i)
    for (i <- std.indices) data(i)(4) = std(i)
    for (i <- sta.indices) data(i)(5) = sta(i)
    for (i <- status.indices) data(i)(6) = status(i)
    for (i <- duration.indices) data(i)(7) = duration(i)(0)
    for (i <- ac.indices) data(i)(8) = ac(i)
    for (i <- target.indices) data(i)(9) = target(i)

    val trainDF = DataFrame.of(data, "DATOP", "FLTID", "DEPSTN", "ARRSTN", "STD", "STA", "STATUS", "DURATION", "AC", "target")

    val datopTest = testCSV.select("DATOP").map { row => datopValues.indexOf(row.getString(0)) }.toArray
    val fltidTest = testCSV.select("FLTID").map { row => fltidValues.indexOf(row.getString(0)) }.toArray
    val depstnTest = testCSV.select("DEPSTN").map { row => depstnValues.indexOf(row.getString(0)) }.toArray
    val arrstnTest = testCSV.select("ARRSTN").map { row => arrstnValues.indexOf(row.getString(0)) }.toArray
    val statusTest = testCSV.select("STATUS").map { row => statusValues.indexOf(row.getString(0)) }.toArray
    val acTest = testCSV.select("AC").map { row => acValues.indexOf(row.getString(0)) }.toArray

    // Feature engineering: to minutes
    val stdTest = testCSV.select("STD").map { row => row.getString(0).subSequence(11, 16).toString.subSequence(0, 2).toString.toInt * 60 +
      row.getString(0).subSequence(11, 16).toString.subSequence(3, 5).toString.toInt }.toArray
    val staTest = testCSV.select("STA").map { row => row.getString(0).subSequence(11, 16).toString.subSequence(0, 2).toString.toInt * 60 +
      row.getString(0).subSequence(11, 16).toString.subSequence(3, 5).toString.toInt }.toArray

    val durationTest = Array.ofDim[Double](staTest.length, 1)
    for (i <- staTest.indices) {
      durationTest(i)(0) = staTest(i) - stdTest(i)
    }

    val dataTest = Array.ofDim[Double](ac.length, 10)
    for (i <- datopTest.indices) dataTest(i)(0) = datopTest(i)
    for (i <- fltidTest.indices) dataTest(i)(1) = fltidTest(i)
    for (i <- depstnTest.indices) dataTest(i)(2) = depstnTest(i)
    for (i <- arrstnTest.indices) dataTest(i)(3) = arrstnTest(i)
    for (i <- stdTest.indices) dataTest(i)(4) = stdTest(i)
    for (i <- staTest.indices) dataTest(i)(5) = staTest(i)
    for (i <- statusTest.indices) dataTest(i)(6) = statusTest(i)
    for (i <- durationTest.indices) dataTest(i)(7) = durationTest(i)(0)
    for (i <- acTest.indices) dataTest(i)(8) = acTest(i)

    val testDF = DataFrame.of(dataTest, "DATOP", "FLTID", "DEPSTN", "ARRSTN", "STD", "STA", "STATUS", "DURATION", "AC", "target")

    (trainDF, testDF)
  }

  def prepareSubmission(sampleSubmissionCSV: DataFrame, prediction: Array[Double]): String = {
    val ID = sampleSubmissionCSV.column("ID").toStringArray()
    var submission = "ID,target\n"
    var counter = 0
    for (i <- ID) {
      submission = submission.concat(i).concat(",").concat(prediction(counter).toString).concat("\n")
      counter += 1
    }
    submission
  }
}