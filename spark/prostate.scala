//$SPARK_HOME/bin/spark-shell --packages com.databricks:spark-csv_2.11:1.4.0
// example from https://spark.apache.org/docs/1.4.1/ml-ensembles.html
// pmml is from here https://spark.apache.org/docs/latest/mllib-pmml-model-export.html


import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils




val data = sqlContext.read
  .format("com.databricks.spark.csv")
  .option("header", "false") // Use first line of all files as header
  .option("inferSchema", "true") // Automatically infer data types
  .load("prostate")

val namedData = data.toDF("id","label","age","race","rectalExam","detectCapsular","prostateSpecAntigen","tumorVol","gleasonScore")

val featureNames = Array("age","race","rectalExam","detectCapsular","prostateSpecAntigen","tumorVol","gleasonScore")

val assembler = new VectorAssembler()
  .setInputCols(featureNames)
  .setOutputCol("features")  
  
val transformed = assembler.transform(namedData)


  
val Array(train, test) = transformed.selectExpr("cast(label as double) label","features").randomSplit(Array(0.7, 0.3))

val training = train.rdd.map(row => LabeledPoint(row.getAs[Double]("label"), row.getAs[org.apache.spark.mllib.linalg.Vector]("features")))
val testing = test.rdd.map(row => LabeledPoint(row.getAs[Double]("label"), row.getAs[org.apache.spark.mllib.linalg.Vector]("features")))


val model = new LogisticRegressionWithLBFGS().run(training)


lrModel.toPMML("prostate.xml")


// Compute raw scores on the test set.
val predictionAndLabels = testing.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}

val metrics = new MulticlassMetrics(predictionAndLabels)
val accuracy = metrics.accuracy
println(s"Accuracy = $accuracy")
