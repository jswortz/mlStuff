import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}
import org.jpmml.sparkml._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.ml.feature.Bucketizer
import org.apache.spark.sql.functions.udf
import org.apache.spark.mllib.linalg.{
  Vector => OldVector, Vectors => OldVectors}

import org.apache.spark.ml.linalg.{
   Vector => NewVector,
   DenseVector => NewDenseVector,
   SparseVector => NewSparseVector
}
import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier}  
import org.apache.spark.mllib.evaluation.MulticlassMetrics



val sqlContext = new HiveContext(sc)


val rawData = sqlContext.sql("select cast(billing as double) billing, currentbalance, accounttype from analysis.comcastBillingFinal4 limit 1000")


val string_indexers = new StringIndexer()
     .setInputCol("accounttype")
     .setOutputCol("accounttypeIDX")

val assembler_test = new VectorAssembler()
  .setInputCols(Array("currentbalance"))
  .setOutputCol("features")
  
val featureIndexer =  new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("featuresIndexed")
  .setMaxCategories(13)
  
val targetIndexer = new StringIndexer()
  .setInputCol("billing")
  .setOutputCol("label")
  
val rf = new RandomForestClassifier()
   .setLabelCol("billing")
   .setFeaturesCol("features")
   .setNumTrees(40)
   .setFeatureSubsetStrategy("onethird")
   .setMaxDepth(8)
   .setMaxBins(32)
   .setImpurity("entropy") 
   .setRawPredictionCol("prob")
   
val stages = Array(assembler_test, rf)
   
val pipeline =  new Pipeline().setStages(stages)

val model_test = pipeline.fit(rawData)

model_test.save("test12345")
val pmml = ConverterUtil.toPMML(rawData.schema, model_test)

JAXBUtil.marshalPMML(pmml, new StreamResult(System.out));