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
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.ml.feature.StandardScaler

import org.apache.spark.ml.linalg.{
   Vector => NewVector,
   DenseVector => NewDenseVector,
   SparseVector => NewSparseVector
}
import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier}  
import org.apache.spark.mllib.evaluation.MulticlassMetrics






val sqlContext = new HiveContext(sc)

val selectStatement = """currentbalance, daysdelinquent, previousdueamount, previouspaymentamount, statmentbalance, billingpin, cardpaymentrestricted, 
bankpaymentrestricted, existingrecurringpaymentonaccount, balancedue, pastduebalance, pendingpaymentamount, projectedbalancedue, recentpaymentamount, statementbalance, 
billing, due_diff, prev_pay_diff, bal_due_diff, past_due_diff, rec_pay_diff, promise_diff, nonpaydiscoii, delinq_days, p_p_payment, p_p_p_payment, p_p_p_p_payment, 
p_p_p_p_p_payment, csgamdocs, outageflag, plannedoutages, boradcaststatus, broadcastfunction, accttype, refacoralloc, boradcaststatusii, anilearning, anilearningii, 
seasonal, closedappt, cancelledappt, install, nonpaydisco, accountstatus, addresstype, accounttype, phonetype, ani2accttype, dwell_code, dta, tpv_outstanding, source, 
accountsource, searchstatus """ 
//reduced list of variables based off of variable importance
val featureStringFields2 = Array( 
//"csgamdocs", "outageflag", 
"plannedoutages", "boradcaststatus", 
"broadcastfunction", "accttype", "refacoralloc", "boradcaststatusii", "anilearning",
"anilearningii", "seasonal", "closedappt", "cancelledappt", "install", "nonpaydisco", "accountstatus", "addresstype", "accounttype", "phonetype", "ani2accttype", "dwell_code", "dta",
"tpv_outstanding", "source", "accountsource", "searchstatus", "cardpaymentrestricted", "bankpaymentrestricted", "existingrecurringpaymentonaccount"
)
//reduced continuous variables based off variable importance
val contFeatures2 = Array("nonpaydiscoii", "delinq_days", "due_diff", "prev_pay_diff", "bal_due_diff", "past_due_diff",
"rec_pay_diff", "promise_diff", "currentbalance", "daysdelinquent", "previousdueamount", "previouspaymentamount", "statmentbalance","p_p_payment","p_p_p_p_p_payment")

val balanceFeat = Array("nonpaydiscoii", "currentbalance", "previousdueamount", "previouspaymentamount", "statmentbalance")

val timeFeat = Array("delinq_days", "bal_due_diff", "past_due_diff", "rec_pay_diff", "promise_diff", "daysdelinquent")

val target = "billing"
 
val rawData = sqlContext.sql("select * from analysis.comcastBillingFinal4")
   
   //had to get rid of "p_billing" - was creating target leak


val typesRight = rawData.withColumn("nonpaydiscoii",rawData("nonpaydiscoii").cast("double")).withColumn("due_diff",rawData("due_diff").cast("double")).withColumn("delinq_days",rawData("delinq_days").cast("double")).withColumn("billing",rawData("billing").cast("double")).withColumn("prev_pay_diff",rawData("prev_pay_diff").cast("double")).withColumn("bal_due_diff",rawData("bal_due_diff").cast("double")).withColumn("past_due_diff",rawData("past_due_diff").cast("double")).withColumn("rec_pay_diff",rawData("rec_pay_diff").cast("double")).withColumn("promise_diff",rawData("promise_diff").cast("double")).withColumn("daysdelinquent",rawData("daysdelinquent").cast("double"))
//.withColumn("delinq_days",rawData("delinq_days").cast("double"))
//.withColumn("prev_pay_diff",rawData("prev_pay_diff").cast("double"))



val impute1 = typesRight.na.fill(0).drop(typesRight.col("p_p_p_p_p_payment")).drop(typesRight.col("p_p_payment")



// was used to try this in Python to try local modeling
//impute1.rdd.saveAsTextFile("/user/jswortz/comcastBillPay")

val Array(trainingData, testData) = impute1.randomSplit(Array(0.8, 0.2))

//val impute1 = impute.cache()

val string_indexers: Array[org.apache.spark.ml.PipelineStage] = featureStringFields2.map(
   cname => new StringIndexer()
     .setInputCol(cname)
     .setOutputCol(s"${cname}_index")
)

val encoders = featureStringFields2.map(
cname => new OneHotEncoder()
  .setInputCol(s"${cname}_index")
  .setOutputCol(s"${cname}_enc")
 )

val splits = Array(Double.NegativeInfinity, 0, 1, 50, 100, 150, 200, 250, Double.PositiveInfinity)

val timeSplits = Array(Double.NegativeInfinity, -10,-5,0,5,10,20,30,40,50, Double.PositiveInfinity)


val bucketizer1 = balanceFeat.map(
   cname => new Bucketizer()
  .setInputCol(cname)
  .setOutputCol(s"${cname}_buk")
  .setSplits(splits)
  )
  
val bucketizer2 = timeFeat.map(
   cname => new Bucketizer()
  .setInputCol(cname)
  .setOutputCol(s"${cname}_buk")
  .setSplits(timeSplits)
  )

val lr = new LogisticRegression()
  .setMaxIter(100)
  .setRegParam(0.01)
  .setElasticNetParam(1.0)
//val assembler_test2 = new VectorAssembler()
//  .setInputCols(featureStringFields2.map(cname => s"${cname}_index") ++ balanceFeat.map(cname => s"${cname}_buk") ++ timeFeat.map(cname => s"${cname}_buk"))
//  .setOutputCol("features")

val assembler_test = new VectorAssembler()
  .setInputCols(featureStringFields2.map(cname => s"${cname}_enc") ++ contFeatures2)
  .setOutputCol("featurez")
  
val scaler = new StandardScaler()
  .setInputCol("featurez")
  .setOutputCol("features")

  //featureStringFields2.map(cname => s"${cname}_index") ++ 
  //contFeatures2
   
val featureIndexer =  new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("featuresIndexed")
  .setMaxCategories(13)

val targetIndexer = new StringIndexer()
  .setInputCol("billing")
  .setOutputCol("label")
  
val rf = new RandomForestClassifier()
   .setLabelCol("label")
   .setFeaturesCol("features")
   .setNumTrees(40)
   .setFeatureSubsetStrategy("onethird")
   .setMaxDepth(8)
   .setMaxBins(32)
   .setImpurity("entropy") 
   .setRawPredictionCol("prob")
 

val layers = Array[Int](23, 5, 4, 3)

val crazyLayers = Array[Int](53,53, 2)

 
val trainer = new MultilayerPerceptronClassifier()
  .setLayers(crazyLayers)
  .setBlockSize(128)
  .setSeed(1234L)
  .setMaxIter(500)
  .setTol(1e-5)
   
val stages = string_indexers ++ encoders ++ Array(assembler_test, targetIndexer, scaler, lr)
   
val pipeline =  new Pipeline().setStages(stages)

val model_test = pipeline.fit(trainingData)

val layerGrid = Array(Array(23, 15, 7, 8),Array(23, 15, 12, 5,3,2),Array(23, 8, 4, 2),Array(23, 20, 15, 10),Array(23, 15, 10, 8,6,4,2))

val paramGrid = new ParamGridBuilder()
  .addGrid(trainer.layers, layerGrid)
  .addGrid(trainer.maxIter,Array(1000,10000,50000))
  .addGrid(trainer.tol,Array(1e-6,1e-7,1e-8,1e-10))
  .build()
  
val cv = new CrossValidator()
   .setEstimator(pipeline)
   .setEvaluator(new MulticlassClassificationEvaluator)
   .setEstimatorParamMaps(paramGrid)
   .setNumFolds(4)
   
val model_test2 = cv.fit(trainingData)

//val pipeline2 = new Pipeline().setStages(string_indexers)

//new attempt with package...	
//val model_test2 = pipeline.fit(trainingData)

//val pmmlBytes = org.jpmml.sparkml.ConverterUtil.toPMMLByteArray(trainingData.schema, test)
//model_test2.write.overwrite().save("/home/jswortz/fittedModel1.11.2016")

// Make predictions.

def modelPerformance(model_test2: org.apache.spark.ml.PipelineModel): Unit = {

val predictions = model_test2.transform(testData)

//val schema2 = trainingData.schema

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("weightedPrecision")
val precision = evaluator.evaluate(predictions)
val evaluator2 = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("weightedRecall")
val recall = evaluator2.evaluate(predictions)
val evaluator3 = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator3.evaluate(predictions)
val predictionsAndLabels = predictions.select("prediction","label")
val metrics = new MulticlassMetrics( predictionsAndLabels.rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))
println("Precision = " + precision)
println("Recall = " + recall)
println("Accuracy = " + accuracy)
println("Confusion matrix:")
println(metrics.confusionMatrix)
}

// below is how we found variable importances using random forest
val rfModel = model_test.stages(10).asInstanceOf[RandomForestClassificationModel]

val importances = rfModel.featureImportances


val fields = featureStringFields2 ++ contFeatures2
//writing the results to a file
val fImp = for (i <- 0 to fields.size) yield (fields(i) +"|"+ importances(i) +"\n")
import java.io._
val pw = new PrintWriter(new File("importances.txt" ))
pw.write(fImp.toString)
pw.close

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier


val layers = Array[Int](12, 5, 4, 2)
// create the trainer and set its parameters
val trainer = new MultilayerPerceptronClassifier()
  .setLayers(layers)
  .setBlockSize(128)
  .setSeed(1234L)
  .setMaxIter(100)
// train the model

val stages2 = string_indexers ++ Array(assembler_test, featureIndexer, targetIndexer, trainer)
val pipeline2 =  new Pipeline().setStages(stages)
val model_test = pipeline2.fit(trainingData)

// export HADOOP_CONF_DIR=/etc/hadoop/conf
//SUPER SIMPLE EXAMPLE FOR PMML

val string_indexer = new StringIndexer()
     .setInputCol("nonpaydisco")
     .setOutputCol("nonpaydisco_index")

val assembler = new VectorAssembler()
  .setInputCols(Array("nonpaydiscoii"))
  .setOutputCol("features")
  
val lr = new LogisticRegression()
  .setMaxIter(10)
  .setRegParam(0.3)
  .setElasticNetParam(0.8)

val stages2 = Array(assembler, targetIndexer, lr)
   
val pipeline2 =  new Pipeline().setStages(stages2)

val model_test = pipeline2.fit(trainingData)

val pmml = ConverterUtil.toPMML(trainingData.schema, model_test)

//get the 1 - level probabilyty


val getP = udf((x: OldVector) => x match {case s => s(1)})

val toOld = udf((v: NewVector) => v match {
  case sv: NewSparseVector => OldVectors.sparse(sv.size, sv.indices, sv.values)
  case dv: NewDenseVector => OldVectors.dense(dv.values)
})
val predictions = model_test.transform(testData)

val data = predictions.withColumn("probability", getP(toOld($"probability")))

data.registerTempTable("decile1")

val dec = sqlContext.sql("select decile, sum(label) as n_act, count(1) as n_records ,min(probability) as minProb from (select *, ntile(20) over (order by probability) as decile from decile1) a group by decile")


//cvModel.getEstimatorParamMaps
//           .zip(cvModel.avgMetrics)
//           .maxBy(_._2)
//           ._1

//val x = PipelineModel.load("rf1ComcastPaymentFitted")

