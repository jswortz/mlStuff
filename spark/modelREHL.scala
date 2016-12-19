import org.apache.spark.mllib._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}

import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}




//note running this with the databricks csv importer $SPARK_HOME/bin/spark-shell --packages com.databricks:spark-csv_2.11:1.4.0

val df = sqlContext.read
  .format("com.databricks.spark.csv")
  .option("header", "false") // Use first line of all files as header
  .option("inferSchema", "true") // Automatically infer data types
  .load("train_RCV_nHead.csv")
  
val test = sqlContext.read
  .format("com.databricks.spark.csv")
  .option("header", "false") // Use first line of all files as header
  .option("inferSchema", "true") // Automatically infer data types
  .load("test_RCV_nHead.csv")
  
// here's the saved version to parquet
//df.write.parquet("trainREHL.parquet") 
//test.write.parquet("testREHL.parquet")
// val test =  sqlContext.read.parquet("testREHL.parquet")  
// val df = sqlContext.read.parquet("trainREHL.parquet")  
  

// dropping the id column
val df2 = df.drop("c0")

val feat = Array("C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", 
  "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", 
  "C25", "C26", "C27", "C28", "C29", "C30", "C31", "C32", "C33", "C34", "C35", "C36", "C37", 
  "C38", "C39", "C40", "C41", "C42", "C43", "C44", "C45", "C46", "C47", "C48", "C49", "C50", 
  "C51", "C52", "C53", "C54", "C55", "C56", "C57", "C58", "C59", "C60", "C61", "C62", "C63", 
  "C64", "C65", "C66", "C67", "C68", "C69", "C70", "C71", "C72", "C73", "C74", "C75", "C76", "C77", 
  "C78", "C79", "C80", "C81", "C82", "C83", "C84", "C85", "C86", "C87", "C88", "C89", "C90", "C91", 
  "C92", "C93", "C94", "C95", "C96", "C97", "C98", "C99", "C100", "C101", "C102", "C103", "C104", 
  "C105", "C106", "C107", "C108", "C109", "C110", "C111", "C112", "C113", "C114", "C115", "C116", "C117", 
  "C118", "C119", "C120", "C121", "C122", "C123", "C124", "C125", "C126", "C127", "C128", "C129", "C130", 
  "C131", "C132", "C133", "C134", "C135", "C136", "C137", "C138", "C139", "C140", "C141", "C142", "C143", 
  "C144", "C145", "C146", "C147", "C148", "C149", "C150", "C151", "C152", "C153", "C154", "C155", 
  "C156", "C157", "C158", "C159", "C160", "C161", "C162", "C163", "C164", "C165", "C166", "C167", "C168", 
  "C169", "C170", "C171", "C172", "C173", "C174", "C175", "C176", "C177", "C178", "C179", "C180", "C181", 
  "C182", "C183", "C184", "C185", "C186", "C187", "C188", "C189", "C190", "C191", "C192", "C193", "C194", 
  "C195", "C196", "C197", "C198", "C199", "C200", "C201", "C202", "C203", "C204", "C205", "C206", "C207", 
  "C208", "C209", "C210", "C211", "C212", "C213", "C214", "C215", "C216", "C217", "C218", "C219", "C220", "C221",
  "C222", "C223", "C224", "C225", "C226", "C227", "C228", "C229", "C230", "C231", "C232", "C233", "C234", 
  "C235", "C236", "C237", "C238", "C239", "C240", "C241", "C242", "C243", "C244", "C245", "C246", "C247", 
  "C248", "C249", "C250", "C251", "C252", "C253", "C254", "C255", "C256", "C257", "C258", "C259", "C260", 
  "C261", "C262", "C263", "C264", "C265", "C266", "C267", "C268", "C269", "C270", "C271", "C272", "C273", 
  "C274", "C275", "C276", "C277", "C278", "C279", "C280", "C281", "C282", "C283", "C284", "C285", "C286", 
  "C287", "C288", "C289", "C290", "C291", "C292", "C293", "C294", "C295", "C296", "C297", "C298", "C299", 
  "C300", "C301", "C302", "C303", "C304", "C305", "C306", "C307", "C308", "C309", "C310", "C311", "C312", "C313", 
  "C314", "C315", "C316", "C317", "C318", "C319", "C320", "C321", "C322", "C323", "C324", "C325", "C326", 
  "C327", "C328", "C329", "C330", "C331", "C332", "C333", "C334", "C335", "C336", "C337", "C338", "C339", 
  "C340", "C341", "C342", "C343", "C344", "C345", "C346", "C347", "C348", "C349", "C350", "C351", "C352", "C353", "C354", "C355" )
  

  
val assembler = new VectorAssembler()
  .setInputCols(feat)
  .setOutputCol("features")

val transformed = assembler.transform(df2)

val data = transformed.select(col("C1").alias("label"), col("features"))
data.registerTempTable("data")
val data2 = sqlContext.sql("select cast(label as double) as label, features from data")

  
val lr = new LogisticRegression()
  .setMaxIter(10)
  .setRegParam(0.3)
  .setElasticNetParam(0.8)
  
val pipeline = new Pipeline().setStages(Array(lr))
  
val paramGrid = new ParamGridBuilder()
  .addGrid(lr.regParam, Array(.01, .1, .05, .001))
  .addGrid(lr.maxIter, Array(10, 100, 500))
  .addGrid(lr.elasticNetParam, Array(.01,.02,.5,.8,.9,.99))
  .build

val cv = new CrossValidator()
  .setNumFolds(3)
  .setEstimator(pipeline)
  .setEstimatorParamMaps(paramGrid)
  .setEvaluator(new BinaryClassificationEvaluator)
  
val cvModel = cv.fit(data2)

 
// RANDOM FOREST  


val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")
  .fit(data2)
  
val rf = new RandomForestClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("features")
  .setNumTrees(10)
  
val pipelineRF = new Pipeline().setStages(Array(labelIndexer, rf))

val paramGridRF = new ParamGridBuilder()
  .addGrid(rf.numTrees, Array(100, 250, 500))
  .addGrid(rf.maxDepth, Array(2, 3, 4))
  .addGrid(rf.maxBins , Array(5, 16, 32))
  .build  
  
val cvRF = new CrossValidator()
  .setNumFolds(3)
  .setEstimator(pipelineRF)
  .setEstimatorParamMaps(paramGridRF)
  .setEvaluator(new BinaryClassificationEvaluator)

val cvModel = cvRF.fit(data2)

cvModel.avgMetrics
//rf results
//scala> cvModel.avgMetrics.max
//res2: Double = 0.9186488791763614


// res0: Array[Double] = Array(0.9305736652418083, 0.911030325861842, 0.5, 0.5
// val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(.01, .1, .05, .001)).build
// res1: Array[Double] = Array(0.9305736652418087, 0.911030325861842, 0.9208930373328335, 0.9336758044911417)\
// WINNER 
// scala> cvModel.avgMetrics.max
// res3: Double = 0.9378709752764747



val predictedDf = cvModel.bestModel.transform(testDf)



val boostingStrategy =
  BoostingStrategy.defaultParams("Classification")
boostingStrategy.numIterations = 500
boostingStrategy.treeStrategy.maxDepth = 5
boostingStrategy.treeStrategy.maxBins = 32

val data3 = data2.select(col("label"), col("features")).rdd
  .map(row => LabeledPoint(row.getAs[Double]("label"), row.getAs[org.apache.spark.mllib.linalg.Vector]("features")))

val splits = data3.randomSplit(Array(0.7, 0.3))

val (trainData, testData) = (splits(0), splits(1))

val model = GradientBoostedTrees.train(trainData, boostingStrategy)


val predictionAndLabels = testData.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}

val metrics = new BinaryClassificationMetrics(predictionAndLabels,100)

val testroc= metrics.roc()

val auROC = metrics.areaUnderROC
println("Area under ROC = " + auROC)
//Area under ROC = 0.8992328749823347 ----------num iterations 100 only
//auROC: Double = 0.9009145623243235 --------------100 iterations and 4 max depth



  val featTest = Array("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", 
  "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", 
  "C25", "C26", "C27", "C28", "C29", "C30", "C31", "C32", "C33", "C34", "C35", "C36", "C37", 
  "C38", "C39", "C40", "C41", "C42", "C43", "C44", "C45", "C46", "C47", "C48", "C49", "C50", 
  "C51", "C52", "C53", "C54", "C55", "C56", "C57", "C58", "C59", "C60", "C61", "C62", "C63", 
  "C64", "C65", "C66", "C67", "C68", "C69", "C70", "C71", "C72", "C73", "C74", "C75", "C76", "C77", 
  "C78", "C79", "C80", "C81", "C82", "C83", "C84", "C85", "C86", "C87", "C88", "C89", "C90", "C91", 
  "C92", "C93", "C94", "C95", "C96", "C97", "C98", "C99", "C100", "C101", "C102", "C103", "C104", 
  "C105", "C106", "C107", "C108", "C109", "C110", "C111", "C112", "C113", "C114", "C115", "C116", "C117", 
  "C118", "C119", "C120", "C121", "C122", "C123", "C124", "C125", "C126", "C127", "C128", "C129", "C130", 
  "C131", "C132", "C133", "C134", "C135", "C136", "C137", "C138", "C139", "C140", "C141", "C142", "C143", 
  "C144", "C145", "C146", "C147", "C148", "C149", "C150", "C151", "C152", "C153", "C154", "C155", 
  "C156", "C157", "C158", "C159", "C160", "C161", "C162", "C163", "C164", "C165", "C166", "C167", "C168", 
  "C169", "C170", "C171", "C172", "C173", "C174", "C175", "C176", "C177", "C178", "C179", "C180", "C181", 
  "C182", "C183", "C184", "C185", "C186", "C187", "C188", "C189", "C190", "C191", "C192", "C193", "C194", 
  "C195", "C196", "C197", "C198", "C199", "C200", "C201", "C202", "C203", "C204", "C205", "C206", "C207", 
  "C208", "C209", "C210", "C211", "C212", "C213", "C214", "C215", "C216", "C217", "C218", "C219", "C220", "C221",
  "C222", "C223", "C224", "C225", "C226", "C227", "C228", "C229", "C230", "C231", "C232", "C233", "C234", 
  "C235", "C236", "C237", "C238", "C239", "C240", "C241", "C242", "C243", "C244", "C245", "C246", "C247", 
  "C248", "C249", "C250", "C251", "C252", "C253", "C254", "C255", "C256", "C257", "C258", "C259", "C260", 
  "C261", "C262", "C263", "C264", "C265", "C266", "C267", "C268", "C269", "C270", "C271", "C272", "C273", 
  "C274", "C275", "C276", "C277", "C278", "C279", "C280", "C281", "C282", "C283", "C284", "C285", "C286", 
  "C287", "C288", "C289", "C290", "C291", "C292", "C293", "C294", "C295", "C296", "C297", "C298", "C299", 
  "C300", "C301", "C302", "C303", "C304", "C305", "C306", "C307", "C308", "C309", "C310", "C311", "C312", "C313", 
  "C314", "C315", "C316", "C317", "C318", "C319", "C320", "C321", "C322", "C323", "C324", "C325", "C326", 
  "C327", "C328", "C329", "C330", "C331", "C332", "C333", "C334", "C335", "C336", "C337", "C338", "C339", 
  "C340", "C341", "C342", "C343", "C344", "C345", "C346", "C347", "C348", "C349", "C350", "C351", "C352", "C353", "C354" )
  
  
val assembler = new VectorAssembler()
  .setInputCols(featTest)
  .setOutputCol("features")

val tfTest = assembler.transform(test)
val dataTest = tfTest.select(col("C0").alias("id"), col("features"))

val res = cvModel.transform(dataTest).select(col("id"),col("prediction"))

res.rdd.saveAsTextFile("REHLresults")



  