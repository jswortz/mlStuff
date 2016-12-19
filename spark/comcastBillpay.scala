import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}
import org.jpmml.sparkml._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.hive.HiveContext

val sqlContext = new HiveContext(sc)

val featureStringFields = Array("csgamdocs", "outageflag", "plannedoutages", "boradcaststatus", "broadcastfunction", 
"accttype", "refacoralloc", "boradcaststatusii", "anilearning", "anilearningii", "seasonal", 
"closedappt", "cancelledappt", "install", "nonpaydisco", "accountstatus", 
"addresstype", "accounttype", "phonetype", "ani2accttype", "dwell_code", "dta", 
"tpv_outstanding", "source", "accountsource", "searchstatus")


//reduced list of variables based off of variable importance
val featureStringFields2 = Array( "outageflag",  "install",  "broadcastfunction",  "boradcaststatus",  "cancelledappt",  "anilearning",  "accountstatus")
//reduced continuous variables based off variable importance
val contFeatures2 = Array("nonpaydiscoii", "delinq_days", "p_p_billing","p_p_p_p_p_billing","p_p_p_billing")

val target = "billing"

val selectStatement = """case when trim(csgamdocs) = '' then 'BLANK' else COALESCE(csgamdocs, 'MISSING') end as csgamdocs,
case when trim(outageflag) = '' then 'BLANK' else COALESCE(outageflag, 'MISSING') end as outageflag,
case when trim(plannedoutages) = '' then 'BLANK' else COALESCE(plannedoutages, 'MISSING') end as plannedoutages,
case when trim(boradcaststatus) = '' then 'BLANK' else COALESCE(boradcaststatus, 'MISSING') end as boradcaststatus,
case when trim(broadcastfunction) = '' then 'BLANK' else COALESCE(broadcastfunction, 'MISSING') end as broadcastfunction,
case when trim(accttype) = '' then 'BLANK' else COALESCE(accttype, 'MISSING') end as accttype,
case when trim(refacoralloc) = '' then 'BLANK' else COALESCE(refacoralloc, 'MISSING') end as refacoralloc,
case when trim(boradcaststatusii) = '' then 'BLANK' else COALESCE(boradcaststatusii, 'MISSING') end as boradcaststatusii,
case when trim(anilearning) = '' then 'BLANK' else COALESCE(anilearning, 'MISSING') end as anilearning,
case when trim(anilearningii) = '' then 'BLANK' else COALESCE(anilearningii, 'MISSING') end as anilearningii,
case when trim(seasonal) = '' then 'BLANK' else COALESCE(seasonal, 'MISSING') end as seasonal,
case when trim(closedappt) = '' then 'BLANK' else COALESCE(closedappt, 'MISSING') end as closedappt,
case when trim(cancelledappt) = '' then 'BLANK' else COALESCE(cancelledappt, 'MISSING') end as cancelledappt,
case when trim(install) = '' then 'BLANK' else COALESCE(install, 'MISSING') end as install,
case when trim(nonpaydisco) = '' then 'BLANK' else COALESCE(nonpaydisco, 'MISSING') end as nonpaydisco,
case when trim(accountstatus) = '' then 'BLANK' else COALESCE(accountstatus, 'MISSING') end as accountstatus,
case when trim(addresstype) = '' then 'BLANK' else COALESCE(addresstype, 'MISSING') end as addresstype,
case when trim(accounttype) = '' then 'BLANK' else COALESCE(accounttype, 'MISSING') end as accounttype,
case when trim(phonetype) = '' then 'BLANK' else COALESCE(phonetype, 'MISSING') end as phonetype,
case when trim(ani2accttype) = '' then 'BLANK' else COALESCE(ani2accttype, 'MISSING') end as ani2accttype,
case when trim(dwell_code) = '' then 'BLANK' else COALESCE(dwell_code, 'MISSING') end as dwell_code,
case when trim(dta) = '' then 'BLANK' else COALESCE(dta, 'MISSING') end as dta,
case when trim(tpv_outstanding) = '' then 'BLANK' else COALESCE(tpv_outstanding, 'MISSING') end as tpv_outstanding,
case when trim(source) = '' then 'BLANK' else COALESCE(source, 'MISSING') end as source,
case when trim(accountsource) = '' then 'BLANK' else COALESCE(accountsource, 'MISSING') end as accountsource,
case when trim(searchstatus) = '' then 'BLANK' else COALESCE( searchstatus, 'MISSING') end as  searchstatus,
 delinq_days, nonpaydiscoii, billing, p_billing, p_p_billing, p_p_p_billing, p_p_p_p_billing, p_p_p_p_p_billing,
 dayofmonth(start_date) as DOM, dayofmonth(start_date) as hour, b.*"""
 
val rawData = sqlContext.sql("select "+ selectStatement + " from `transform`.histNAPIData a left join analysis.Z5_ACXIOM b on a.zipcode = b.zip_code")

val contFeatures = Array("nonpaydiscoii", "delinq_days", "p_p_billing", "p_p_p_billing", "p_p_p_p_billing", "p_p_p_p_p_billing", 
   "hour","DOM","number_of_households", "per_hh_child_0to2", "per_hh_child_3to5", "per_hh_child_6to10", "per_hh_child_11to15", 
   "per_hh_child_16to17", "per_hh_with_small_office", "per_hh_1st_indiv_comp_hsorless", "per_hh_1st_indiv_comp_college", "per_hh_1st_indiv_comp_gradschl", 
   "per_hh_1st_indiv_comp_voctech", "per_hh_with_1_person", "per_hh_with_2_people", "per_hh_with_3_people", "per_hh_with_4_people", 
   "per_hh_with_5_people", "per_hh_with_6_people", "per_hh_with_7_people", "per_hh_with_8ormore_people", "per_hh_with_1_adult", 
   "per_hh_with_2_adults", "per_hh_with_3_adults", "per_hh_with_4_adults", "per_hh_with_5_adults", "per_hh_with_6ormore_adults", 
   "per_hh_with_1_child", "per_hh_with_2_children", "per_hh_with_3ormore_children", "per_hh_with_no_children", "per_hh_with_children", 
   "per_hh_with_1st_indiv_noparty", "per_hh_with_1st_indiv_republ", "per_hh_with_1st_indiv_democrat", "per_hh_with_1st_indiv_indep", 
   "per_hh_with_1_generation", "per_hh_with_2_generations", "per_hh_with_3_generations", "mean_hh_size", "per_hh_income_less_than_15", 
   "per_hh_income_15to20", "per_hh_income_20to30", "per_hh_income_30to40", "per_hh_income_40to50", "per_hh_income_50to75", "per_hh_income_75to100", 
   "per_hh_income_100to125", "per_hh_income_125plus", "per_hh_net_worth_less_than_0", "per_hh_net_worth_0to5", "per_hh_net_worth_5to10", 
   "per_hh_net_worth_10to25", "per_hh_net_worth_25to50", "per_hh_net_worth_50to100", "per_hh_net_worth_100to250", "per_hh_net_worth_250to500", 
   "per_hh_net_worth_500to1million", "per_hh_net_worth_1to2million", "per_hh_net_worth_2millionplus", "per_underbanked_01", "per_underbanked_02", 
   "per_underbanked_03", "per_underbanked_04", "per_underbanked_05", "per_underbanked_06", "per_underbanked_07", "per_underbanked_08", 
   "per_underbanked_09", "per_underbanked_10", "per_underbanked_11", "per_underbanked_12", "per_underbanked_13", "per_underbanked_14", 
   "per_underbanked_15", "per_underbanked_16", "per_underbanked_17", "per_underbanked_18", "per_underbanked_19", "per_underbanked_20", 
   "per_heavy_transactor_01", "per_heavy_transactor_02", "per_heavy_transactor_03", "per_heavy_transactor_04", "per_heavy_transactor_05", 
   "per_heavy_transactor_06", "per_heavy_transactor_07", "per_heavy_transactor_08", "per_heavy_transactor_09", "per_heavy_transactor_10",
   "per_heavy_transactor_11", "per_heavy_transactor_12", "per_heavy_transactor_13", "per_heavy_transactor_14", "per_heavy_transactor_15", 
   "per_heavy_transactor_16", "per_heavy_transactor_17", "per_heavy_transactor_18", "per_heavy_transactor_19", "per_heavy_transactor_20", 
   "per_home_value_lessthan50", "per_home_value_50to100", "per_home_value_100to150", "per_home_value_150to200", "per_home_value_200to250", 
   "per_home_value_250to300", "per_home_value_300to350", "per_home_value_350to400", "per_home_value_400to450", "per_home_value_450to500", 
   "per_home_value_500to750", "per_home_value_750to1million", "per_home_value_1millionplus", "mean_hh_income", "mean_net_worth", 
   "mean_home_market_value", "median_hh_income", "per_hh_interests_gardening", "per_hh_interests_coll_antiques", "per_hh_interests_cooking_food", 
   "per_hh_interests_elect_comp", "per_hh_interests_exercise_health", "per_hh_interests_home_impr", "per_hh_interests_investing_fin", 
   "per_hh_interests_movies_music", "per_hh_interests_outdoors", "per_hh_interests_reading", "per_hh_interests_sports", "per_hh_interests_travel", 
   "per_hh_lor_less_than_1yr", "per_hh_lor_1to2_yrs", "per_hh_lor_3to5_yrs", "per_hh_lor_6to10_yrs", "per_hh_lor_11to14_yrs", "per_hh_lor_14plus_yrs", 
   "per_owner_households", "per_renter_households", "per_hh_mult_fam_dwelling_units", "per_hh_single_fam_dwelling_units", "per_hh_condos", 
   "per_hh_2to4_unit_complexes", "per_hh_misc_residences", "per_hh_apartments", "per_hh_mobile_homes", "per_hh_homes_built_last_year", 
   "per_hh_homes_built_2_yrs_ago", "per_hh_homes_built_3_yrs_ago", "per_hh_homes_built_4_yrs_ago", "per_hh_homes_built_5to9_yrs_ago", 
   "per_hh_homes_built_10to19_y_ago", "per_hh_homes_built_20to29_y_ago", "per_hh_homes_built_30to39_y_ago", "per_hh_homes_built_40to49_y_ago", 
   "per_hh_homes_built_50to74_y_ago", "per_hh_homes_built_75to99_y_ago", "per_hh_homes_built_100_yrs_ago", "per_hh_homes_purch_last_yr", 
   "per_hh_homes_purch_2_yrs_ago", "per_hh_homes_purch_3_yrs_ago", "per_hh_homes_purch_4_yrs_ago", "per_hh_homes_purch_5_yrs_ago", 
   "per_hh_homes_purch_6to9_yrs_ago", "per_hh_homes_purch_10to14_y_ago", "per_hh_homes_purch_15to19_y_ago", "per_hh_homes_purch_20to24_y_ago", 
   "per_hh_homes_purch_25to29_y_ago", "per_hh_homes_purch_30_yrs_ago", "mean_length_residence")
   
   //had to get rid of "p_billing" - was creating target leak
 
val typesRight = rawData.withColumn("nonpaydiscoii",rawData("nonpaydiscoii").cast("double")).withColumn("delinq_days",rawData("delinq_days").cast("double")).withColumn("billing",rawData("billing").cast("double"))

val impute1 = typesRight.na.fill(0).select("billing","outageflag",  "install",  "broadcastfunction",  
"boradcaststatus",  "cancelledappt",  "anilearning",  "accountstatus","nonpaydiscoii", "delinq_days", 
"p_p_billing","p_p_p_p_p_billing","p_p_p_billing")

val Array(trainingData, testData) = impute1.randomSplit(Array(0.9, 0.1))

//val impute1 = impute.cache()

val string_indexers: Array[org.apache.spark.ml.PipelineStage] = featureStringFields2.map(
   cname => new StringIndexer()
     .setInputCol(cname)
     .setOutputCol(s"${cname}_index")
)

val assembler_test = new VectorAssembler()
  .setInputCols(featureStringFields2.map(cname => s"${cname}_index") ++ contFeatures2)
  .setOutputCol("features")
   
val featureIndexer =  new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("featuresIndexed")
  .setMaxCategories(13)

val targetIndexer = new StringIndexer()
  .setInputCol("billing")
  .setOutputCol("label")
  
val rf = new RandomForestClassifier()
   .setLabelCol("label")
   .setFeaturesCol("featuresIndexed")
   .setNumTrees(120)
   .setFeatureSubsetStrategy("onethird")
   .setMaxDepth(4)
   .setMaxBins(32)
   .setImpurity("gini") 
   
val stages = string_indexers ++ Array(assembler_test, featureIndexer,targetIndexer, rf)
   
val pipeline =  new Pipeline().setStages(stages)

val model_test = pipeline.fit(trainingData)

val paramGrid = new ParamGridBuilder()
  .addGrid(rf.numTrees, Array(100,200))
  .addGrid(rf.maxDepth, Array(3,7))
//  .addGrid(rf.impurity, Array("gini","entropy"))
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
val f1 = evaluator3.evaluate(predictions)
println("Precision = " + precision)
println("Recall = " + recall)
println("Accuracy = " + precision)
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


