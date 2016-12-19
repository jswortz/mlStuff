import com.databricks.spark.csv
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.jpmml.sparkml._
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}


val sqlContext = new SQLContext(sc)

val schema = StructType(Array(
    StructField("x1", DoubleType, true),
    StructField("x2", DoubleType, true),
    StructField("x3", DoubleType, true),
    StructField("x4", DoubleType, true),
    StructField("type", StringType, true)))
	
val df = sqlContext.read
    .format("com.databricks.spark.csv")
    .option("header", "false") // Use first line of all files as header
    .schema(schema)
    .load("iris.data")
	
val assembler_test = new VectorAssembler()
  .setInputCols(Array("x1", "x2", "x3", "x4"))
  .setOutputCol("features")
  
val targetIndexer = new StringIndexer()
  .setInputCol("type")
  .setOutputCol("label")
  
val rf = new RandomForestClassifier()
   .setLabelCol("label")
   .setFeaturesCol("features")

val stages = Array(assembler_test,targetIndexer, rf)
   
val pipeline =  new Pipeline().setStages(stages)

val model_test = pipeline.fit(df)


val pmml = ConverterUtil.toPMML(schema, model_test)

model_test.save("test123456")

