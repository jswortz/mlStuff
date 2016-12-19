import org.apache.spark.ml.{Pipeline, PipelineModel}

val sameModel = PipelineModel.load("comcastPaymentForest11192916")

val rawData = sqlContext.sql("select * from analysis.comcastBillingFinal4")
   
val typesRight = rawData.withColumn("nonpaydiscoii",rawData("nonpaydiscoii").cast("double")).withColumn("due_diff",rawData("due_diff").cast("double")).withColumn("delinq_days",rawData("delinq_days").cast("double")).withColumn("billing",rawData("billing").cast("double")).withColumn("prev_pay_diff",rawData("prev_pay_diff").cast("double")).withColumn("bal_due_diff",rawData("bal_due_diff").cast("double")).withColumn("past_due_diff",rawData("past_due_diff").cast("double")).withColumn("rec_pay_diff",rawData("rec_pay_diff").cast("double")).withColumn("promise_diff",rawData("promise_diff").cast("double")).withColumn("daysdelinquent",rawData("daysdelinquent").cast("double"))

val schema = typesRight.schema

val pmml = ConverterUtil.toPMML(schema, sameModel)

JAXBUtil.marshalPMML(pmml, new StreamResult(System.out));


