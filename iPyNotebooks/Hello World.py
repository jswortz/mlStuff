# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

df = sqlContext.read.parquet('/analysis/db/call_tier_analysis')
df.count()

# <codecell>

df.printSchema()

# <codecell>

import pandas as pd
import matplotlib as plt
summary = sqlContext.sql("select division, count(1) as count from analysis.call_tier_analysis group by division").collect()
counts = pd.DataFrame(map(lambda x: x.asDict(), summary))
counts

# <codecell>

import matplotlib.pyplot as plt
plt.bar(4,counts["count"])
plt.xticks(4,counts["division"])
plt.show()

# <codecell>

analysis = sqlContext.sql("select a.call_id, "+cols+",case when flag_ss = 1 or flag_infd = 1 then 1 else 0 end as goodCall"\
                         " from analysis.call_response_analysis a inner join "\
                         "analysis.call_tier_analysis b on a.call_id = b.call_id"\
                         " ")
#calls = sqlContext.sql("select call_id, case when flag_ss = 1 or flag_infd = 1 then 1 else 0 end as goodCall "\
#                       "from analysis.call_tier_analysis")
#analysis = calls.join(toggles, toggles.call_id == calls.call_id)
analysis = analysis.drop('call_id')
analysis.cache()

# <codecell>

#del analysis
#analysis.unpersist()
analysis.describe().toPandas()
#print(analysis.filter(analysis.targus > 0).count())

# <codecell>

from pyspark.mllib.stat import Statistics
from pyspark.ml.feature import VectorAssembler
#this is to build the dense vector of features
assembler = VectorAssembler(
    inputCols=["access_mop_data", "active_chk", "agent_block",\
               "ani_confirm", "appt_cancel_ani_block", "appt_last_24_hrs",\
               "appt_module", "appt_questions", "appt_reschedule", "appt_reschedule_ani_block",\
               "appt_submit_sms", "billing_module", "call_soft_disco_routing", "delinquent", \
               "disabled_equip", "disconnected_account", "ecobill", "espn_full_court",\
               "eta_callback", "etr", "false_alarm_tech", "fulfillment", "home_alarm_tech", \
               "home_and_waiting", "interdivision_moving", "internet_usage", "language_preference",\
               "mailing_address", "mlb_direct_kick", "mlb_xtra_innings", "modem_reset", "modem_reset_last_call", \
               "mpeg4_business", "nhl_center_ice", "outage_sms", "pre_nonpay", "prepaid_service", "promise_to_pay_offer",\
               "rapid_resolve", "rec_disclosure_startofcall", "rec_disclosure_transfer", "remote_programming_sms",\
               "sales_repeat", "same_day_payment", "self_service", "speed_increase", "store_mop", \
               "targus", "tech_module", "trans_soft_disco_routing", "vip", "voc_offer", "vtn",\
               "web_ref", "wireless_gateway", "x1_guide", "xi3"],
    outputCol="features")

transformed = assembler.transform(analysis)
#transformed.take(1)

# <codecell>

from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col
#add back in the label as a lableld point.
data=(transformed.select(col("goodCall").alias("label"), col("features"))
  .map(lambda row: LabeledPoint(row.label, row.features)))
#print(data[0].show(1))

# <codecell>

#split the data
split = data.randomSplit(weights=[1, 2])
test = split[0]
train = split[1]
#train.count()

# <codecell>

from pyspark.mllib.classification import SVMWithSGD, SVMModel
model = SVMWithSGD.train(train, iterations=100,regType = 'l1', regParam = .0001)
labelsAndPreds = test.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(test.count())
print("Training Error = " + str(trainErr))

# <codecell>

from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
model = LogisticRegressionWithLBFGS.train(train)

# Evaluating the model on training data
labelsAndPreds = test.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(test.count())
print("Training Error = " + str(trainErr))

# <codecell>

cols = "access_mop_data, active_chk, agent_block, ani_confirm, appt_cancel_ani_block, appt_last_24_hrs, appt_module, appt_questions, appt_reschedule, appt_reschedule_ani_block, appt_submit_sms, billing_module, call_soft_disco_routing, delinquent, disabled_equip, disconnected_account, ecobill, espn_full_court, eta_callback, etr, false_alarm_tech, fulfillment, home_alarm_tech, home_and_waiting, interdivision_moving, internet_usage, language_preference, mailing_address, mlb_direct_kick, mlb_xtra_innings, modem_reset, modem_reset_last_call, mpeg4_business, nhl_center_ice, outage_sms, pre_nonpay, prepaid_service, promise_to_pay_offer, rapid_resolve, rec_disclosure_startofcall, rec_disclosure_transfer, remote_programming_sms, sales_repeat, same_day_payment, self_service, speed_increase, store_mop, targus, tech_module, trans_soft_disco_routing, vip, voc_offer, vtn, web_ref, wireless_gateway, x1_guide, xi3"

# <codecell>

#toRdd = analysis.map(lambda (a,b): (row.goodCall, row.vip)).take(23)

