
import sqlContext.implicits._
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.{StructType,StructField,StringType};

val pattern = ".*(linux[0-9]+|lcu[0-9]+)-.*".r

val bPatt2(itGram, number, x, y, z, q) = testString

val bPatt2 = "CNTNT=.*interpretation grammar=\"(session:.*)\" confidence=\"([0-9]*)\"><input mode=\"([a-z]*)\">(.*)</input><instance><AskCue confidence=\"([0-9]*)\">(.*)</AskCue>.*".r

val gramNm = "GURI.*#(.*)$".r

def serverString(input: String): String = input match {
  case bPatt2(gram, conf, inputMode, utter, aqConf, aqUtter ) => "grammar="+gram+"|conf="+conf+"|inputMode="+inputMode+"|utterance="+utter+"|askQueConf="+aqConf+"|askQueUtterance="+aqUtter
  case pattern(server) => "server="+server
  case gramNm(guri) => "GURI="+guri
  case _ => input}
      
val files = sc.wholeTextFiles("/incoming/nuanceLanding/test")

val fileArray = files.flatMapValues(_.split("\n")
  ).map((x) => x._1 + '|' + x._2).filter(x => (x.contains("EVNT=SWIrcnd") || x.contains("EVNT=SWIrcst") || x.contains("EVNT=SWIrslt"))).map(_.split('|')).map(_.map(serverString(_)).map(_.split('|')).flatten.
  map(_.split("="))).map(_.filter(x=> x(0)!= "NAPD" && x(0)!= "LAPD" && x(0)!= "EOFT" && x(0)!= "EOST" && x(0)!= "CRPT" && x(0)!= 
  "RCPU" && x(0)!= "KEYS" && x(0)!= "RCPU" && x(0)!= "MDVR" &&
    x(0)!= "CPARR" && x(0)!= "GRMR" && x(0)!= "RAWS" && x(0)!= "RSLT" && x(0)!= "SCAL" && x(0)!= "NBST" && x(0)!= "NAPD" && x(0)!= 
   "CADP" && x(0)!= "LADP" && x(0)!= "MPNM" && x(0)!= "DPNM" && x(0)!= "MEDIA" && x(0)!= "MACC" 
   && x(0)!= "DPNM" && x(0)!= "DPNM" && x(0)!= "CPRT" && x(0)!= "CPAR" && x(0)!= "BORT" && x(0)!= "RWST" && x(0)!= "EOSD" && x(0)!= 
   "SRCH" && x(0)!= "OFFS" )).
  map(x => x.collect{ 
   case Array(a: String,b: String) => (a,b) 
      })
	   
case class Mast(server: String, time: String, chan: String, evnt: String, appName: String, acst: Int, grammar: String, gramName: String, lang: String, grmt: String, wght: Int, lst: String, osrver: String, ucpu: Int, scpu: Int, conf: Int, inputMode: String, utterance: String, askqueConf: Int, askQueUtterance: String, endr: String, LA: String)

 def orgData(t: Array[String]): Mast = {  
   if (t. length == 15 && (t contains "SWIrcst")) Mast(t(0),t(1),t(2),t(3),t(4),t(5).toInt,t(6),t(7),t(8),t(9),t(10).toInt,t(11),t(12),t(13).toInt,t(14).toInt,0,"","",0,"","","")
   else if (t.length == 18 && (t contains "SWIrcnd"))  Mast(t(0),t(1),t(2),t(3),"",0,"","","","",0,"","",t(16).toInt,t(17).toInt,t(9).toInt,"",t(7),0,t(8),t(6),t(14))
   else if (t contains "SWIrslt" ) Mast(t(0),t(1),t(2),t(3),"",0,t(4),"","","",0,"","",t(10).toInt,t(11).toInt,t(5).toInt,t(6),t(7),t(8).toInt,t(9),"","")
   else Mast("","","","","",0,"","","","",0,"","",0,0,0,"","",0,"","","")
   }

val data2 = fileArray.map(_.map(_._2)).map(orgData(_)).filter(elem => elem != Mast("","","","","",0,"","","","",0,"","",0,0,0,"","",0,"","","")).toDF()

val format = new java.text.SimpleDateFormat("yyyyMMdd")

val date = format.format(new java.util.Date())

data2.registerTempTable("x")

val data3 = sqlContext.sql("select *, "+ date +"  as yyyymmdd from x")


data3.write.mode("append").parquet("/incoming/nuanceLanding/testTarg/nuance.parquet"+date)
  


   
   







