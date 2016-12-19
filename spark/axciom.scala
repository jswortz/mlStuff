

def fullRow(x: Array[String]): Interests = x match {
  case Array(a,b,c,d,e,f,g,h,i,j,k,l,m,n) => Interests(a,b.toInt,c.toInt,d.toInt,e.toInt,f.toInt,g.toInt,h.toInt,i.toInt,j.toInt,k.toInt,l.toInt,m.toInt,n.toInt)
  case _ => Interests("",0,0,0,0,0,0,0,0,0,0,0,0,0)
}

case class Interests(ZIP9_Code:String,
  Number_Of_Households:Int,
  HH_Interests_Gardening:Int,
  HH_Interests_Collect_Antiques:Int,
  HH_Interests_Cooking_Food:Int,
  HH_Interests_Electronics_Comp:Int,
  HH_Interests_Exercise_Health:Int,
  HH_Interests_Home_Improvement:Int,
  HH_Interests_Investing_Finance:Int,
  HH_Interests_Movies_Music:Int,
  HH_Interests_Outdoors:Int,
  HH_Interests_Reading:Int,
  HH_Interests_Sports:Int,
  HH_Interests_Travel:Int)

val data = sc.textFile("/incoming/acxiom/Interests")
val data2 = data.map(_.split("  ")).map(fullRow(_)).toDF()
data2.write.save("/incoming/acxiom/intClean")