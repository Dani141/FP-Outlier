package Test

import org.apache.spark.sql.{Dataset, Row, SparkSession}

class TestMetrics {
  def confusionMatrix(dataset: Dataset[Row], spark: SparkSession): Array[Int] = {

    import spark.implicits._

      val a = dataset.select("Class", "Anomaly").map(x =>{
      val realValue = x.get(0)

      val prediction = if(x.getString(1) == "normal") 0 else 1

      if (realValue == 1 && prediction == 0) "fn"
      else if (realValue == 1 && prediction == 1) "tp"
      else if (realValue == 0 && prediction == 0) "tn"
      else "fp"
    }).collect()
    val array = Array(a.count(_=="tp") , a.count(_=="tn") , a.count(_=="fp") , a.count(_=="fn") )
    array
  }

  //precisión
  def accuracy (tp: Double, tn: Double, fp: Double, fn: Double): Double = (tp + tn) / (tp + tn + fp + fn)

  //exactitud
  def precision (tp: Double, fp: Double): Double = tp / (tp + fp)

  //sencibilidad
  def recall (tp: Double, fn: Double): Double = tp / (tp + fn)

  //especificidad
  def especificity(tn: Double, fp: Double) = tn/(tn+fp)

  def f1 (precision: Double,recall: Double): Double = 2*((precision*recall)/(precision+recall))
}
