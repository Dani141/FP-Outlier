package Test

import org.apache.spark.sql.{Dataset, Row, SparkSession}

class TestMetrics {
  def confusionMatrix(dataset: Dataset[Row], spark: SparkSession): Unit = {

    import spark.implicits._
    val a = dataset.select("Class", "Anomaly").map { x =>
      val realValue = x.get(0)
      /*val v = x.getString(x.fieldIndex("Class"))
      val realValue = v.substring(1, v.length-1).split(',').map{ x => x.toDouble}.last.toInt*/
      val prediction = if(x.getString(1) == "normal") 0 else 1

      if (realValue == 0 && prediction == 0) "tn"
      else if (realValue == 1 && prediction == 1) "tp"
      else if (realValue == 0 && prediction == 1) "fp"
      else "fn"
    }.collect()
    val array = Array(a.count(_=="tp") , a.count(_=="tn") , a.count(_=="fp") , a.count(_=="fn") )

    //metrics
    println("accuracy: "+ accuracy(array(0),array(1),array(2),array(3)))
    println("precision: "+ precision(array(0),array(2)))
    println("recall: "+ recall(array(0),array(3)))

  }

  def accuracy (tp: Double, tn: Double, fp: Double, fn: Double): Double = (tp + tn) / (tp + tn + fp + fn)

  def precision (tp: Double, fp: Double): Double = tp / (tp + fp)

  def recall (tp: Double, fn: Double): Double = tp / (tp + fn)
}
