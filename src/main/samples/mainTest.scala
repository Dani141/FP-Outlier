package main.samples

import Test.TestMetrics
import com.citi.ml.Principal
import org.apache.spark.sql.SparkSession

import scala.concurrent.duration.{Duration, NANOSECONDS}

object mainTest {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().master("local").appName("TEST").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    //Carega de datos original
    val data = spark.read.option("header", "true").option("inferSchema", "true").csv("data/dataset/smtp.csv")
      .drop("ID")

    val splitData = data.randomSplit(Array(0.7, 0.1, 0.1, 0.1), 2)

    val dataResult = splitData.map { partition =>

      val ini_time = System.nanoTime()
      //Algoritmo
      val principal = new Principal()
      principal.principal(spark, partition)

      val end_time = System.nanoTime()

      //Matriz de confusion
      val results = spark.read.option("header", "true").option("inferSchema", "true").csv("data/temporaryBasis")
      val metrics = new TestMetrics()
      val m = metrics.confusionMatrix(results, spark)

      val tp = m(0)
      val tn = m(1)
      val fp = m(2)
      val fn = m(3)
      val accuracy = BigDecimal(metrics.accuracy(tp, tn, fp, fn) * 100).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
      val precision = BigDecimal(metrics.precision(tp, fp) * 100).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
      val recall = BigDecimal(metrics.recall(tp, fn) * 100).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble

      (Duration(end_time - ini_time, NANOSECONDS).toMillis.toDouble / 1000, tp, tn, fp, fn, accuracy, precision, recall)
    }


    println("************************************************")
    println("************************************************")
    println("************************************************")
    println("Time: " + dataResult.map(_._1).mkString("", "seg , ", ""))
    println("True Positives: " + dataResult.map(_._2).mkString("", ", ", ""))
    println("True Negatives: " + dataResult.map(_._3).mkString("", ", ", ""))
    println("False Positives: " + dataResult.map(_._4).mkString("", ", ", ""))
    println("False Negatives: " + dataResult.map(_._5).mkString("", ", ", ""))
    println("Accuracy: " + dataResult.map(_._6).mkString("", ", ", ""))
    println("Precision: " + dataResult.map(_._7).mkString("", ", ", ""))
    println("Recall: " + dataResult.map(_._8).mkString("", ", ", ""))
    println("************************************************")
    println("************************************************")
    println("************************************************")
    println("************************************************")

  }


}
