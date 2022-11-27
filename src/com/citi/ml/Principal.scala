package com.citi.ml

import com.citi.transformations.EqualRangeBinner
import org.apache.spark.sql._

class Principal {

  var patterns: Array[Array[String]] = Array.fill(0)(Array.fill(0)(""))
  var modeWrite = SaveMode.Append

  def getPatterns()={patterns}

  def setPatterns(newPatterns: Array[Array[String]]):this.type ={
    patterns=newPatterns
    this
  }

  def principal(spark: SparkSession, data: DataFrame,counter: Int): Unit ={

    var bin = data
    for (d <- data.columns){
      if (!d.equals("ID") && !d.equals("Class") && !d.contains("_bin")){
        //Discretizacón de los intervalos en 6
        val bin1= new EqualRangeBinner()
          .setNumBuckets(6)
          .setInputColName(d)
          .setOutputColName(d +"_bin")
          .fit(bin)
          .transform(bin)
        bin = bin1
      }
    }
    var algLFPOF = new FP_Outlier()
      .setMinConfidence(0.6)
      .setMinSupport(0.1)
      .setColumns(bin.columns
        .filter(x => x.contains("_bin")))

      if(counter==0) {
        try {
          //Carega de datos temporal
          val dataTemporary = spark.read.option("header", "true").option("inferSchema", "true").csv("data/temporaryBasis")
            .drop("LFPOF_METRIC", "Anomaly")
          //Unir los dos datos
          dataTemporary.cache()
          bin = dataTemporary.union(bin)
        } catch {
          case e: Exception => {
          }
        }

        //Generaciòn de patrones frecuentes
         algLFPOF= algLFPOF.train(bin)
        patterns=algLFPOF.patterns

        modeWrite = SaveMode.Overwrite
      }

    //Ejecución del algoritmo
    val result= algLFPOF.setPatterns(patterns)
      .transform(bin,spark)
    result.show(160,false)

    //Escribiendo resultados
    result.write
      .mode(modeWrite)
      .option("header","true")
      .csv("data/temporaryBasis")
  }
}
