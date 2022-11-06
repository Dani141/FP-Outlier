package com.citi.ml

import com.citi.transformations.EqualRangeBinner
import org.apache.spark.sql.functions.{concat_ws, lit, monotonically_increasing_id}
import org.apache.spark.sql._

class Principal {
  def principal(spark: SparkSession, data: DataFrame): Unit ={

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

    try {
      //Carega de datos temporal
      val dataTemporary = spark.read.option("header", "true").option("inferSchema", "true").csv("data/temporaryBasis")
        .drop("ID", "features", "LFPOF_METRIC", "Anomaly")
      //Unir los dos datos
      dataTemporary.cache()
      bin = dataTemporary.union(bin).withColumn("ID", monotonically_increasing_id())
    }catch {
      case e: Exception =>{
      }
    }

    //Generaciòn de patrones frecuentes
    val algLFPOF= new FP_Outlier()
      .setMinConfidence(0.6)
      .setMinSupport(0.1)
      .setColumns(bin.columns
        .filter(x=>x.contains("_bin")))
      .train(bin)

    //Ejecución del algoritmo
    val result= algLFPOF.transform(bin,spark)
    result.show(160,false)

    //Escribiendo resultados
    result.withColumn("features", stringify(result.col("features")))
      .write
      .mode(SaveMode.Overwrite)
      .option("header","true")
      .csv("data/temporaryBasis")

}
  //Procesando arrays para guardarlos en csv
  def stringify(c: Column) = functions.concat(lit("["), concat_ws(",", c), lit("]"))
}