package main.samples

import Test.TestMetrics
import com.citi.ml.FP_Outlier
import com.citi.transformations._
import org.apache.spark.sql._
import org.apache.spark.sql.functions.{concat_ws, lit, monotonically_increasing_id}

object main {
  def main(args: Array[String]): Unit = {
    val spark=SparkSession.builder().master("local").appName("TEST").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    import spark.implicits._

    //Carega de datos original
    val data= spark.read.option("header","true").option("inferSchema","true").csv("data/dataset/shuttle_id.csv")
      .drop("ID")

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

    //Test
    var test = new TestMetrics()
      .confusionMatrix(result,spark)
  }
  //Procesando arrays para guardarlos en csv
  def stringify(c: Column) = functions.concat(lit("["), concat_ws(",", c), lit("]"))

}
