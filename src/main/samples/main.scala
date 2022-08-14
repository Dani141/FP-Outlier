package main.samples

import com.citi.ml.FP_Outlier
import com.citi.transformations._
import org.apache.hadoop.fs.FileSystem
import org.apache.spark.sql._
import org.apache.spark.sql.functions.{concat_ws, lit, monotonically_increasing_id}

object main {
  def main(args: Array[String]): Unit = {
    val spark=SparkSession.builder().master("local[*]").appName("TEST").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    val fileLocalSystem=FileSystem.get(spark.sparkContext.hadoopConfiguration)

    import spark.implicits._

    //Carega de datos original
    var data= spark.read.option("header","true").option("inferSchema","true").csv("data/dataset/mammography_id.csv")
      .drop("ID")
    try {
      //Carega de datos temporal
      var dataTemporary = spark.read.option("header", "true").option("inferSchema", "true").csv("data/temporaryBasis")
        .drop("ID", "features", "LFPOF_METRIC", "FPOF_METRIC", "WCPOF_METRIC")
      for (d <- dataTemporary.columns)
        if (d.contains("_bin"))
          dataTemporary = dataTemporary.drop(d)
      //Unir los dos datos
      data = dataTemporary.union(data).withColumn("ID", monotonically_increasing_id())
    }catch {
      case e: Exception =>{}
    }

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

    //Discretizacón de los intervalos en 6
    val algLFPOF= new FP_Outlier()
      .setMinConfidence(0.6)
      .setMinSupport(0.1)
      .setColumns(bin.columns
      .filter(x=>x.contains("_bin")))
      .train(bin)

    //Ejecución del algoritmo
    var result= algLFPOF.transform(bin,spark)
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
  /*def deleteTemporaryBasis(fs: FileSystem): Unit ={
    if(fs.exists("data/temporaryBasis") && fs.isFile("data/temporaryBasis"))
      fs.delete("data/temporaryBasis",true)
  }*/
}
