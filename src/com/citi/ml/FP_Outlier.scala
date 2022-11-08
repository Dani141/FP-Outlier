package com.citi.ml
import org.apache.spark.ml.fpm.{FPGrowth, FPGrowthModel}
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StringType, StructField}

 class FP_Outlier extends Serializable{
  var minSupport = 0.0
  var cols: Array[String] = Array.fill(0)("")
  var minConfidence = 0.5
  var patterns = Array.fill(0)((Array.fill(0)(""), 0L))
  var totalElement = 0L
  var model: FPGrowthModel = null

  //Funcion para especificar el soporte minimo de los patrones a minar
  def setMinSupport(supportParam: Double): this.type = {
    minSupport = supportParam
    this
  }

  //Funcion para especificar la confianza minima de los patrones a minar
  def setMinConfidence(confidenceParam: Double): this.type = {
    minConfidence = confidenceParam
    this
  }

  //Funcion para especificar el nombre de las columnas a utilizar para el minado
  def setColumns(columnsNamesBinners: Array[String]): this.type = {
    cols = columnsNamesBinners
    this
  }

  //Funcion para entrenar el modelos de datos y obtener los patrones k describan el set de datos
  def train(trainData: Dataset[Row]): this.type = {
    var arrCols = Array.fill(0)(col(""))
    var notAnalitycCols = Array.fill(0)(col(""))
    cols.foreach(x => arrCols = arrCols :+ col(x))
    trainData.columns
      .foreach(x => notAnalitycCols = notAnalitycCols :+ col(x))
    totalElement = trainData.count()
    val dataFeatured = trainData.select(notAnalitycCols :+ array(arrCols: _*).as("features"): _*)
    val fpgrowth = new FPGrowth().setItemsCol("features").setMinSupport(minSupport).setMinConfidence(minConfidence)

    //Creaciòn del modelo
    val cuncurrenteModel = fpgrowth.fit(dataFeatured)
    model=cuncurrenteModel
    
    // Display frequent itemsets
    val freqPatterns=model.freqItemsets
    freqPatterns.show(10, false)

    val freqPatternSize = freqPatterns.agg(sum("freq")).collectAsList().get(0).get(0).asInstanceOf[Long]
    println("Mean size of patterns "+ freqPatternSize / totalElement)

    patterns = model.freqItemsets.collect().map(x => (x.getAs[Seq[String]](0).toArray, x.getLong(1)))
    this
  }

  //Funcion para evaluar las trnsacciones basado en el modelo obtenido utlizando LFPOF
  def transform(data: Dataset[Row], spark: SparkSession): Dataset[Row] = {
    import spark.implicits._
    var arrCols = Array.fill(0)(col(""))
    var notAnalitycCols = Array.fill(0)(col(""))
    cols.foreach(x => arrCols = arrCols :+ col(x))
    data.columns
      .foreach(x => notAnalitycCols = notAnalitycCols :+ col(x))
    var dataFeatured = data.select(notAnalitycCols :+ array(arrCols: _*).as("features"): _*)
    val posFeatures = dataFeatured.schema.fields.map(x => x.name).indexOf("features")
    var tmpSch = dataFeatured.schema
      .add(new StructField("LFPOF_METRIC", DoubleType))
    dataFeatured = dataFeatured.map(x => Row(x.toSeq :+ (lfpof(x.getAs[Seq[String]](posFeatures))): _*))(RowEncoder.apply(tmpSch))

    //Add column of anomaly
    val lfpofCoefficent=anomalyCoefficient(dataFeatured)
    tmpSch = dataFeatured.schema
      .add(new StructField("Anomaly",StringType ))
    dataFeatured = dataFeatured.map(x => Row(x.toSeq :+ ( itsAbnormal(lfpof(x.getAs[Seq[String]](posFeatures)),lfpofCoefficent)) : _*))(RowEncoder.apply(tmpSch))

    dataFeatured
  }

  //Funcion para calcular FPOF
  def findCoverFPOF(features: Seq[String], patterns: Array[(Array[String], Long)]): Double = {
    var sum = 0.0
    for (ptr <- patterns) {
      if (matchWithFullPattern(features, ptr)) {
        sum += ptr._2.toDouble / totalElement.toDouble
      }
    }
    sum / patterns.length
  }

  //Funcion para calcular WCFPOF
  def findCoverWCFPOF(features: Seq[String], patterns: Array[(Array[String], Long)]): Double = {
    var sum = 0.0
    for (ptr <- patterns) {
      if (matchWithFullPattern(features, ptr)) {
        sum += (ptr._2.toDouble / totalElement.toDouble) * ptr._1.length / features.length
      }
    }
   sum / patterns.length
  }

  private def fpof(features: Seq[String]) = (findCoverFPOF(features, patterns))

  private def wcpof(features: Seq[String]) = (findCoverWCFPOF(features, patterns))

  private def lfpof(features: Seq[String]) = (findMaxSizePatternLFPOF(features, patterns))

  //Funcion para calcular LFPOF
  private def findMaxSizePatternLFPOF(feature: Seq[String], patterns: Array[(Array[String], Long)]): Double = {
    var maxSize = 0
    for (pttrn <- patterns) {
      if (matchWithFullPattern(feature, pttrn) && maxSize < pttrn._1.length)
        maxSize = pttrn._1.length
    }
    maxSize.toDouble / feature.length.toDouble
  }

  private def matchWithFullPattern(item: Seq[String], pattern: (Array[String], Long)): Boolean = {
    var counter = 0
    pattern._1.foreach(x => if (item.contains(x)) counter += 1)
    if (counter != pattern._1.length)
      false
    else
      true
  }

 /*private def countMatchingPattern = udf((colitem: Seq[String]) => {
    var cant = 0
    patterns.foreach(x => if (itemMatchWithPattern(colitem, x) > 0) cant += 1)
    cant
  })

  private def countMatchingItemSet = udf((colitem: Seq[String]) => {
    var cantMatch, cantTotalItem = 0
    patterns.foreach { x =>
      cantMatch += itemMatchWithPattern(colitem, x)
    }
    cantMatch
  })

  private def countTotalItemSet = udf((colitem: Seq[String]) => {
    var cantMatch, cantTotalItem = 0
    patterns.foreach { x =>
      if (itemMatchWithPattern(colitem, x) > 0)
        cantTotalItem += x._1.length
    }
    cantTotalItem
  })*/

  private case class matching(CantItemSetMatch: Int, CantTotalItemSet: Int)

  private def itemMatchWithPattern(item: Seq[String], pattern: (Array[String], Long)): Int = {
    var counter = 0
    pattern._1.foreach(x => if (item.contains(x)) counter += 1)
    counter
  }

   private def anomalyCoefficient(dataset: Dataset[Row]): Double ={
    var coefficient=1.0

    val valuesList=dataset.select("LFPOF_METRIC").collect().toList
    for (i<-valuesList){
      val x=i.getDouble(0)
      if (coefficient>x)
        coefficient=x
    }
      if(coefficient<0.4)
       coefficient=coefficient+0.2
     coefficient
  }

   private def itsAbnormal(lfpofValue: Double,lfpofCoefficent: Double ): String={
     var result="normal"
     if(lfpofValue<=lfpofCoefficent) {
       result="abnormal"
     }
     result
   }
}
