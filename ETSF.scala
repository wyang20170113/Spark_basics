/**
  * Created by weiyang on 4/10/17.
  */

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{HashingTF,IDF,Tokenizer}

import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row

object ETSF {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().config("spark.master","local").appName("ETSF").getOrCreate()
    val sentenceData = spark.createDataFrame(Seq(
      (0.0,"Hi, I heard about spark"),
      (0.0,"I wish Java could use case classes"),
      (1.0,"Logistic regression model are neat")
    )).toDF("label","sentence")

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData)
    //wordsData.collect().foreach(println)
    val hashingTF = new HashingTF()
    .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)
    val featurizedData = hashingTF.transform(wordsData)
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.select("label","features").show()

  }


}
