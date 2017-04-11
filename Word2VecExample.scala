/**
  * Created by weiyang on 4/10/17.
  */
package org.apache.spark.examples.ml

// $example on$
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
// $example off$
import org.apache.spark.sql.SparkSession

object Word2VecExample {
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .appName("Word2Vec example")
      .config("spark.master","local")
      .getOrCreate()

    // $example on$
    // Input data: Each row is a bag of words from a sentence or document.
    val documentDF = spark.createDataFrame(Seq(
      ("Hi I heard about Spark".split(" ")),
      ("I wish Java could use case classes".split(" ")),
      ("Logistic regression models are neat".split(" "))
    ).map(Tuple1.apply)).toDF("text")

    documentDF.show()
    // Learn a mapping from words to Vectors.
    val word2Vec = new Word2Vec()
      .setInputCol("text")
      .setOutputCol("result")
      .setVectorSize(3)
      .setMinCount(0)
    val model = word2Vec.fit(documentDF)

    val result = model.transform(documentDF)
    result.collect().foreach { case Row(text: Seq[_], features: Vector) =>
      println(s"Text: [${text.mkString(", ")}] => Vector: $features\n") }
    // $example off$

    spark.stop()
  }
}
