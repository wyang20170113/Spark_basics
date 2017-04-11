/**
  * Created by weiyang on 4/10/17.
  */

/**
  * Here ETP stands for Estimator, Transformer, and Param
  */

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.{Vector,Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame


object ETP {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().config("spark.master","local").appName("ETP").getOrCreate()

    // Preparing trainging data from a list of (label, features) tuples.
    val training = spark.createDataFrame(Seq(
      (1.0,Vectors.dense(0.0,1.1,0.1)),
      (0.0,Vectors.dense(2.0,1.0,-1.0)),
      (0.0,Vectors.dense(2.0,1.3,1.0)),
      (1.0,Vectors.dense(0.0,1.2,-0.5))
    )).toDF("label","features")


    // Create a LogisticRegression instance. This instance is an Estimator.
    val lr = new LogisticRegression()
    //print out the parameters, documentation, and any default values.
    println("Logistic Regression parameters \n" +  lr.explainParams() + "\n")

    //we may set  parameters using setter methods
    lr.setMaxIter(10).setRegParam(0.01)

    // Learng a LogisticRegression model. This uses the parameters stored in lr.
    val model1 = lr.fit(training)
    println("Model 1 was fit using paramters: " + model1.parent.extractParamMap())

    // We may alternatively specify parameters using a ParamMap,
    // which supports several methods for specifying parameters.
    val paramMap = ParamMap(lr.maxIter -> 20).put(lr.maxIter,30).put(lr.regParam->0.1).put(lr.threshold,0.55)

    // One can also combine ParamMaps
    val paramMap2 = ParamMap(lr.predictionCol->"myProbability")
    val paramMapCombined = paramMap ++ paramMap2

    // Now learn a new model using hte paramMapCombined parameters.
    val model2 = lr.fit(training,paramMapCombined)

    // prepare test data
    val test = spark.createDataFrame(Seq(
      (1.0,Vectors.dense(-1.0,1.5,1.3)),
      (0.0,Vectors.dense(3.0,2.0,-0.1)),
      (1.0,Vectors.dense(0.0,2.2,-1.5))
    )).toDF("label","features")

    model2.transform(test).select("features","label","myProbability","probability")
    .collect()
    .foreach{case Row(features: Vector,label: Double, prob: Double,prediction: Vector) =>
    println(s"($features,$label) -> prob = $prob, prediction=$prediction")
    }

    //select($"features",$"label",$"myProbability",$"prediction")

  }

}
