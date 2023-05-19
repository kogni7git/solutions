ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.10"

lazy val root = (project in file("."))
  .settings(
    name := "FlightAI_Spark"
  )

enablePlugins(JavaAppPackaging)

libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.4.6" % Runtime
libraryDependencies += "org.apache.spark" % "spark-sql_2.13" % "3.3.2"
libraryDependencies += "org.apache.spark" % "spark-mllib_2.13" % "3.3.2"