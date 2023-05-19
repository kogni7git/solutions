ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.10"

lazy val root = (project in file("."))
  .settings(
    name := "FlightAI_Scala"
  )

enablePlugins(JavaAppPackaging)

libraryDependencies += "com.github.haifengl" % "smile-core" % "3.0.1"
libraryDependencies += "com.github.haifengl" %% "smile-scala" % "3.0.1"
libraryDependencies += "com.github.haifengl" % "smile-mkl" % "3.0.1"
libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.4.6" % Runtime