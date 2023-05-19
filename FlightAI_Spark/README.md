# Flight Delay Prediction Challenge
https://zindi.africa/competitions/flight-delay-prediction-challenge

author: kogni7

This project uses only the data sets provided by ZINDI. These data sets contain information about flight delays. These are the only used features in this project. The task is to predict the delay.

The file system for this project is:
* FlightAI_Spark (root)
    * Data
    * Submission
        * 1 - x: Submission directories named by the version number
            * submission.csv
    * project/plugins.sbt
    * src/main/scala/Main
    * build.sbt
    * Dockerfile
    * script.sh
    * README.md

This project uses Apache Spark MLlib with a Generalized Linear Model.

Start the program with:
```
docker build -t flight_ai_spark .
docker run -it --rm --name flight_ai_spark -v $PWD/Submission:/Submission flight_ai_spark
```