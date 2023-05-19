# Flight Delay Prediction Challenge
https://zindi.africa/competitions/flight-delay-prediction-challenge

author: kogni7

date: 2022/23

This project uses only the data sets provided by ZINDI. These data sets contain information about flight delays. These are the only used features in this project. The task is to predict the delay.

The file system for this project is:
* FlightAI_Scala (root)
    * Data
    * Submission
        * 1 - x: Submission directories named by the version number
            * submission.csv
            * output.md
    * project/plugins.sbt
    * src/main/scala
        * main
        * modelling
        * preparing
    * build.sbt
    * Dockerfile
    * README.md

This project uses SMILE.

Start the program with:
```
docker build -t flight_ai_scala .
docker run -it --rm --name flight_ai_scala -v $PWD/Submission:/Submission flight_ai_scala
```