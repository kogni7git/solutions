FROM sbtscala/scala-sbt:eclipse-temurin-11.0.17_8_1.8.2_2.13.10 as build
WORKDIR /flight_ai

COPY ./project/plugins.sbt ./project/plugins.sbt
COPY ./src ./src
COPY ./build.sbt ./build.sbt

RUN sbt stage

FROM debian:stable-slim

COPY ./Data ./Data
COPY --from=build /flight_ai/target ./target

RUN apt-get update && apt-get install -y openjdk-11-jre

CMD ./target/universal/stage/bin/flightai_scala