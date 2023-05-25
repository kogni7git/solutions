FROM maven:3.9 as build
WORKDIR /insurance_ai

COPY ./src ./src
COPY ./pom.xml ./pom.xml

RUN mvn clean dependency:copy-dependencies package

FROM debian:stable-slim

COPY ./Data ./Data
COPY --from=build /insurance_ai/target ./target

RUN apt-get update && apt-get install -y openjdk-11-jre

CMD java -classpath "target/classes:target/dependency/*" org.InsuranceAI.Main