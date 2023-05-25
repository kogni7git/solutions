package org.InsuranceAI;

import java.io.*;
import java.net.URISyntaxException;
import java.util.*;
import java.util.Scanner;

import smile.data.DataFrame;
import smile.data.type.DataTypes;
import smile.data.type.StructField;
import smile.data.type.StructType;
import smile.io.*;

import org.apache.commons.csv.CSVFormat;


public class Main {
    public static void main(String[] args) throws IOException, URISyntaxException {
        // Welcome!
        System.out.println("Welcome! This program trains a classification model for the Data Science Nigeria 2019 Challenge #1: Insurance Prediction competition on zindi.africa " +
                "(https://zindi.africa/competitions/insurance-prediction-challenge)!");

        List<String> possibleModels = new ArrayList<>();

        possibleModels.add("K-Nearest Neighbor");
        possibleModels.add("Linear Discriminant Analysis");
        possibleModels.add("Fisher's Linear Discriminant");
        possibleModels.add("Regularized Discriminant Analysis");
        possibleModels.add("Logistic Regression");
        possibleModels.add("Radial Basis Function Networks");

        System.out.println("Please choose a classification model for training: ");
        int counter = 0;
        for (String i : possibleModels) {
            System.out.printf("%d for %s.%n", counter, i);
            counter++;
        }

        System.out.println("Please enter the number of the desired model: ");
        Scanner in = new Scanner(System.in);
        String chosenModelString = in.nextLine();
        int chosenModel = Integer.parseInt(chosenModelString);

        System.out.println("On how many folds for cross-validation shall the model be trained? ");
        int cv = Integer.parseInt(in.nextLine());

        System.out.printf("A %s will be trained on %d folds.\n", possibleModels.get(chosenModel), cv);

        // Load data!
        CSVFormat format = CSVFormat.DEFAULT.withFirstRecordAsHeader().withDelimiter(',');
        StructType trainSchema = DataTypes.struct(
                new StructField("Customer Id", DataTypes.StringType),
                new StructField("YearOfObservation", DataTypes.IntegerType),
                new StructField("Insured_Period", DataTypes.DoubleType),
                new StructField("Residential", DataTypes.IntegerType),
                new StructField("Building_Painted", DataTypes.StringType),
                new StructField("Building_Fenced", DataTypes.StringType),
                new StructField("Garden", DataTypes.StringType),
                new StructField("Settlement", DataTypes.StringType),
                new StructField("Building Dimension", DataTypes.IntegerType),
                new StructField("Building_Type", DataTypes.IntegerType),
                new StructField("Date_of_Occupancy", DataTypes.IntegerType),
                new StructField("NumberOfWindows", DataTypes.StringType),
                new StructField("Geo_Code", DataTypes.StringType),
                new StructField("Claim", DataTypes.IntegerType));

        StructType testSchema = DataTypes.struct(
                new StructField("Customer Id", DataTypes.StringType),
                new StructField("YearOfObservation", DataTypes.IntegerType),
                new StructField("Insured_Period", DataTypes.DoubleType),
                new StructField("Residential", DataTypes.IntegerType),
                new StructField("Building_Painted", DataTypes.StringType),
                new StructField("Building_Fenced", DataTypes.StringType),
                new StructField("Garden", DataTypes.StringType),
                new StructField("Settlement", DataTypes.StringType),
                new StructField("Building Dimension", DataTypes.IntegerType),
                new StructField("Building_Type", DataTypes.IntegerType),
                new StructField("Date_of_Occupancy", DataTypes.IntegerType),
                new StructField("NumberOfWindows", DataTypes.StringType),
                new StructField("Geo_Code", DataTypes.StringType));

        DataFrame trainCSV = Read.csv(System.getProperty("user.dir") + "/Data/train_data.csv", format, trainSchema);
        DataFrame testCSV = Read.csv(System.getProperty("user.dir") + "/Data/test_data.csv", format, testSchema);

        // Prepare data!
        List<Object> data = Preparing.prepareData(trainCSV, testCSV);
        double[][] X = (double[][]) data.get(0);
        int[] y = (int[]) data.get(1);
        double[][] X_test = (double[][]) data.get(2);
        List<String> customerId = (List<String>) data.get(3);

        // Train model!
        Modelling model = new Modelling(cv);
        List<Object> result = model.trainModel(X, y, X_test, possibleModels.get(chosenModel));
        int[] y_test = (int[]) result.get(0);
        String auc = (String) result.get(1);
        String settings = (String) result.get(2);

        System.out.println("Experiment successful!");

        auc = auc.replace('{', ' ');
        auc = auc.replace("fit time", "* fit time");
        auc = auc.replace("score time", "* score time");
        auc = auc.replace("validation data size", "* validation data size");
        auc = auc.replace("error", "* error");
        auc = auc.replace("accuracy", "* accuracy");
        auc = auc.replace("sensitivity", "* sensivity");
        auc = auc.replace("specificity", "* specificity");
        auc = auc.replace("precision", "* precision");
        auc = auc.replace("F1 score", "* F1 score");
        auc = auc.replace("MCC", "* MCC");
        auc = auc.replace("AUC", "* AUC");
        auc = auc.replace("log loss", "* log loss");
        auc = auc.replace('}', ' ');
        auc = auc.replace(",\n", "\n");
        System.out.println(auc);

        // Make Submission!
        String submission = Preparing.prepareSubmission(customerId, y_test);
        counter = 1;
        while (new File(System.getProperty("user.dir") + "/Submission/" + counter).exists()) counter++;

        String output = "# Experiment Number " + counter + "\n## Model:\n" + possibleModels.get(chosenModel) + settings + "\n## Results:" + auc;

        new File(System.getProperty("user.dir") + "/Submission/" + counter).mkdir();

        BufferedWriter writer = new BufferedWriter(new FileWriter((System.getProperty("user.dir") + "/Submission/" + counter).concat("/submission.csv")));
        writer.write(submission);
        writer.close();

        writer = new BufferedWriter(new FileWriter((System.getProperty("user.dir") + "/Submission/" + counter).concat("/output.md")));
        writer.write(output);
        writer.close();
        System.out.printf("submission.csv and output.md are saved under Submission/%s", counter);
    }
}