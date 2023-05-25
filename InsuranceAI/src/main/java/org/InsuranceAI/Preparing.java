package org.InsuranceAI;

import smile.data.DataFrame;

import java.util.Arrays;
import java.util.*;

public class Preparing {
    static List<Object> prepareData(DataFrame trainCSV, DataFrame testCSV) {
        // We don't use Building Dimension, Data_of_Occupancy, NumberOfWindows and Geo_Code because they contain missing data.
        // Building_Painted, Building_Fenced, Garden and Settlement are categorical and need to be encoded.

        String[] buildingPaintedValues = Arrays.stream(trainCSV.column("Building_Painted").toStringArray()).distinct().toArray(String[]::new);
        String[] buildingFencedFalues = Arrays.stream(trainCSV.column("Building_Fenced").toStringArray()).distinct().toArray(String[]::new);
        String[] gardenValues = Arrays.stream(trainCSV.column("Garden").toStringArray()).distinct().toArray(String[]::new);
        String[] settlementValues = Arrays.stream(trainCSV.column("Settlement").toStringArray()).distinct().toArray(String[]::new);

        int[] yearOfObservation = trainCSV.column("YearOfObservation").toIntArray();
        double[] insuredPeriod = trainCSV.column("Insured_Period").toDoubleArray();
        int[] residential = trainCSV.column("Residential").toIntArray();
        Object[] buildingPainted = trainCSV.select("Building_Painted").stream().map(row -> Arrays.asList(buildingPaintedValues).indexOf(row.getString(0))).toArray();
        Object[] buildingFenced = trainCSV.select("Building_Fenced").stream().map(row -> Arrays.asList(buildingFencedFalues).indexOf(row.getString(0))).toArray();
        Object[] garden = trainCSV.select("Garden").stream().map(row -> Arrays.asList(gardenValues).indexOf(row.getString(0))).toArray();
        Object[] settlement = trainCSV.select("Settlement").stream().map(row -> Arrays.asList(settlementValues).indexOf(row.getString(0))).toArray();
        int[] buildingType = trainCSV.column("Building_Type").toIntArray();

        double[][] X = new double[yearOfObservation.length][8];
        for (int i = 0; i < yearOfObservation.length; i++) X[i][0] = yearOfObservation[i];
        for (int i = 0; i < insuredPeriod.length; i++) X[i][1] = insuredPeriod[i];
        for (int i = 0; i < residential.length; i++) X[i][2] = residential[i];
        for (int i = 0; i < buildingPainted.length; i++) X[i][3] = (int) buildingPainted[i];
        for (int i = 0; i < buildingFenced.length; i++) X[i][4] = (int) buildingFenced[i];
        for (int i = 0; i < garden.length; i++) X[i][5] = (int) garden[i];
        for (int i = 0; i < settlement.length; i++) X[i][6] = (int) settlement[i];
        for (int i = 0; i < buildingType.length; i++) X[i][7] = buildingType[i];

        int[] claim = trainCSV.column("Claim").toIntArray();

        int[] y = new int[claim.length];
        System.arraycopy(claim, 0, y, 0, claim.length);

        int[] yearOfObservationTest = testCSV.column("YearOfObservation").toIntArray();
        double[] insuredPeriodTest = testCSV.column("Insured_Period").toDoubleArray();
        int[] residentialTest = testCSV.column("Residential").toIntArray();
        Object[] buildingPaintedTest = testCSV.select("Building_Painted").stream().map(row -> Arrays.asList(buildingPaintedValues).indexOf(row.getString(0))).toArray();
        Object[] buildingFencedTest = testCSV.select("Building_Fenced").stream().map(row -> Arrays.asList(buildingFencedFalues).indexOf(row.getString(0))).toArray();
        Object[] gardenTest = testCSV.select("Garden").stream().map(row -> Arrays.asList(gardenValues).indexOf(row.getString(0))).toArray();
        Object[] settlementTest = testCSV.select("Settlement").stream().map(row -> Arrays.asList(settlementValues).indexOf(row.getString(0))).toArray();
        int[] buildingTypeTest = testCSV.column("Building_Type").toIntArray();

        double[][] X_t = new double[yearOfObservationTest.length][8];
        for (int i = 0; i < yearOfObservationTest.length; i++) X_t[i][0] = yearOfObservationTest[i];
        for (int i = 0; i < insuredPeriodTest.length; i++) X_t[i][1] = insuredPeriodTest[i];
        for (int i = 0; i < residentialTest.length; i++) X_t[i][2] = residentialTest[i];
        for (int i = 0; i < buildingPaintedTest.length; i++) X_t[i][3] = (int) buildingPaintedTest[i];
        for (int i = 0; i < buildingFencedTest.length; i++) X_t[i][4] = (int) buildingFencedTest[i];
        for (int i = 0; i < gardenTest.length; i++) X_t[i][5] = (int) gardenTest[i];
        for (int i = 0; i < settlementTest.length; i++) X_t[i][6] = (int) settlementTest[i];
        for (int i = 0; i < buildingTypeTest.length; i++) X_t[i][7] = buildingTypeTest[i];

        // sort testcsv for submission
        Map<String, Integer> customerIdIndexMapping = new HashMap<>();
        int customerIdLength = testCSV.column("Customer Id").toStringArray().length;
        for (int i = 0; i < customerIdLength; i++) customerIdIndexMapping.put(testCSV.select("Customer Id").get(i).toString(), i);

        ArrayList<String> customerId = new ArrayList<>(customerIdIndexMapping.keySet());
        Collections.sort(customerId);

        double[][] X_test = new double[X_t.length][8];
        for (int i = 0; i < X_test.length; i++) X_test[i] = X_t[customerIdIndexMapping.get(customerId.get(i))];

        // CustomerId in submission.csv
        List<String> Id = new ArrayList<>(List.of(testCSV.column("Customer Id").toStringArray()));
        Collections.sort(Id);

        return Arrays.asList(X, y, X_test, Id);
    }

    static String prepareSubmission(List<String> customerId, int[] prediction) {
        StringBuilder submission = new StringBuilder("Customer Id,Claim\n");
        int counter = 0;
        for (String i : customerId) {
            submission.append(i).append(',').append(prediction[counter]).append("\n");
            counter += 1;
        }
        return submission.toString();
    }
}