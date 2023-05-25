package org.InsuranceAI;

import smile.base.rbf.RBF;
import smile.classification.*;
import smile.validation.ClassificationValidations;
import smile.validation.CrossValidation;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Scanner;

public class Modelling {
    int nFolds;

    public Modelling(int cv) {
        nFolds = cv;
    }

    private List<Object> trainKNN(double[][] X, int[] y, double[][] X_test) {
        System.out.println("k? ");
        Scanner in = new Scanner(System.in);
        String kString = in.nextLine();
        int k = Integer.parseInt(kString);

        String settings = "\n## Settings:\n* nFolds: " + this.nFolds + "\n* k: " + k;

        ClassificationValidations<KNN<double[]>> modelCV = CrossValidation.classification(this.nFolds, X, y, (X1, y1) -> KNN.fit(X1, y1, k));
        KNN<double[]> model = KNN.fit(X, y);
        return Arrays.asList(model.predict(X_test), modelCV.toString(), settings);
    }

    private List<Object> trainLDA(double[][] X, int[] y, double[][] X_test) {
        String settings = "\n## Settings:\n* nFolds: " + this.nFolds;
        ClassificationValidations<LDA> modelCV = CrossValidation.classification(this.nFolds, X, y, LDA::fit);
        LDA model = LDA.fit(X, y);
        return Arrays.asList(model.predict(X_test), modelCV.toString(), settings);
    }

    private List<Object> trainFLD(double[][] X, int[] y, double[][] X_test) {
        String settings = "\n## Settings:\n* nFolds: " + this.nFolds;
        ClassificationValidations<FLD> modelCV = CrossValidation.classification(this.nFolds, X, y, FLD::fit);
        FLD model = FLD.fit(X, y);
        return Arrays.asList(model.predict(X_test), modelCV.toString(), settings);
    }

    private List<Object> trainRDA(double[][] X, int[] y, double[][] X_test) {
        System.out.println("alpha? ");
        Scanner in = new Scanner(System.in);
        String alphaString = in.nextLine();
        double alpha = Double.parseDouble(alphaString);

        String settings = "\n## Settings:\n nFolds: " + this.nFolds + "\n* alpha: " + alpha;

        ClassificationValidations<RDA> modelCV = CrossValidation.classification(this.nFolds, X, y, (X1, y1) -> RDA.fit(X1, y1, alpha));
        RDA model = RDA.fit(X, y, alpha);
        return Arrays.asList(model.predict(X_test), modelCV.toString(), settings);
    }
    private List<Object> trainLogisticRegression(double[][] X, int[] y, double[][] X_test) {
        String settings = "\n## Settings:\n* nFolds: " + this.nFolds;
        ClassificationValidations<LogisticRegression> modelCV = CrossValidation.classification(this.nFolds, X, y, LogisticRegression::fit);
        LogisticRegression model = LogisticRegression.fit(X, y);
        return Arrays.asList(model.predict(X_test), modelCV.toString(), settings);
    }

    private List<Object> trainRBF(double[][] X, int[] y, double[][] X_test) {
        System.out.println("Neurons? ");
        Scanner in = new Scanner(System.in);
        String neuronsString = in.nextLine();
        int neurons = Integer.parseInt(neuronsString);

        System.out.println("normalized? ");
        String normalizedString = in.nextLine();
        boolean normalized = Boolean.parseBoolean(normalizedString);

        String settings = "\n## Settings:\n* nFolds: " + this.nFolds + "\n* neurons: " + neurons + "\n* normalized: " + normalized;

        ClassificationValidations<RBFNetwork> modelCV = CrossValidation.classification(this.nFolds, X, y, (X1, y1) -> RBFNetwork.fit(X, y, RBF.fit(X, neurons), normalized));
        RBFNetwork model = RBFNetwork.fit(X, y, RBF.fit(X, neurons), normalized);
        return Arrays.asList(model.predict(X_test), modelCV.toString(), settings);
    }

    public List<Object> trainModel(double[][] X, int[] y, double[][] X_test, String model) {
        if (Objects.equals(model, "K-Nearest Neighbor")) {
            return this.trainKNN(X, y, X_test);
        } else if (Objects.equals(model, "Linear Discriminant Analysis")) {
            return this.trainLDA(X, y, X_test);
        } else if (Objects.equals(model, "Fisher's Linear Discriminant")) {
            return this.trainFLD(X, y, X_test);
        } else if (Objects.equals(model, "Regularized Discriminant Analysis")) {
            return this.trainRDA(X, y, X_test);
        } else if (Objects.equals(model, "Logistic Regression")) {
            return this.trainLogisticRegression(X, y, X_test);
        } else if (Objects.equals(model, "Radial Basis Function Networks")) {
            return this.trainRBF(X, y, X_test);
        } else {
            return Arrays.asList(new int[1000], " ", " ");
        }
    }
}