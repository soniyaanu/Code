package weka1;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.WekaPackageManager;
import weka.core.Wilcoxon ;

import weka.experiment.InstanceQuery;
import weka.experiment.*;
import weka.estimators.*;
import weka.core.Statistics;

import java.util.ArrayList;

public class Wilcoxon {
    public static void main(String[] args) throws Exception {
        // Load your dataset
        DataSource source = new DataSource("C:\\Users\\hella\\Desktop\\results\\zookeeper_result.arff"); // Replace with your dataset path
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        J48 classifier = new J48(); // Replace with your classifier setup

        int nIterations = 10; // Number of iterations
        int kFolds = 10;      // Number of folds

        // Initialize arrays to store evaluation results
        double[] precisionScores = new double[nIterations];
        double[] f1Scores = new double[nIterations];
        double[] mccScores = new double[nIterations];
        double[] aucScores = new double[nIterations];
        double[] gMeanScores = new double[nIterations];
        // ... Initialize other arrays

        for (int i = 0; i < nIterations; i++) {
            // ... Perform cross-validation and calculate evaluation metrics
        	Evaluation eval = new Evaluation(data);
            precisionScores[i] = eval.weightedPrecision();
            // ... Assign other metrics

            // Calculate the evaluation metrics
        }

        // Perform Wilcoxon Signed-Rank Test
        double pValuePrecision = performWilcoxonSignedRankTest(precisionScores);

        // Print the p-value
        System.out.println("Wilcoxon Signed-Rank Test p-value for Precision: " + pValuePrecision);

        // ... Perform Wilcoxon Signed-Rank Test for other metrics

        // Calculate and print the mean and standard deviation of the metrics
        // ... Calculate and print mean and standard deviation

        // Calculate and print Wilcoxon Signed-Rank Test results for other metrics
    }

    private static double performWilcoxonSignedRankTest(double[] scores) {
        // Perform the test using the wilcoxonSignedRankTest method from Statistics class
        return Statistics.wilcoxonSignedRankTest(scores);
    }
}
