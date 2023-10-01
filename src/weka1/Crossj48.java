package weka1;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;

import java.util.Random;
import java.util.Arrays;

public class Crossj48 {
    public static void main(String[] args) throws Exception {
        // Load your dataset
        DataSource source = new DataSource("C:\\Users\\hella\\Desktop\\results\\avro_result.arff"); // Replace with your dataset path
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        J48 classifier = new J48(); // Replace with your classifier setup

        int nIterations =10; // Number of iterations
        int kFolds = 10;      // Number of folds

        // Initialize arrays to store cross-validation results
        double[][] cvScores = new double[nIterations][kFolds];

        for (int i = 0; i < nIterations; i++) {
            // Create an Evaluation object
            Evaluation eval = new Evaluation(data);

            // Perform cross-validation
            eval.crossValidateModel(classifier, data, kFolds, new Random());

            // Store the scores
            for (int j = 0; j < kFolds; j++) {
                cvScores[i][j] = eval.pctCorrect();
            }
        }

        // Print the cross-validation scores for each iteration and fold
        for (int i = 0; i < nIterations; i++) {
            System.out.println("Iteration " + (i + 1) + ":");
            for (int j = 0; j < kFolds; j++) {
                System.out.println("Fold " + (j + 1) + ": " + cvScores[i][j]);
            }
            System.out.println();
        }

        // Calculate and print the mean and standard deviation of the scores
        double[] meanScores = new double[nIterations];
        double[] stdScores = new double[nIterations];
        for (int i = 0; i < nIterations; i++) {
            double sum = 0;
            for (int j = 0; j < kFolds; j++) {
                sum += cvScores[i][j];
            }
            meanScores[i] = sum / kFolds;

            double squaredDiffSum = 0;
            for (int j = 0; j < kFolds; j++) {
                squaredDiffSum += Math.pow(cvScores[i][j] - meanScores[i], 2);
            }
            stdScores[i] = Math.sqrt(squaredDiffSum / (kFolds - 1));
        }
        System.out.println("Mean Scores: " + Arrays.toString(meanScores));
        System.out.println("Standard Deviations: " + Arrays.toString(stdScores));
        
    }
}
