package weka1;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;

import java.util.Random;
import java.util.Arrays;

public class Cross4 {
    public static void main(String[] args) throws Exception {
        // Load  dataset
        DataSource source = new DataSource("C:\\Users\\hella\\Desktop\\ALL\\results\\avro_result.arff"); // Replace with your dataset path
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        J48 classifier = new J48(); // Replace with your classifier setup

        int nIterations = 40; // Number of iterations
        int kFolds = 10;      // Number of folds

        // Initialize arrays to store evaluation results
        double[] precisionScores = new double[nIterations];
        double[] f1Scores = new double[nIterations];
        double[] mccScores = new double[nIterations];
        double[] aucScores = new double[nIterations];
        double[] gMeanScores = new double[nIterations];

        for (int i = 0; i < nIterations; i++) {
            // Create an Evaluation object
            Evaluation eval = new Evaluation(data);

            // Perform cross-validation
            
            eval.crossValidateModel(classifier, data, kFolds, new Random());
            

            
            

            // Calculate the evaluation metrics
            precisionScores[i] = eval.weightedPrecision();
            f1Scores[i] = eval.weightedFMeasure();
            mccScores[i] = eval.weightedMatthewsCorrelation();
            aucScores[i] = eval.weightedAreaUnderROC();
            gMeanScores[i] = Math.sqrt(eval.weightedTruePositiveRate() * eval.weightedTrueNegativeRate());
        }

        // Calculate and print the mean and standard deviation of the metrics
        calculateAndPrintMetrics("Precision", precisionScores);
        calculateAndPrintMetrics("F1 Score", f1Scores);
        calculateAndPrintMetrics("MCC", mccScores);
        calculateAndPrintMetrics("AUC", aucScores);
        calculateAndPrintMetrics("G-Mean", gMeanScores);
    }

    private static void calculateAndPrintMetrics(String metricName, double[] scores) {
        System.out.println(metricName + ":");
        System.out.println("Scores: " + Arrays.toString(scores));
        
        // Calculate and print the mean and standard deviation of the metrics
        double sum = 0;
        for (int i = 0; i < scores.length; i++) {
            sum += scores[i];
        }
        double meanScore = sum / scores.length;

        double squaredDiffSum = 0;
        for (int i = 0; i < scores.length; i++) {
            squaredDiffSum += Math.pow(scores[i] - meanScore, 2);
        }
        double stdScore = Math.sqrt(squaredDiffSum / (scores.length - 1));
        
        System.out.println("Mean " + metricName + ": " + meanScore);
        System.out.println("Standard Deviation " + metricName + ": " + stdScore);
        System.out.println();
    }
}
