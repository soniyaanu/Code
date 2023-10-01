package weka1;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instance;

import java.util.Random;
import java.util.Arrays;

public class Train {
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource source = new DataSource("C:\\Users\\hella\\Desktop\\results\\avro_result.arff"); // Replace with your dataset path
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        // Split dataset into training and testing sets (80%/20%)
        int trainSize = (int) Math.round(data.numInstances() * 0.8);
        int testSize = data.numInstances() - trainSize;
        Instances trainingData = new Instances(data, 0, trainSize);

        J48 classifier = new J48(); // Replace with your classifier setup
        classifier.buildClassifier(trainingData); // Train the classifier on training data

        int nIterations = 10; // Number of iterations
        int kFolds = 10;      // Number of folds

        // Initialize arrays to store evaluation results
        double[] precisionScores = new double[nIterations * kFolds];
        double[] f1Scores = new double[nIterations * kFolds];
        double[] mccScores = new double[nIterations * kFolds];
        double[] aucScores = new double[nIterations * kFolds];
        double[] gMeanScores = new double[nIterations * kFolds];

        int scoreIndex = 0; // Index to keep track of evaluation scores

        for (int i = 0; i < nIterations; i++) {
            for (int j = 0; j < kFolds; j++) {
                System.out.println("Iteration " + (i + 1) + ", Fold " + (j + 1));

                // Create an Evaluation object
                Evaluation eval = new Evaluation(trainingData);

                // Get the test instance for this fold
                Instance testInstance = data.instance(j);

                // Calculate the evaluation metrics for this instance
                eval.evaluateModelOnce(classifier, testInstance);

                // Store the evaluation metrics
                precisionScores[scoreIndex] = eval.weightedPrecision();
                f1Scores[scoreIndex] = eval.weightedFMeasure();
                mccScores[scoreIndex] = eval.weightedMatthewsCorrelation();
                aucScores[scoreIndex] = eval.weightedAreaUnderROC();
                gMeanScores[scoreIndex] = Math.sqrt(eval.weightedTruePositiveRate() * eval.weightedTrueNegativeRate());

                scoreIndex++;
            }
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
