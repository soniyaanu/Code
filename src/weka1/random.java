package weka1;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;

import java.util.Random;
import java.util.Arrays;

public class random {
    public static void main(String[] args) throws Exception {
        // Load your dataset
        DataSource source = new DataSource("C:\\Users\\hella\\Desktop\\ALL\\results\\maven_result.arff"); // Replace with your dataset path
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        int trainSize = (int) Math.round(data.numInstances() * 0.8);
        int testSize = data.numInstances() - trainSize;
        Instances trainingData = new Instances(data, 0, trainSize);
        Instances testingData = new Instances(data, trainSize, testSize); // Create a testing set

        RandomForest classifier = new RandomForest(); // Replace with your classifier setup
        classifier.buildClassifier(trainingData);

        int nEpochs =40;
        int nIterations = 10; // Number of iterations
        int kFolds = 10;      // Number of folds

        // Initialize arrays to store cross-validation results
        double[][] cvScores = new double[nIterations][kFolds];
        double[] precisionScores = new 
        		double[nIterations];
        double[] f1Scores = new double[nIterations];
        double[] mccScores = new double[nIterations];
        
        
        
        double[] aucScores = new double[nIterations];
        double[] gMeanScores = new double[nIterations];
        double[] stdScores = new double[nIterations]; // Array to store standard deviation scores

        double[] epochAverageMeanScores = new double[nEpochs]; // Array to store average mean scores for each epoch
        double[] epochAverageStdScores = new double[nEpochs]; // Array to store average standard deviation scores for each epoch

        for (int epoch = 1; epoch <= nEpochs; epoch++) {
            System.out.println("Epoch " + epoch + ":");

            for (int i = 0; i < nIterations; i++) {
                // Create an Evaluation object
                Evaluation eval = new Evaluation(trainingData);

                // Perform cross-validation
                eval.crossValidateModel(classifier, trainingData, kFolds, new Random());
                precisionScores[i] = eval.weightedPrecision();
                f1Scores[i] = eval.weightedFMeasure();
                mccScores[i] = eval.weightedMatthewsCorrelation();
                aucScores[i] = eval.weightedAreaUnderROC();
                gMeanScores[i] = Math.sqrt(eval.weightedTruePositiveRate() * eval.weightedTrueNegativeRate());

                // Store the scores
                for (int j = 0; j < kFolds; j++) {
                    cvScores[i][j] = eval.pctCorrect();
                }
            }

            // Calculate and print the mean and standard deviation of the scores for this epoch
            double sumOfMeanScores = 0;
            double sumOfStdScores = 0;
            for (int i = 0; i < nIterations; i++) {
                double sum = 0;
                for (int j = 0; j < kFolds; j++) {
                    sum += cvScores[i][j];
                }
                double meanScore = (sum / kFolds) / 100;
                sumOfMeanScores += meanScore;

                double squaredDiffSum = 0;
                for (int j = 0; j < kFolds; j++) {
                    squaredDiffSum += Math.pow(cvScores[i][j] - meanScore, 2);
                }
                stdScores[i] = (Math.sqrt(squaredDiffSum / (kFolds - 1))) / 100;
                sumOfStdScores += stdScores[i];
            }

            double epochAverageMeanScore = sumOfMeanScores / nIterations;
            epochAverageMeanScores[epoch - 1] = epochAverageMeanScore;

            double epochAverageStdScore = sumOfStdScores / nIterations;
            epochAverageStdScores[epoch - 1] = epochAverageStdScore;

            System.out.println("Average Mean Score for Epoch " + epoch + ": " + epochAverageMeanScore);
            System.out.println("Average Std Score for Epoch " + epoch + ": " + epochAverageStdScore);
        }

        // Calculate and print the overall average of average mean scores and standard deviation scores across all epochs
        double sumOfAllEpochAverageMeanScores = 0;
        double sumOfAllEpochAverageStdScores = 0;
        for (int i = 0; i < nEpochs; i++) {
            sumOfAllEpochAverageMeanScores += epochAverageMeanScores[i];
            sumOfAllEpochAverageStdScores += epochAverageStdScores[i];
        }
        double overallAverageMeanScore = sumOfAllEpochAverageMeanScores / nEpochs;
        double overallAverageStdScore = sumOfAllEpochAverageStdScores / nEpochs;
        System.out.println("Overall Average Mean Score: " + overallAverageMeanScore);
        System.out.println("Overall Average Std Score: " + overallAverageStdScore);
     // Create an Evaluation object for training
        Evaluation trainEval = new Evaluation(trainingData);
        trainEval.crossValidateModel(classifier, trainingData, kFolds, new Random());

        System.out.println("Training Set Evaluation Results:");
        System.out.println("Accuracy: " + (trainEval.pctCorrect() / 100));
        System.out.println("Weighted Precision: " + trainEval.weightedPrecision());
        System.out.println("Weighted F-Measure: " + trainEval.weightedFMeasure());
        System.out.println("Weighted MCC: " + trainEval.weightedMatthewsCorrelation());
        System.out.println("Weighted AUC: " + trainEval.weightedAreaUnderROC());
        System.out.println("Weighted G-Mean: " + Math.sqrt(trainEval.weightedTruePositiveRate() * trainEval.weightedTrueNegativeRate()));


     // Create an Evaluation object for testing
        Evaluation testEval = new Evaluation(testingData);

        // Evaluate the classifier on the testing set
        testEval.crossValidateModel(classifier, testingData, kFolds, new Random());
        System.out.println("Testing Set Evaluation Results:");
        System.out.println("Accuracy: " + (testEval.pctCorrect() / 100));
        System.out.println("Weighted Precision: " + testEval.weightedPrecision());
        System.out.println("Weighted F-Measure: " + testEval.weightedFMeasure());
        System.out.println("Weighted MCC: " + testEval.weightedMatthewsCorrelation());
        System.out.println("Weighted AUC: " + testEval.weightedAreaUnderROC());
        System.out.println("Weighted G-Mean: " + Math.sqrt(testEval.weightedTruePositiveRate() * testEval.weightedTrueNegativeRate()));
    }
}
