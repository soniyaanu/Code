package weka1;

import java.util.Random;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;

public class Newj48 {
    public static void main(String[] args) throws Exception {
        // Load your dataset
        DataSource source = new DataSource("C:\\Users\\hella\\Desktop\\ALL\\results\\avro_result.arff"); // Replace with your dataset path
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        int trainSize = (int) Math.round(data.numInstances() * 0.7);
        int testSize = data.numInstances() - trainSize;
        Instances trainingData = new Instances(data, 0, trainSize);
        Instances testingData = new Instances(data, trainSize, testSize); // Create a testing set

        BayesNet classifier = new BayesNet(); // Replace with your classifier setup
        classifier.buildClassifier(trainingData);

        int nEpochs = 40;
        int kFolds = 10; // Number of folds

        // Initialize arrays to store cross-validation results
        double[][] cvScores = new double[nEpochs][kFolds];
        double[] precisionScores = new double[nEpochs];
        double[] f1Scores = new double[nEpochs];
        double[] mccScores = new double[nEpochs];
        double[] aucScores = new double[nEpochs];
        double[] gMeanScores = new double[nEpochs];
        double[] stdScores = new double[nEpochs]; // Array to store standard deviation scores

        for (int epoch = 1; epoch <= nEpochs; epoch++) {
            System.out.println("Epoch " + epoch + ":");

            // Create an Evaluation object
            Evaluation eval = new Evaluation(trainingData);

            // Perform cross-validation
            eval.crossValidateModel(classifier, trainingData, kFolds, new Random());

            precisionScores[epoch - 1] = eval.weightedPrecision();
            f1Scores[epoch - 1] = eval.weightedFMeasure();
            mccScores[epoch - 1] = eval.weightedMatthewsCorrelation();
            aucScores[epoch - 1] = eval.weightedAreaUnderROC();
            gMeanScores[epoch - 1] = Math.sqrt(eval.weightedTruePositiveRate() * eval.weightedTrueNegativeRate());

            // Store the scores
            for (int j = 0; j < kFolds; j++) {
                cvScores[epoch - 1][j] = eval.pctCorrect();
            }

            // Calculate and print the mean and standard deviation of the scores for this epoch
            double sum = 0;
            double squaredDiffSum = 0;

            for (int j = 0; j < kFolds; j++) {
                sum += cvScores[epoch - 1][j];
            }

            double meanScore = (sum / kFolds) / 100;

            for (int j = 0; j < kFolds; j++) {
                squaredDiffSum += Math.pow(cvScores[epoch - 1][j] - meanScore, 2);
            }

            stdScores[epoch - 1] = (Math.sqrt(squaredDiffSum / (kFolds - 1))) / 100;

            System.out.println("Mean Score for Epoch " + epoch + ": " + meanScore);
            System.out.println("Std Score for Epoch " + epoch + ": " + stdScores[epoch - 1]);
        }

        // Calculate and print the overall average of average mean scores and standard deviation scores across all epochs
        double sumOfAllMeanScores = 0;
        double sumOfAllStdScores = 0;

        for (int i = 0; i < nEpochs; i++) {
            sumOfAllMeanScores += precisionScores[i];
            sumOfAllStdScores += stdScores[i];
        }

        double overallAverageMeanScore = sumOfAllMeanScores / nEpochs;
        double overallAverageStdScore = sumOfAllStdScores / nEpochs;

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
