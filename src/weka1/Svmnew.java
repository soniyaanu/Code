package weka1;


import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

public class Svmnew {
    public static void main(String[] args) throws Exception {
        // Load your dataset
        DataSource source = new DataSource("C:\\Users\\hella\\Desktop\\ALL\\results\\avro_result.arff");
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        int nEpochs = 40;
        int nIterations = 10;
        int kFolds = 10;

        // Initialize arrays to store cross-validation results
        double[][] cvScores = new double[nIterations][kFolds];
        double[] precisionScores = new double[nIterations];
        double[] f1Scores = new double[nIterations];
        double[] mccScores = new double[nIterations];
        double[] aucScores = new double[nIterations];
        double[] gMeanScores = new double[nIterations];
        double[] stdScores = new double[nIterations];

        // Create the base classifier (SVM)
        SMO baseClassifier = new SMO();

        double sumOfAllEpochAverageMeanScores = 0;
        double sumOfAllEpochAverageStdScores = 0;

        for (int epoch = 1; epoch <= nEpochs; epoch++) {
            System.out.println("Epoch " + epoch + ":");

            double sumOfMeanScores = 0;
            double sumOfStdScores = 0;

            for (int i = 0; i < nIterations; i++) {
                Instances trainingData = data.trainCV(kFolds, i);
                Instances testingData = data.testCV(kFolds, i);

                baseClassifier.buildClassifier(trainingData);

                Evaluation eval = new Evaluation(trainingData);
                eval.evaluateModel(baseClassifier, testingData);

                precisionScores[i] = eval.weightedPrecision();
                f1Scores[i] = eval.weightedFMeasure();
                mccScores[i] = eval.weightedMatthewsCorrelation();
                aucScores[i] = eval.weightedAreaUnderROC();
                gMeanScores[i] = Math.sqrt(eval.weightedTruePositiveRate() * eval.weightedTrueNegativeRate());

                cvScores[i] = eval.predictions();

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
            double epochAverageStdScore = sumOfStdScores / nIterations;

            System.out.println("Average Mean Score for Epoch " + epoch + ": " + epochAverageMeanScore);
            System.out.println("Average Std Score for Epoch " + epoch + ": " + epochAverageStdScore);

            sumOfAllEpochAverageMeanScores += epochAverageMeanScore;
            sumOfAllEpochAverageStdScores += epochAverageStdScore;
        }

        double overallAverageMeanScore = sumOfAllEpochAverageMeanScores / nEpochs;
        double overallAverageStdScore = sumOfAllEpochAverageStdScores / nEpochs;

        System.out.println("Overall Average Mean Score: " + overallAverageMeanScore);
        System.out.println("Overall Average Std Score: " + overallAverageStdScore);

        // Perform evaluation on the training set
        Evaluation trainEval = new Evaluation(data);
        trainEval.crossValidateModel(baseClassifier, data, kFolds, new Random());

        System.out.println("Training Set Evaluation Results:");
        System.out.println("Accuracy: " + (trainEval.pctCorrect() / 100));
        System.out.println("Weighted Precision: " + trainEval.weightedPrecision());
        System.out.println("Weighted F-Measure: " + trainEval.weightedFMeasure());
        System.out.println("Weighted MCC: " + trainEval.weightedMatthewsCorrelation());
        System.out.println("Weighted AUC: " + trainEval.weightedAreaUnderROC());
        System.out.println("Weighted G-Mean: " + Math.sqrt(trainEval.weightedTruePositiveRate() * trainEval.weightedTrueNegativeRate()));
    }
}
