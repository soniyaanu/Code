package weka1;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;

import java.util.Random;
import java.util.Arrays;

public class Crosstrain {
    public static void main(String[] args) throws Exception {
        // Load your dataset
        DataSource source = new DataSource("C:\\Users\\hella\\Desktop\\results\\maven_result.arff"); // Replace with your dataset path
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        int trainSize = (int) Math.round(data.numInstances() * 0.8);
        int testSize = data.numInstances() - trainSize;
        Instances trainingData = new Instances(data, 0, trainSize);
        

        J48 classifier = new J48(); // Replace with your classifier setup
        classifier.buildClassifier(trainingData);
        
        //System.out.println(trainingData);
       
        //System.out.println(testData);

        //J48 classifier = new J48(); // Replace with your classifier setup
         int nEpochs = 5;
        int nIterations = 10; // Number of iterations
        int kFolds = 10;      // Number of folds

        // Initialize arrays to store cross-validation results
        double[][] cvScores = new double[nIterations][kFolds];
        double[] precisionScores = new double[nIterations];
        double[] f1Scores = new double[nIterations];
        double[] mccScores = new double[nIterations];
        double[] aucScores = new double[nIterations];
        double[] gMeanScores = new double[nIterations];
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

        // Print the cross-validation scores for each iteration and fold
     
        for (int i = 0; i < nIterations; i++) {
           System.out.println("Iteration " + (i + 1) + ":");
            for (int j = 0; j < kFolds; j++) {
                System.out.println("Fold " + (j + 1) + ": " + (cvScores[i][j]/100));
            }
            System.out.println();
        }

        // Calculate and print the mean and standard deviation of the scores
        double[] meanScores = new double[nIterations];
        double[] stdScores = new double[nIterations];
        double sumOfMeanScores = 0;
        double sumofStdScores=0;
        double sumOfPrecisionScores = 0;
        double sumOff1Scores = 0;
        double sumOfaucScores = 0;
        double sumOfGmeanScores = 0;
        double sumOfmccScores = 0;
        double  sumofaveragef1Score=0;
        double avgsumOfMeanScores=0;
    
        //double sumOfallf1Scores = 0;
        for (int i = 0; i < nIterations; i++) {
            double sum = 0;
            for (int j = 0; j < kFolds; j++) {
                sum += cvScores[i][j];
            }
            meanScores[i] = (sum / kFolds)/100;
           sumOfMeanScores += meanScores[i];
           avgsumOfMeanScores+=sumOfMeanScores;
           
           sumOff1Scores += f1Scores[i];
           sumOfGmeanScores += gMeanScores[i];
           sumOfaucScores += aucScores[i];
           sumOfPrecisionScores += precisionScores[i];
           sumOfmccScores += mccScores[i];
           
           
            
            double squaredDiffSum = 0;
            for (int j = 0; j < kFolds; j++) {
                squaredDiffSum += Math.pow(cvScores[i][j] - meanScores[i], 2);
            }
            stdScores[i] = (Math.sqrt(squaredDiffSum / (kFolds - 1))/100);
            sumofStdScores+=stdScores[i];
            
        }
        System.out.println("Mean Scores: " + Arrays.toString(meanScores));
        
        //System.out.println("Average Mean Scores: " + Arrays.toString(meanScores));
        double averageMeanScore = sumOfMeanScores / nIterations;
        System.out.println("Average Mean Score: " + averageMeanScore);
        
        System.out.println("Standard Deviations: " + Arrays.toString(stdScores));
        
        double averagestdScore = sumofStdScores / nIterations;
        System.out.println("Average std Score: " + averagestdScore);
        double averagef1Score = sumOff1Scores / nIterations;
        //double avgsumOfMeanScores1 = averageMeanScore / 5;
        //System.out.println("AVg of Average Mean Score: " + avgsumOfMeanScores1);
       
        //double sumofaveragef1Score = averagef1Score / nEpochs;
        //System.out.println("f1scores: " + Arrays.toString(f1Scores));
        //System.out.println("Average f1Score: " + averagef1Score);
       // System.out.println("sumof Average f1Score: " + sumofaveragef1Score);
        // double averageprecisionScore = sumOfPrecisionScores/ nIterations;
        //System.out.println("Average precision Score: " + averageprecisionScore);
        //double averagemccScore = sumOfmccScores/ nIterations;
        // System.out.println("Average mcc Score: " + averagemccScore);
        //double averagegmeanScore = sumOfGmeanScores/ nIterations;
        //System.out.println("Average gmean Score: " + averagegmeanScore);
        //double averageaucScore = sumOfaucScores/ nIterations;
        //System.out.println("Average aucScore: " + averageaucScore);
        
        
        
        
        //System.out.println("mccscores: " + Arrays.toString(mccScores));
        //System.out.println("aucscores: " + Arrays.toString(aucScores));
        //System.out.println(" gMeanScores: " + Arrays.toString( gMeanScores));
        //System.out.println(" precisionScores: " + Arrays.toString( precisionScores));
       // sumOfallf1Scores= averagef1Score/5;
        
       
}
}
 }

