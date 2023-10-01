package weka1;

import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.classifiers.evaluation.ThresholdCurve;

import java.util.Random;
import java.util.Arrays;

public class New {
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
        int nEpochs = 10;     // Number of epochs

        // Initialize arrays to store evaluation results
        double[] foldResults = new double[kFolds];

        for (int epoch = 1; epoch <= nEpochs; epoch++) {
            System.out.println("Epoch " + epoch + ":");

            for (int i = 0; i < nIterations; i++) {
                System.out.println("Iteration " + (i + 1) + ":");
                
                // Create an Evaluation object
                Evaluation eval = new Evaluation(data);

                // Perform cross-validation
                eval.crossValidateModel(classifier, data, kFolds, new Random());

                // Calculate the evaluation metrics for each fold
                for (int fold = 0; fold < kFolds; fold++) {
                    foldResults[fold] = eval.areaUnderROC(fold);
                }

                // Print results for each fold
                for (int fold = 0; fold < kFolds; fold++) {
                    System.out.println("Fold " + (fold + 1) + ": " + foldResults[fold]);
                }
            }
        }
    }
}
