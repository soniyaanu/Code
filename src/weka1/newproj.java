package weka1;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;

import java.util.Random;

public class newproj {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("C:\\Users\\hella\\Desktop\\results\\avro_result.arff");
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        
        int nEpochs = 5;
        int nIterations = 10;
        int kFolds = 10;

        double sumOfAllAveragef1Scores = 0;

        for (int epoch = 1; epoch <= nEpochs; epoch++) {
            for (int i = 0; i < nIterations; i++) {
                Instances trainingData = data.trainCV(kFolds, i, new Random());

                J48 classifier = new J48();
                classifier.buildClassifier(trainingData);

                Evaluation eval = new Evaluation(trainingData);
                eval.crossValidateModel(classifier, trainingData, kFolds, new Random());

                double averagef1Score = eval.weightedFMeasure();
                sumOfAllAveragef1Scores += averagef1Score;
            }
        }

        double averageOfAllAveragef1Scores = sumOfAllAveragef1Scores / (nEpochs * nIterations);
        System.out.println("Average of all averagef1Scores: " + averageOfAllAveragef1Scores);
    }
}
