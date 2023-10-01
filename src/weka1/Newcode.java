package weka1;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;

public class CustomCode {

    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("C:\\Users\\hella\\Desktop\\ALL\\results\\avro_result.arff");
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        int kFolds = 10;
        int nRepetitions = 40;

        Logistic classifier = new Logistic();

        for (int rep = 0; rep < nRepetitions; rep++) {
            Instances shuffledData = new Instances(data);
            shuffledData.randomize(new java.util.Random(rep));

            // Split data into training and testing sets
            int trainSize = (int) Math.round(shuffledData.numInstances() * 0.7);
            int testSize = shuffledData.numInstances() - trainSize;
            Instances trainingData = new Instances(shuffledData, 0, trainSize);
            Instances testingData = new Instances(shuffledData, trainSize, testSize);

            classifier.buildClassifier(trainingData);

            // Evaluate the classifier on the testing set
            Evaluation testEval = new Evaluation(trainingData);
            testEval.evaluateModel(classifier, testingData);

            // Print classifier information
            System.out.println("Classifier: Logistic");
            System.out.println("kFolds: " + kFolds);
            System.out.println("Repetition: " + (rep + 1));

            // Print testing set evaluation results
            printEvaluationResults(testEval);

            System.out.println("============================");
        }
    }

    private static void printEvaluationResults(Evaluation eval) {
        System.out.println("Testing Set Evaluation Results:");
        System.out.println("Accuracy: " + (eval.pctCorrect() / 100));
        System.out.println("Weighted Precision: " + eval.weightedPrecision());
        System.out.println("Weighted F-Measure: " + eval.weightedFMeasure());
        System.out.println("Weighted MCC: " + eval.weightedMatthewsCorrelation());
        System.out.println("Weighted AUC: " + eval.weightedAreaUnderROC());
        System.out.println("Weighted G-Mean: " + Math.sqrt(eval.weightedTruePositiveRate() * eval.weightedTrueNegativeRate()));
    }
}
