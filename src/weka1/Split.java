package weka1;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.instance.Randomize;

public class SplitDataset {

    public static void main(String[] args) throws Exception {
        // Load the dataset
        DataSource source = new DataSource("data.arff");
        Instances data = source.getDataSet();

        // Set the random seed
        int seed = 1234;

        // Create the randomizer filter
        Randomize randomizer = new Randomize();
        randomizer.setRandomSeed(seed);
        randomizer.setInputFormat(data);

        // Split the dataset into training and testing sets
        int trainSize = (int) (data.numInstances() * 0.8);
        Instances trainingData = Filter.useFilter(data, randomizer, trainSize);
        int[] testIndices = randomizer.getIndices("Test");
        Instances testingData = data.getSubset(testIndices);

        // Print the number of instances in the training and testing sets
        System.out.println("Number of instances in training set: " + trainingData.numInstances());
        System.out.println("Number of instances in testing set: " + testingData.numInstances());
    }
}
