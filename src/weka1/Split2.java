package weka1;

import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;

public class Split2 {
    public static void main(String[] args) throws Exception {
        // Load your dataset
        DataSource source = new DataSource("C:\\Users\\hella\\Desktop\\ALL\\results\\avro_result.arff");
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        // Define the split ratio (e.g., 80% training, 20% testing)
        int trainSize = (int) Math.round(data.numInstances() * 0.8);
        int testSize = data.numInstances() - trainSize;

        // Create training and testing sets
        Instances trainingData = new Instances(data, 0, trainSize);
        Instances testingData = new Instances(data, trainSize, testSize);

        // Optionally, you can save these sets to separate ARFF files if needed
        ArffSaver saver = new ArffSaver();
         saver.setInstances(trainingData);
        saver.setFile(new File("C:\\Users\\hella\\Desktop\\ALL\\results\\trainingData.arff"));
         saver.writeBatch();
        
        saver.setInstances(testingData);
        saver.setFile(new File("C:\\Users\\hella\\Desktop\\ALL\\results\\testingData.arff"));
         saver.writeBatch();
    }
}
