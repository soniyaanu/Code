package weka1;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Tree {
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource source = new DataSource("C:\\Users\\hella\\Desktop\\ALL\\results\\avro_result.arff");
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        // Create Random Forest
        RandomForest randomForest = new RandomForest();
        randomForest.buildClassifier(data);

        // Export and visualize decision trees
        for (int i = 0; i < randomForest.getNumFeatures(); i++) {
            RandomForest singleTreeForest = new RandomForest();
            singleTreeForest.setNumFeatures(1);
            singleTreeForest.buildClassifier(data);

            // Print tree
            String treeString = singleTreeForest.toString();
            System.out.println("Tree " + (i + 1) + ":\n" + treeString);
        }
    }
}
