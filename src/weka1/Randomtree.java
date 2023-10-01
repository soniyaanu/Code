package weka1;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

// Import additional classes for tree visualization
import weka.classifiers.trees.RandomTree;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;
import weka.gui.treevisualizer.TreeVisualizerListener;

public class Randomtree {
    // Main driver method
    public static void main(String args[]) {
        // Try block to check for exceptions
        try {
            // Create RandomForest classifier by creating an object of the RandomForest class
            RandomForest RFClassifier = new RandomForest();

            // Dataset path
            String Dataset = "C:\\Users\\hella\\Desktop\\results\\camel_result.arff";

            // Creating buffered reader to read the dataset
            BufferedReader bufferedReader = new BufferedReader(new FileReader(Dataset));

            // Create dataset instances
            Instances datasetInstances = new Instances(bufferedReader);

            // Set Target Class
            datasetInstances.setClassIndex(datasetInstances.numAttributes() - 1);

            // Evaluating by creating an object of Evaluation class
            Evaluation evaluation = new Evaluation(datasetInstances);

            // Cross Validate Model with 10 folds
            evaluation.crossValidateModel(RFClassifier, datasetInstances, 10, new Random(1));

            System.out.println(evaluation.toSummaryString("\nResults", true));

            System.out.println("precision Class0:" + evaluation.precision(0));
            System.out.println("precision Class1:" + evaluation.precision(1));
            System.out.println("weighted precision = " + evaluation.weightedPrecision());
            System.out.println("recall Class0:" + evaluation.recall(0));
            System.out.println("recall Class1:" + evaluation.recall(1));
            System.out.println("weighted recall  = " + evaluation.weightedRecall());
            System.out.println("Fmeasure Class0:" + evaluation.fMeasure(0));
            System.out.println("Fmeasure Class1:" + evaluation.fMeasure(1));
            System.out.println("weighted Fmeasure  = " + evaluation.weightedFMeasure());
            System.out.println("MCC Class0:" + evaluation.matthewsCorrelationCoefficient(0));
            System.out.println("MCC Class1:" + evaluation.matthewsCorrelationCoefficient(1));
            System.out.println("weighted MCC  = " + evaluation.weightedMatthewsCorrelation());
            System.out.println("Roc area Class0:" + evaluation.areaUnderROC(0));
            System.out.println("Roc area Class1:" + evaluation.areaUnderROC(1));
            System.out.println("weighted Roc area  = " + evaluation.weightedAreaUnderROC());

            // Visualize the first tree in the forest
            if (RFClassifier.getNumTrees() > 0) {
                RandomTree tree = RFClassifier.getTree(0);
                visualizeTree(tree);
            }

        } catch (Exception e) {
            // Print message on the console
            System.out.println("Error Occurred!!!! \n" + e.getMessage());
        }
    }

    // Method to visualize a decision tree
    public static void visualizeTree(RandomTree tree) throws Exception {
        // Create a TreeVisualizer instance
        TreeVisualizer treeVisualizer = new TreeVisualizer(null, tree.graph(), new PlaceNode2());

        // Register a listener for the TreeVisualizer
        TreeVisualizerListener listener = new TreeVisualizerListener() {
            public void beforeExpand(TreeVisualizerEvent e) {
                // Handle tree expansion if needed
            }

            public void afterExpand(TreeVisualizerEvent e) {
                // Handle tree expansion if needed
            }

            public void beforeCollapse(TreeVisualizerEvent e) {
                // Handle tree collapse if needed
            }

            public void afterCollapse(TreeVisualizerEvent e) {
                // Handle tree collapse if needed
            }
        };

        // Add the listener to the TreeVisualizer
        treeVisualizer.addTreeVisualizerListener(listener);

        // Display the tree visualization
        treeVisualizer.setVisible(true);
    }
}
