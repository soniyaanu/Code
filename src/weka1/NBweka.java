package weka1;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
//import weka.classifiers.bayes.net.*;
import weka.core.Instances;

// Main class
// BreastCancer
public class NBweka{

	// Main driver method
	public static void main(String args[])
	{

		// Try block to check for exceptions
		try {

			// Create Naivebayes classifier by
			// creating object of NB class
			BayesNet BayesClassifier = new BayesNet();

			// Dataset path
			String Dataset
				= "C:\\Users\\hella\\Desktop\\results\\zookeeper_result.arff";

			// Creating bufferedreader to read the dataset
			BufferedReader bufferedReader
				= new BufferedReader(
					new FileReader(Dataset));

			// Create dataset instances
			Instances datasetInstances
				= new Instances(bufferedReader);

			// Set Target Class
			datasetInstances.setClassIndex(
				datasetInstances.numAttributes() - 1);

			// Evaluating by creating object of Evaluation
			// class
			Evaluation evaluation
				= new Evaluation(datasetInstances);

			// Cross Validate Model with 10 folds
			evaluation.crossValidateModel(
					BayesClassifier, datasetInstances, 10,
				new Random(1));
			 

			System.out.println(evaluation.toSummaryString(
				"\nResults", true));
			
	        

	        

			System.out.println("precision Class0:"+evaluation.precision(0));
			System.out.println("precision Class1:"+evaluation.precision(1));
			System.out.println("weighted precision = "+evaluation.weightedPrecision());
			System.out.println("recall Class0:"+evaluation.recall(0));
			System.out.println("recall Class1:"+evaluation.recall(1));
			System.out.println("weighted recall  = "+evaluation.weightedRecall());
			System.out.println("Fmeasure Class0:"+evaluation.fMeasure(0));
			System.out.println("Fmeasure Class1:"+evaluation.fMeasure(1));
			System.out.println("weighted Fmeasure  = "+evaluation.weightedFMeasure());
			System.out.println("MCC Class0:"+evaluation.matthewsCorrelationCoefficient(0));
			System.out.println("MCC Class1:"+evaluation.matthewsCorrelationCoefficient(1));
			System.out.println("weighted MCC  = "+evaluation.weightedMatthewsCorrelation());
			System.out.println("Roc area Class0:"+evaluation.areaUnderROC(0));
			System.out.println("Roc area Class1:"+evaluation.areaUnderROC(1));
			System.out.println("weighted Roc area  = "+evaluation.weightedAreaUnderROC());
			//System.out.println("Mcc:"+  evaluation.matthewsCorrelationCoefficient(1));
			//System.out.println("Roc area:"+  evaluation.areaUnderROC(1));
			//System.out.println("Recall:"+evaluation.recall(1));
			//System.out.println("Fmeasure:"+evaluation.fMeasure(1));
			//System.out.println("G-mean:"+Math.sqrt(evaluation.truePositiveRate(1)*evaluation.trueNegativeRate(1)));
			System.out.println("G-mean Class0:"+Math.sqrt(evaluation.truePositiveRate(0)*evaluation.trueNegativeRate(0)));
			System.out.println("G-mean Class1:"+Math.sqrt(evaluation.truePositiveRate(1)*evaluation.trueNegativeRate(1)));
			System.out.println("weighted G-mean :"+Math.sqrt(evaluation.weightedTruePositiveRate()*evaluation.weightedTrueNegativeRate()));
		}

		// Catch block to handle the exceptions
		catch (Exception e) {

			// Print message on the console
			System.out.println("Error Occurred!!!! \n"
							+ e.getMessage());
		}
	}
}
