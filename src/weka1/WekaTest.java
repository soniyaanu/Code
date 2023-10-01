package weka1;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
public class WekaTest {

	public static void main(String args[])throws IOException,Exception{
		Instances instances=new Instances(new BufferedReader(new FileReader("C:\\Users\\hella\\Desktop\\results\\avro_result.arff")));
		instances.setClassIndex(instances.numAttributes()-1);
		double precision=0;
		double recall=0;
		double fmeasure=0;
		double errro=0;
		double auc=0;
		double mcc=0;
		double g_mean=0;
		int size=instances.numInstances()/10;
		int begin=0;
		int end=size-1;
		for(int i=1;i<=10;i++){
			System.out.println("Iteration:"+i);
			Instances trainingInstances=new Instances(instances);
			Instances testingInstances=new Instances(instances,begin,(end-begin));
			for(int j=0;j<(end-begin);j++) {
				trainingInstances.delete(begin);
			}
			J48 j48Classifier = new J48();
            j48Classifier.buildClassifier(trainingInstances);
            Evaluation evaluation
			= new Evaluation(testingInstances);
            evaluation.crossValidateModel(j48Classifier, testingInstances, end, new Random());
            System.out.println("weighted precision = "+evaluation.weightedPrecision());
            System.out.println("weighted recall = "+evaluation.weightedRecall());
            System.out.println("weighted fmeasuren = "+evaluation.weightedFMeasure());
            precision+=evaluation.weightedPrecision();
            recall+=evaluation.weightedRecall();
            fmeasure+=evaluation.weightedFMeasure();
            begin=end+1;
            end+=size;
            if(i==(9)) {
            	end=instances.numInstances();
            }
			}
		System.out.println("precison:"+precision/10.0);
		System.out.println("recall:"+recall/10.0);
		System.out.println("fmeasure:"+fmeasure/10.0);
		
		}
}
			

	
		
	
