import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.trees.J48;
import weka.core.FastVector;
import weka.core.Instances;
 
public class MLAssignment1 {
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
		
		try {
			InputStream fis = new MLAssignment1().getClass().getResourceAsStream(filename);
			inputReader = new BufferedReader(new InputStreamReader(fis));
		} catch(Exception e) {
			e.printStackTrace();
		}
		return inputReader;
	}
 
	public static double calculateAccuracy(FastVector predictions) {
		double correct = 0;
 
		for (int i = 0; i < predictions.size(); i++) {
			NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
			if (np.predicted() == np.actual()) {
				correct++;
			}
		}
 
		return 100 * correct / predictions.size();
	}
 
	public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds) {
		Instances[][] split = new Instances[2][numberOfFolds];
 
		for (int i = 0; i < numberOfFolds; i++) {
			split[0][i] = data.trainCV(numberOfFolds, i);
			split[1][i] = data.testCV(numberOfFolds, i);
		}
 
		return split;
	}
 
	public static void main(String[] args) throws Exception {
		BufferedReader datafile = readDataFile("datatraining.arff");
		BufferedReader testfile = readDataFile("datatest_new.arff");
		Instances trainingInstances = new Instances(datafile);
		Instances testInstances = new Instances(testfile);
		trainingInstances.setClassIndex(trainingInstances.numAttributes() - 1);
		testInstances.setClassIndex(testInstances.numAttributes()-1);

		// Use a set of classifiers
		Classifier[] models = { 
				new J48()
		};
 
		// Train and Test for each model
		for (int j = 0; j < models.length; j++) {
			
			//Training Eval..
			Evaluation trainingEval = null;
			trainingEval = new Evaluation(trainingInstances);
			models[j].setOptions(weka.core.Utils.splitOptions("-R -N 3 -Q 1 -M 2"));
			models[j].buildClassifier(trainingInstances);
			trainingEval.crossValidateModel(models[j], trainingInstances, 10, new Random());
			trainingEval.evaluateModel(models[j], trainingInstances);
			
			System.out.println(trainingEval.toClassDetailsString());
			System.out.println(trainingEval.toSummaryString());
			//Test Eval..
			Evaluation testEval = new Evaluation(testInstances);
			testEval.evaluateModel(models[j], testInstances);

			System.out.println(testEval.toClassDetailsString());
			System.out.println(testEval.toSummaryString());
			
				
		}
		
	}
}