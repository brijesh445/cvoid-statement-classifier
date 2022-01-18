import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
 
import opennlp.tools.doccat.DoccatFactory;
import opennlp.tools.doccat.DoccatModel;
import opennlp.tools.doccat.DocumentCategorizer;
import opennlp.tools.doccat.DocumentCategorizerME;
import opennlp.tools.doccat.DocumentSample;
import opennlp.tools.doccat.DocumentSampleStream;
import opennlp.tools.ml.AbstractTrainer;
import opennlp.tools.ml.naivebayes.NaiveBayesTrainer;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.MarkableFileInputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.TrainingParameters;
 
public class NAIVE {
 
    public static void main(String[] args) {
 
  		
    	final String  ModelName="en-covid-classifier-naive-bayes.bin.bin";
    	final String  TrainFileName="motheroftraindataP90.train";
    	
        try {
            // read the training data
            InputStreamFactory dataIn = new MarkableFileInputStreamFactory(new File("train"+File.separator+TrainFileName));
            ObjectStream lineStream = new PlainTextByLineStream(dataIn, "UTF-8");
            ObjectStream sampleStream = new DocumentSampleStream(lineStream);
 
            // define the training parameters
            TrainingParameters params = new TrainingParameters();
            params.put(TrainingParameters.ITERATIONS_PARAM, 200+"");
            params.put(TrainingParameters.CUTOFF_PARAM, 0+"");
            params.put(AbstractTrainer.ALGORITHM_PARAM, NaiveBayesTrainer.NAIVE_BAYES_VALUE);
 
            // create a model from traning data
            DoccatModel model = DocumentCategorizerME.train("en", sampleStream, params, new DoccatFactory());
            System.out.println("\nModel is successfully trained.");
 
            // save the model to local
            BufferedOutputStream modelOut = new BufferedOutputStream(new FileOutputStream("model"+File.separator+ModelName));
            model.serialize(modelOut);
            System.out.println("\nTrained Model is saved locally at : "+"model"+File.separator+ModelName);
 
            // test the model file by subjecting it to prediction
            DocumentCategorizer doccat = new DocumentCategorizerME(model);
            String[] docWords = "Are the COVID-19 vaccines safe?".replaceAll("[^A-Za-z]", " ").split(" ");
            double[] aProbs = doccat.categorize(docWords);
 
            // print the probabilities of the categories
            System.out.println("\n---------------------------------\nCategory : Probability\n---------------------------------");
            for(int i=0;i<doccat.getNumberOfCategories();i++){
                System.out.println(doccat.getCategory(i)+" : "+aProbs[i]);
            }
            System.out.println("---------------------------------");
 
            System.out.println("\n"+doccat.getBestCategory(aProbs)+" : is the predicted category for the given sentence.");
        }
        catch (IOException e) {
            System.out.println("An exception in reading the training file. Please check.");
            e.printStackTrace();
        }
    }
}