import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
 import java.io.InputStream;
 import java.io.FileInputStream;
import opennlp.tools.doccat.DoccatFactory;
import opennlp.tools.doccat.DoccatModel;
import opennlp.tools.doccat.DocumentCategorizer;
import opennlp.tools.doccat.DocumentCategorizerME;
import opennlp.tools.doccat.DocumentSample;
import opennlp.tools.doccat.DocumentSampleStream;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.MarkableFileInputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.TrainingParameters;
 
/**
 * oepnnlp version 1.7.2
 * Training of Document Categorizer using Maximum Entropy Model in OpenNLP
 * @author www.tutorialkart.com
 */
public class SD {
 
    public static void main(String[] args) {
 
        try {
       
			
        	String[] INPUTS={
        			"Will a COVID-19 vaccine alter my DNA?",
        			"Can a COVID-19 vaccine make me sick with COVID-19?",
        			"Can receiving a COVID-19 vaccine cause you to be magnetic?",
        			"Can hot drinks stop COVID-19?",
        			"Can COVID-19 vaccines cause variants?",
        			"Can drinking alcohol cure or prevent COVID-19?",
        			
        			"Are the COVID-19 vaccines safe?",
        			"Can COVID-19 be passed on in warm sunny weather?",
        			"Should I use a strong disinfectant to clean my hands and body to protect myself from COVID-19?",
        			
        			"Patients should avoid taking ibuprofen to relieve pain and fever associated with COVID-19 infections",
        			"People who have survived the COVID-19 coronavirus disease can become reinfected by the virus",
        			"Pressure builds on Hong Kong hotels to turn away guests from mainland",
        			"Tens of millions of masks on way to Hong Kong and prisoners will work around the clock to make more as city confirms its 12th case",
        			"Japan bans coronavirus-infected travellers after outcry over lax response China goes global in search for protective suits, masks and goggles as coronavirus infections begin to take off"

        			};
			        	
        	
       
         File test = new File("model"+File.separator+"en-covid-classifier-maxent.bin");
        	  String classificationModelFilePath = test.getAbsolutePath();
        	  DocumentCategorizer doccat = new DocumentCategorizerME(new DoccatModel(
             	      new FileInputStream(classificationModelFilePath)));
            
        	  
        	  for (String IN : INPUTS) {
 	
        		  String[] docWords = IN.replaceAll("[^A-Za-z]", " ").split(" ");
        		  
        		  for (String WORD:docWords) {
        			  System.out.print(WORD);
        		  	}
                  double[] aProbs = doccat.categorize(docWords);
       
                  // print the probabilities of the categories
               
                  System.out.println("\n"+IN);
                  
                  for(int i=0;i<doccat.getNumberOfCategories();i++){
                      System.out.println(doccat.getCategory(i)+" : "+aProbs[i]);
                  }
                  
                  System.out.println(doccat.getBestCategory(aProbs)+" : is the predicted category for the given sentence.");
                  
                  System.out.println("---------------------------------");
       
               
        		  
        		}
        	  
        	  		
        	  
        	 
                     
        }
        catch (IOException e) {
            System.out.println("An exception in reading the training file. Please check.");
            e.printStackTrace();
        }
    }
}

