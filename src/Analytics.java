import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;


import opennlp.tools.doccat.DoccatModel;
import opennlp.tools.doccat.DocumentCategorizer;
import opennlp.tools.doccat.DocumentCategorizerME;

public class Analytics {
	public static void main(String[] args) {
		String line = "";
		String splitBy = ",";
		double Right_Predections=0;
		double Wrong_Predections=0;
		double Total_Predections=0;
		
		try {
//parsing a CSV file into BufferedReader class constructor  
			BufferedReader br = new BufferedReader(new FileReader("train" + File.separator + "TestData.csv"));
			
			
			
		     File test = new File("model"+File.separator+"en-covid-classifier-maxent.bin");
	       	  String classificationModelFilePath = test.getAbsolutePath();
	       	  DocumentCategorizer doccat = new DocumentCategorizerME(new DoccatModel(
	            	      new FileInputStream(classificationModelFilePath)));
			
			
			
			while ((line = br.readLine()) != null) // returns a Boolean value
			{
			
				String[] DATA = line.split(splitBy); // use comma as separator
				//System.out.println(DATA[0]);
				//Count++;
		
				//read the input
				String[] docWords = DATA[1].replaceAll("[^A-Za-z]", " ").split(" ");
      		  
				//generate the probability
				double[] aProbs = doccat.categorize(docWords);
				 
				
				
				String InputCategory = new String(DATA[0]);
				String BestCategory = new String(doccat.getBestCategory(aProbs));
				 
				//System.out.println(DATA[0] + BestCategory );
				 
				 if(InputCategory.equals(BestCategory)) {
					//increase right prediction counter by 1
					 Right_Predections++;
				 }else {
					 //if not increase wrong prediction counter
					 Wrong_Predections++;
				 }
				 //increase total prediction counter 
				 Total_Predections++;
			}
			
			System.out.println("Right Predections = "+Right_Predections+" Wrong Predections = "+Wrong_Predections+"  Total Predections  = "+Total_Predections + "\t");
			
			
			double Acc=Right_Predections/Total_Predections;
			
			System.out.println("Accuracy :- "+Acc);
			
		} 
			
		catch (IOException e) {
			e.printStackTrace();
		}
	}
}