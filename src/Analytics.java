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
		
		
		
		String[] Models={"en-covid-classifier-maxent.bin","en-covid-classifier-naive-bayes.bin","en-covid-classifier-ngram.bin"};
		for (String Model : Models) 
		{
		   calacc(Model);
		}
	}
	
	
	
	static void calacc(String ModelName) {
		String line = "";
		String splitBy = ",";
		double Right_Predections=0;
		double Wrong_Predections=0;
		double Total_Predections=0;
		int NumberofTrues=0;
		int NumberofFalse=0;
		int NumberofUidentified=0;
		
		double TrueOnly=0;
		double FalseOnly=0;
		double UnindetifiedOnly=0;
	
		
	
		try {
			
			
			//parsing a CSV file into BufferedReader class constructor  
			BufferedReader br = new BufferedReader(new FileReader("test" + File.separator + "TestDataP10.csv"));
		
			//MAXNET
		     File Model = new File("model"+File.separator+ModelName);
	       	 String classificationModelFilePath = Model.getAbsolutePath();
	       	 DocumentCategorizer cat = new DocumentCategorizerME(new DoccatModel(
	            	      new FileInputStream(classificationModelFilePath)));
		
			 
			while ((line = br.readLine()) != null) // returns a Boolean value
			{
			
				String[] DATA = line.split(splitBy); // use comma as separator
				
		
				//read the input
				String[] docWords = DATA[1].replaceAll("[^A-Za-z]", " ").split(" ");
      		  
				//generate the probability
				double[] MaxNetProbs = cat.categorize(docWords);
				
				
			
				
				String InputCategory = new String(DATA[0]);
				
			
				
				switch (InputCategory) {
				case "T":
					NumberofTrues++;
					break;
				
				case "F":
					NumberofFalse++;
					break;					
				default:
					NumberofUidentified++;
					break;
				}
				
				
				String BestCategory = new String(cat.getBestCategory(MaxNetProbs));
				 
				 if(InputCategory.equals(BestCategory)) {
					//increase right prediction counter by 1
					 Right_Predections++;
					 
					 if(InputCategory.equals("T") && BestCategory.equals("T")) {
						 TrueOnly++;
					 }
					 if(InputCategory.equals("F") && BestCategory.equals("F")) {
						 FalseOnly++;
					 }
					 if(InputCategory.equals("U") && BestCategory.equals("U")) {
						 UnindetifiedOnly++;
					 }
				 
				 }else {
					 //if  wrong prediction increase wrong prediction counter by 1
					 Wrong_Predections++;
				 }
				 //increase total prediction counter 
				 Total_Predections++;
		
			
			}
					
			
			 System.out.println("-------------------------------"+  ModelName  +"----------------------------"+"\n");
			 System.out.println("  ");
			
			//print the prediction Data
			System.out.println("Right Predections = "+Right_Predections+" Wrong Predections = "+Wrong_Predections+"  Total Predections  = "+Total_Predections + "\n");
			
			
			
			System.out.println("True Only :- "+TrueOnly+"  False Only :- "+FalseOnly+" Unidentified Only :- "+UnindetifiedOnly+"\n");

			System.out.println("True :- "+NumberofTrues+"  False :- "+NumberofFalse+" Unidentified :- "+NumberofUidentified+"\n");
			
			//calculate the prediction
			double Acc=Right_Predections/Total_Predections;
			
			
			//for true
			double TPrecision=TrueOnly/(TrueOnly+NumberofTrues-TrueOnly);
			double FPrecision=FalseOnly/(FalseOnly+NumberofFalse-FalseOnly);
			double UPrecision=UnindetifiedOnly/(UnindetifiedOnly+NumberofUidentified-UnindetifiedOnly);

			
			//print the prediction value
			System.out.println("Accuracy :- "+Acc);
			System.out.println("Percision For True  :- " + TPrecision);
			System.out.println("Percision For False  :- " + FPrecision);
			System.out.println("Percision For UnIdentified  :- " + UPrecision);
			System.out.println("  ");
			
		} 
			
		catch (IOException e) {
			e.printStackTrace();
		}
	}
}