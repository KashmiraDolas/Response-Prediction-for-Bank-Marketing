========================
BUILD OUTPUT DESCRIPTION
========================

This is a readme file:
Contains information about how to configure and run the files for predicting the success of Bank Marketing.
1.	
	Configuration instructions

	1.1	
		Downloading the Dataset The data set can be found at https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip. Simply click on the link to download the dataset. ensure that the unzipped files are stored in the same directory as the project.
	1.2	
		Setting up Python  To run this project the packages required are: sklearn, csv, time. Please ensure that these packages are installed for error free execution.

2.	
	Operating instructions

	2.1	
		Preparing the dataset As described in the report, the data set downloaded from the link contains categorical values. For building the models we need numeric values. For this purpose, you need to run the DataPreparation.py file first.  Now this file will prompt you to enter the name of the input and output file. It will then convert the input file into the required format. Please ensure correct names of files are provided so that the program can successfully fetch those files. 
	2.2	
		Training Now that the data has been prepared, we can begin training. Simply run the Main.py file giving the name of the output file created by DataPreparation.py when prompted and see the results.

3.	A file manifest (list of files included) This zip contains 2 files: 
	
	DataPreparation.py 
	Main.py 
	
	Run in order

	
