__author__ = "Kashmira Dolas"
"""
language:   Python3
file:   DataPreperation.py
author: kashmira Dolas
description: Load data and prepare it by converting the categorial
attributes to numeric
"""

import csv


def DataPrep(filein, fileout):
    """
    Convert the categorical variables to numeric values for building the data.
    :param filein: Input file
    :param fileout: Output file
    :return: None
    """
    dataset = []
    with open(filein) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=';')
        for row in readCSV:
            dataset.append(row)

    # Build a dictionary to convert categorical values to numeric values.
    job = {"admin.": 1,
           "unknown": 2,
           "unemployed": 3,
           "management": 4,
           "housemaid": 5,
          "entrepreneur": 6,
           "student": 7,
           "blue-collar": 8,
           "self-employed": 9,
          "retired": 10,
           "technician": 11,
           "services": 12}
    marital = {"married": 1, "divorced": 2, "single": 3}
    education = {"unknown": 1, "secondary": 2, "primary": 3, "tertiary": 4}
    binary = {"yes": 1, "no": 0}
    contact = {"unknown": 1, "telephone": 2, "cellular": 3}
    poutcome = {"unknown": 1, "other": 2, "failure": 3, "success": 4}
    month = {"jan":1,"feb":2,"mar":3,"apr": 4, "may":5,"jun":6, "jul":7,
             "aug":8, "sep":9,\
            "oct":10, "nov":11, "dec":12}

    # Use the dictionary to convert the values
    for i in dataset[1:]:
        i[1] = job[i[1]]
        i[2] = marital[i[2]]
        i[3] = education[i[3]]
        i[4] = binary[i[4]]
        i[6] = binary[i[6]]
        i[7] = binary[i[7]]
        i[8] = contact[i[8]]
        i[10] = month[i[10]]
        i[15] = poutcome[i[15]]
        i[16] = binary[i[16]]

    # Write the corrected files to a new csv file to be used for training and
    #  testing.
    with open(fileout, 'w') as f:
        write = csv.writer(f, delimiter=",")
        for row in dataset:
            write.writerow(row)


def main():
    """
    Main Function
    :return: None
    """
    DataPrep(input("Please enter the name of the input file without "
                   ".csv ")+".csv",input("Please enter the name of the output "
                                        "file without .csv ")+".csv")

if __name__ == '__main__':
    main()
