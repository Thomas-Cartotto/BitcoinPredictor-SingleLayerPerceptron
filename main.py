# Thomas Cartotto

import csv
from node import node

# Define three nodes. Each one represents a type of market action.

downNode = node("down")
upNode = node("up")
noChangeNode = node("noChange")



def main():

    # Sequence of actions the program will take
    trainANN()
    testANN()



def trainANN():

    epochCounter = 0

    while epochCounter < 1000:

        print(epochCounter)

        epochCounter += 1

        with open('trainDataPythonSLP.csv') as csvDataFile:

            trainingData = csv.reader(csvDataFile)

            for row in trainingData:

                # Get data from the training data
                value1 = float(row[0])
                value2 = float(row[1])
                value3 = float(row[2])
                value4 = float(row[3])
                value5 = float(row[4])
                value6 = float(row[5])
                type = str(row[6])

                downNode.teach(value1, value2, value3, value4, value5, value6, type)
                upNode.teach(value1, value2, value3, value4, value5, value6, type)
                noChangeNode.teach(value1, value2, value3, value4, value5, value6, type)



def testANN():

    with open('testDataPythonSLP.csv') as csvDataFile:

        csvReader = csv.reader(csvDataFile)

        # Counts the number of correct classifications
        correctCounter = 0

        # Identification Counters for each class
        idDown = 0
        idUP = 0
        idNo = 0

        # Correct prediction counters for each class
        correctDown = 0
        correctUP = 0
        correctNo = 0

        # Counters for occurrences of each class in the dataset
        actualDown = 0
        acutualUP = 0
        actualNo = 0


        for row in csvReader:

            value1 = float(row[0])
            value2 = float(row[1])
            value3 = float(row[2])
            value4 = float(row[3])
            value5 = float(row[4])
            value6 = float(row[5])
            correctAnswer = str(row[6])

            # Get the decision of each node
            downResult = downNode.getActualOutput(value1, value2, value3, value4, value5, value6)
            upResult = upNode.getActualOutput(value1, value2, value3, value4, value5, value6)
            noneResult = noChangeNode.getActualOutput(value1, value2, value3, value4, value5, value6)

            result = "NoID"


            if downResult == 1:
                result = "down"
                idDown += 1
            elif upResult == 1:
                result = "up"
                idUP += 1
            else:
                result = "noChange"
                idNo += 1


            if correctAnswer == result:
                correctCounter += 1

                if correctAnswer == "down":

                    correctDown += 1
                elif correctAnswer == "up":

                    correctUP += 1
                else:

                    correctNo += 1


            if correctAnswer == "down":
                actualDown += 1
            elif correctAnswer == "up":
                acutualUP += 1
            else:
                actualNo += 1

        percision = ((correctDown/idDown * 100) + (correctUP/idUP * 100) + (correctNo/idUP * 100)) / 3
        recall = ((correctDown/actualDown * 100) + (correctUP/acutualUP * 100) + (correctNo/actualNo * 100)) / 3


        print("\n\nAccuracy on test data-set: %d" %(correctCounter/526*100))
        print("\n\npercision on test data-set: %d" %(percision))
        print("\n\nrecall on test data-set: %d" %(recall))



if __name__ == "__main__":
    main()