import random

class node:

    # Specifying the random seed to gain consistent results
    random.seed(3)

    # The six input weights of the node
    weightOne = float(random.uniform(-1, 1))
    weightTwo = float(random.uniform(-1, 1))
    weightThree = float(random.uniform(-1, 1))
    weightFour = float(random.uniform(-1, 1))
    weightFive = float(random.uniform(-1, 1))
    weightSix = float(random.uniform(-1, 1))

    # Specifications in the network
    learningRate = 0.1
    bias = 0


    def __init__(self, identification):

        self.type = identification
        self.saveStartingWeights()


    def getActualOutput(self, vlaue1, vlaue2, vlaue3, vlaue4, vlaue5, vlaue6):

        # Get the weighted sum of the node inputs
        total = (self.weightOne * vlaue1) + (self.weightTwo * vlaue2) + (self.weightThree * vlaue3) + (self.weightFour * vlaue4) + (self.weightFive * vlaue5) + (self.weightFour * vlaue6) + self.bias

        # Determine if the node thinks its the correct value (Step Function)
        if total >= 0:
            return 1
        else:
            return 0


    def teach(self, vlaue1, vlaue2, vlaue3, vlaue4, vlaue5, vlaue6, identification):

        actualOutput = self.getActualOutput(vlaue1, vlaue2, vlaue3, vlaue4, vlaue5, vlaue6)

        # Compare the node's set type to the identification from the data row to get desired output
        if self.type == identification:
            desiredOutput = 1
        else:
            desiredOutput = 0

        # Adjust the weights based on the error correction learning model
        self.weightOne = self.weightOne + ((desiredOutput - actualOutput) * self.learningRate * vlaue1)
        self.weightTwo = self.weightTwo + ((desiredOutput - actualOutput) * self.learningRate * vlaue2)
        self.weightThree = self.weightThree + ((desiredOutput - actualOutput) * self.learningRate * vlaue3)
        self.weightFour = self.weightFour + ((desiredOutput - actualOutput) * self.learningRate * vlaue4)
        self.weightFive = self.weightFive + ((desiredOutput - actualOutput) * self.learningRate * vlaue5)
        self.weightSix = self.weightSix + ((desiredOutput - actualOutput) * self.learningRate * vlaue6)

        self.bias += ((desiredOutput - actualOutput) * self.learningRate)



    def saveStartingWeights(self):

        self.star1 = self.weightOne
        self.star2 = self.weightTwo
        self.star3 = self.weightThree
        self.star4 = self.weightFour
        self.star5 = self.weightFive
        self.star6 = self.weightSix


