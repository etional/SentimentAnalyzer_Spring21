import sys
import getopt
import os
import math
import operator

class Perceptron:
  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
    """
    def __init__(self):
      self.train = []
      self.test = []

  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []


  def __init__(self):
    """Perceptron initialization"""
    self.numFolds = 10
    self.docs = []
    self.bag_words = {}
    self.num_pos = 0
    self.num_neg = 0
  #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the Perceptron classifier 

  def classify(self, words):
    """ TODO
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """


    # Write code here
    sum_pos = 0.0
    sum_neg = 0.0
    for word in words:
      if word in self.bag_words.keys():
        if 'pos' in self.bag_words[word]['klass']:
          sum_pos = sum_pos + self.bag_words[word]['w0']
        elif 'neg' in self.bag_words[word]['klass']:
          sum_neg = sum_neg + self.bag_words[word]['w0']
    # sum_neg > sum_pos is the same as prob_neg > prob_pos
    if sum_neg > sum_pos:
      return 'neg'
    return 'pos'
  

  def addExample(self, klass, words):
    """
     * TODO
     * Train your model on an example document with label klass ('pos' or 'neg') and
     * words, a list of strings.
     * You should store whatever data structures you use for your classifier 
     * in the Perceptron class.
     * Returns nothing
    """

    # Write code here
    if klass == 'pos':
      self.num_pos = self.num_pos + 1
    else:
      self.num_neg = self.num_neg + 1

    doc = {}
    doc['klass'] = klass
    doc['words'] = words
    self.docs.append(doc)
    # initialize weight for each word
    for word in words:
      if word not in self.bag_words.keys():
        self.bag_words[word] = {}
        self.bag_words[word]['klass'] = []
        self.bag_words[word]['w0'] = 0.0
        self.bag_words[word]['wa'] = 0.0
      if klass not in self.bag_words[word]['klass']:
        if klass == 'pos':
          self.bag_words[word]['klass'].append('pos')
        else:
          self.bag_words[word]['klass'].append('neg')
    pass
  
  def train(self, split, iterations):
    """
    * TODO 
    * iterates through data examples
    * TODO 
    * use weight averages instead of final iteration weights
    """
    for example in split.train:
        words = example.words
        self.addExample(example.klass, words)
    
    c = 1
    for i in range(iterations):
      for doc in self.docs:
        if self.classify(doc['words']) != doc['klass']:
          y = 1
          if doc['klass'] == 'neg':
            y = -1
          for word in doc['words']:
            w_0 = self.bag_words[word]['w0']
            w_a = self.bag_words[word]['wa']
            self.bag_words[word]['w0'] = w_0 + y
            self.bag_words[word]['wa'] = w_a + c * y
        c = c + 1
    for word in self.bag_words.keys():
      self.bag_words[word]['w0'] = self.bag_words[word]['w0'] - float(self.bag_words[word]['wa'] / c)
      

  # END TODO (Modify code beyond here with caution)
  #############################################################################
  
  
  def readFile(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here, 
     * unless you don't like the way we segment files.
    """
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    result = self.segmentWords('\n'.join(contents)) 
    return result

  
  def segmentWords(self, s):
    """
     * Splits lines on whitespace for file reading
    """
    return s.split()

  
  def trainSplit(self, trainDir):
    """Takes in a trainDir, returns one TrainSplit with train set."""
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      example.klass = 'pos'
      split.train.append(example)
    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
      example.klass = 'neg'
      split.train.append(example)
    return split


  def crossValidationSplits(self, trainDir):
    """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
    splits = [] 
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      splits.append(split)
    return splits
  

def test10Fold(args):
  pt = Perceptron()
  
  iterations = int(args[1])
  splits = pt.crossValidationSplits(args[0])
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = Perceptron()
    accuracy = 0.0
    classifier.train(split,iterations)
  
    for example in split.test:
      words = example.words
      guess = classifier.classify(words)
      if example.klass == guess:
        accuracy += 1.0

    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print('[INFO]\tFold %d Accuracy: %f' % (fold, accuracy)) 
    fold += 1
  avgAccuracy = avgAccuracy / fold
  print('[INFO]\tAccuracy: %f' % avgAccuracy)
    
    
def classifyDir(trainDir, testDir,iter):
  classifier = Perceptron()
  trainSplit = classifier.trainSplit(trainDir)
  iterations = int(iter)
  classifier.train(trainSplit,iterations)
  testSplit = classifier.trainSplit(testDir)
  #testFile = classifier.readFile(testFilePath)
  accuracy = 0.0
  for example in testSplit.train:
    words = example.words
    guess = classifier.classify(words)
    if example.klass == guess:
      accuracy += 1.0
  accuracy = accuracy / len(testSplit.train)
  print('[INFO]\tAccuracy: %f' % accuracy)
    
def main():
  (options, args) = getopt.getopt(sys.argv[1:], '')
  
  if len(args) == 3:
    classifyDir(args[0], args[1], args[2])
  elif len(args) == 2:
    test10Fold(args)

if __name__ == "__main__":
    main()
