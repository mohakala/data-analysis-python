""" From http://spark.apache.org/docs/latest/quick-start.html
"""

from pyspark import SparkContext
#import unittest

def wordFrequencies(fileName):
    sc = SparkContext("local")
    textFile=sc.textFile(fileName).cache()
    wordCounts = textFile.flatMap(lambda line: line.split()).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a+b)
    freqlist=wordCounts.collect()
    ninstances=wordCounts.count()
#    a=wordCounts.collect()
#    b=wordCounts.count()
    sc.stop()
#    return(a,b)
    return(freqlist,ninstances)

def countWord(fileName,stringName):
    sc = SparkContext("local")
    textFile=sc.textFile(fileName).cache()
    nstrings=textFile.filter(lambda s: stringName in s).count()
    sc.stop()
    return(nstrings)

def test_wordFrequencies():
    freqlist,ninstances = wordFrequencies('unittest_inputref.md')
    assertEqual(ninstances,260)

    
if __name__ == '__main__':
    fileName="README.md"

    
    # word frequences
    # test_wordFrequencies()
    
    freqlist,ninstances = wordFrequencies(fileName)
    print('Number of different words:',ninstances)
    if(False): print(freqlist)

    # number of occurences of a string
    stringToFind='a'
    print('String to find:',stringToFind)
    print("# lines with the string: %i" % (countWord(fileName,stringToFind)))



