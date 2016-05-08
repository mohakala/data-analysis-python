""" From http://spark.apache.org/docs/latest/quick-start.html
"""

from pyspark import SparkContext
import sys
#import numpy as np
import psutil

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

def simpletest_wordFrequencies():
    freqlist,ninstances = wordFrequencies('unittest_inputref.md')
    if ninstances != 259:
        print('Simpletest not passed')
        sys.exit(1)
    else:
        print('Simpletest OK')
        return()
    
def countString(fileName,stringName):
    # Counts the number of occurrences of stringName
    sc = SparkContext("local")
    textFile=sc.textFile(fileName).cache()
    nstrings=textFile.filter(lambda s: stringName in s).count()
    sc.stop()
    return(nstrings)

      
    
if __name__ == '__main__':
  
    fileName="README.md"

    # | Some general tests
    # print(np.zeros(5)) # (numpy ok)
    
    # | wordFrequences
    simpletest_wordFrequencies() # Old; see tests_textfileop_spark
    
    freqlist,ninstances = wordFrequencies(fileName)
    print('Number of different words:',ninstances)
    if(False): print(freqlist)

    # | countString: number of occurrences of a string
    stringToFind='for'
    print("String --> %s <-- occurs %i times" % (stringToFind,countString(fileName,stringToFind)))



