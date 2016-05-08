import unittest
from textfileop_spark import *

class TestMethods(unittest.TestCase):

    def test_wordFrequencies(self):
        # For reference file, must return 259 instances
        freqlist,ninstances = wordFrequencies('unittest_inputref.md')
        self.assertEqual(ninstances, 259)

    def test_countString(self):
        # For reference file, must return 9 occurences for 'for'
        self.assertEqual(countString('unittest_inputref.md','for'),9)

if __name__ == '__main__':

    unittest.main()
