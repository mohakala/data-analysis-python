import unittest
from mfunc import *

class TestMethods(unittest.TestCase):

    def test_pcnt(self):
        # For 8.1,12.8 must return string '8.1/12=63.28%'
        # print(pcnt(8.1,12.8))
        self.assertEqual(pcnt(8.1,12.8), '8.1/12.8=63.28%')

if __name__ == '__main__':

    unittest.main()
