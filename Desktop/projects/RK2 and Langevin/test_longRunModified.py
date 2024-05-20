import longRunModified
import unittest

class TesterForFuns(unittest.TestCase):
    def test_is_greater_than_zero(self):
        self.assertEqual(longRunModified.is_greater_than_zero(3), 1)
        self.assertEqual(longRunModified.is_greater_than_zero(-1), 0)

