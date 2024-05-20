import longRun
import unittest
import numpy as np

class TestGetTimesAndDurations(unittest.TestCase):
    def setUp(self):
        self.U_History = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0])
        self.time_values = np.arange(0, len(self.U_History))
        self.total_time = 10

    def test_getTimesAndDurations(self):
        reversal_array, duration_before_reversals = longRun.getTimesAndDurations(self.U_History, self.time_values, self.total_time)

        expected_reversal_array = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        expected_duration_before_reversals = [1, 1, 1, 1, 1, 1, 1, 1, 1]

        self.assertEqual(reversal_array, expected_reversal_array)
        self.assertEqual(duration_before_reversals, expected_duration_before_reversals)


class TestNewCountReversal(unittest.TestCase):
    
    def test_newCountReversal(self):
        U_History = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0])
        time_values = np.arange(0, len(U_History))

        expected_reversal_array = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        reversal_array = longRun.newCountReversal(U_History, time_values)

        self.assertEqual(reversal_array, expected_reversal_array)

    def test_no_reversals(self):
        U_History = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        time_values = np.arange(0, len(U_History))

        expected_reversal_array = []

        reversal_array = longRun.newCountReversal(U_History, time_values)

        self.assertEqual(reversal_array, expected_reversal_array)

    def test_all_reversals(self):
        U_History = np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
        time_values = np.arange(0, len(U_History))

        expected_reversal_array = [1, 2, 3, 4, 5]

        reversal_array = longRun.newCountReversal(U_History, time_values)

        self.assertEqual(reversal_array, expected_reversal_array)

if __name__ == '__main__':
    unittest.main()

