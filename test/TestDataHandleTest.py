import unittest

import numpy as np
import pandas

from model.IdealFunction import IdealFunction
from service.TestDataHandle import TestDataHandle


class MyTestCase(unittest.TestCase):
    test_data = {'x': [1, 2, 3, 4], 'y': [16, 12, 138, 11]}
    df_test = pandas.DataFrame(test_data)
    ideal_functions = [IdealFunction(np.array([5, 6, 7, 8]), np.array([27, 32, 37, 42]), 1, 0.23),
                       IdealFunction(np.array([5, 6, 7, 8]), np.array([13, 15, 17, 19]), 2, 1.23),
                       IdealFunction(np.array([5, 6, 7, 8]), np.array([57, 67, 77, 87]), 3, 2.23)]
    test_data = TestDataHandle()

    def test_calc_deviation_test(self):
        res = self.test_data.calc_deviation_test(self.ideal_functions, 1, 16)
        self.assertEqual(3, len(res))
        self.assertEqual(81, res[0].deviation)
        self.assertEqual(121, res[1].deviation)
        self.assertEqual(1, res[2].deviation)

    def test_process(self):
        res = self.test_data.process(self.df_test, self.ideal_functions)
        self.assertEqual(3, len(res))
        self.assertEqual(3, res['No. ideal func'].values[0])
        self.assertEqual(1, res['No. ideal func'].values[1])
        self.assertEqual(2, res['No. ideal func'].values[2])


if __name__ == '__main__':
    unittest.main()
