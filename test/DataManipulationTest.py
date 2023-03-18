import unittest

import numpy as np

from service.DataManipulation import DataManipulation


class MyTestCase(unittest.TestCase):
    manipulation = DataManipulation()

    def test_find_coefficients(self):
        x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
        y = np.array([5, 20, 14, 32, 22, 38])
        model = self.manipulation.find_coefficients(x, y)
        self.assertEqual(model.intercept_, 5.633333333333329)  # add assertion here
        self.assertEqual(model.coef_[0], 0.54)

    def test_calc_y_hat(self):
        res = self.manipulation.calc_y_hat(2,10,2)
        self.assertEqual(14,res)


    def test_calc_deviation(self):
        res = self.manipulation.calc_deviation(10,12)
        self.assertEqual(4,res)
if __name__ == '__main__':
    unittest.main()
