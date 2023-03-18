import unittest

import pandas

from service.TrainDataHandle import TrainDataHandle


class MyTestCase(unittest.TestCase):
    training_data = {'x': [1, 2, 3, 4], 'y1': [10, 20, 30, 40], 'y2': [100, 200, 300, 400]}
    ideal_functions = {'x': [5, 6, 7, 8], 'y1': [48, 62, 71, 83], 'y2': [25, 22, 23, 100], 'y3': [520, 600, 710, 895]}
    df_train = pandas.DataFrame(training_data)
    df_ideal = pandas.DataFrame(ideal_functions)
    train_data = TrainDataHandle()

    def test_process(self):
        ideal_functions = self.train_data.process(self.df_train, self.df_ideal)
        self.assertEqual(2, len(ideal_functions))
        self.assertEqual(1, ideal_functions[0].y_number)
        self.assertEqual(3, ideal_functions[1].y_number)

    def test_find_best_fit(self):
        x = self.df_train.x.values.reshape(-1, 1)
        y = self.df_train.y1.values
        model = self.train_data.find_coefficients(x, y)
        res = self.train_data.find_best_fit(self.df_ideal, model.intercept_, model.coef_[0])
        self.assertEqual(res.y_number, 1)
        self.assertEqual(res.x_values.all(), self.df_ideal.x.values.all())

    def test_find_best_fit_2(self):
        x = self.df_train.x.values.reshape(-1, 1)
        y = self.df_train.y2.values
        model = self.train_data.find_coefficients(x, y)
        res = self.train_data.find_best_fit(self.df_ideal, model.intercept_, model.coef_[0])
        self.assertEqual(res.y_number, 3)
        self.assertEqual(res.x_values.all(), self.df_ideal.x.values.all())

    def test_calc_y_hat_values(self):
        res = self.train_data.calc_y_hat_values(self.df_train.y1, 3, 2)
        expected = [23, 43, 63, 83]
        self.assertEqual(expected, res)

    def test_calc_y_deviation_squared_sum(self):
        y_hat_values = [12, 22, 32, 42]
        res = self.train_data.calc_y_deviation_squared_sum(self.df_train.y1, y_hat_values)
        self.assertEqual(16, res)


if __name__ == '__main__':
    unittest.main()
