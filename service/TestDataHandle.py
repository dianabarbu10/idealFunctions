import math

from pandas import DataFrame

from model.TestDataResults import TestDataResults
from service.DataManipulation import DataManipulation


class TestDataHandle(DataManipulation):

    def process(self, df_test, ideal_functions):
        results = []
        x_values = df_test.x.values
        y_values = df_test.y.values
        for i in range(len(x_values)):
            x_test = x_values[i]
            y_test = y_values[i]
            test_data_results = self.calc_deviation_test(ideal_functions, x_test, y_test)
            # sort the results by deviation (ascending)
            test_data_results.sort(key=lambda e: e.deviation)
            # only add to the final results the test data where the deviation is smaller than sqrt 2
            if test_data_results[0].deviation <= math.sqrt(2):
                results.append([x_test, y_test, test_data_results[0].deviation, test_data_results[0].y_number,
                                test_data_results[0].y_hat])
        # return a DataFrame
        df = DataFrame(results, columns=['x', 'y', 'Delta y', 'No. ideal func', 'y_hat'])
        return df

    '''
        Calculates the deviations between y_test and an estimated y_hat from all ideal functions
            Parameters:
                ideal_functions (IdealFunction[]): a list of all the ideal functions chosen, as IdealFunction objects
                x_test, y_test (number): the x and y we want to calculate for
            Returns: a list of TestDataResults 
    '''
    def calc_deviation_test(self, ideal_functions, x_test, y_test):
        test_data_results = []
        # loop through each ideal function
        for i in range(len(ideal_functions)):
            # calculate the coefficients for the function
            x = ideal_functions[i].x_values.reshape(-1, 1)
            y = ideal_functions[i].y_values
            model = self.find_coefficients(x, y)
            alpha = model.intercept_
            beta = model.coef_[0]
            # calculate the estimated y_hat based on the coefficients
            y_hat = self.calc_y_hat(x_test, alpha, beta)
            # calculate the squared deviation between the measured y and the estimated y
            deviation = self.calc_deviation(y_test, y_hat)
            # store the x,y,deviation, ideal function number, and the estimated y for later use
            test_data_results.append(TestDataResults(x_test, y_test, deviation, ideal_functions[i].y_number, y_hat))
        return test_data_results
