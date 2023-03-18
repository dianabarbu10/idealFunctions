from model.IdealFunction import IdealFunction
from service.DataManipulation import DataManipulation


class TrainDataHandle(DataManipulation):
    def process(self, data_train, data_ideal):
        ideal_functions = []
        x = data_train['x'].to_numpy().reshape(-1, 1)
        # loop through each training data set
        for i in range(1, len(data_train.columns)):
            y = data_train["y{}".format(i)].to_numpy()
            # calculate the coefficients of the training function using linear regression
            lr = self.find_coefficients(x, y)
            # find the best fitting function from the ideal functions
            fit = self.find_best_fit(data_ideal, lr.intercept_, lr.coef_[0])
            ideal_functions.append(fit)
        return ideal_functions

    def find_best_fit(self, df_ideal, alpha, beta):
        ideal_functions = []
        x_values = df_ideal.x.values
        # loop through all ideal functions
        for i in range(1, len(df_ideal.columns)):
            y_values = df_ideal["y{}".format(i)].values
            # calculate the estimated ys of the ideal functions, using the coefficients of the training data
            y_hat_values = self.calc_y_hat_values(x_values, alpha, beta)
            # calculate the sum of the squared deviations between the measured y values of the ideal function and
            # the estimated ys
            deviation = self.calc_y_deviation_squared_sum(y_values, y_hat_values)
            # create a new IdealFunction object with the x and y values of the ideal function, its number
            # and the deviation
            ideal_functions.append(IdealFunction(x_values, y_values, i, deviation))
        ideal_functions.sort(key=lambda e: e.deviation)
        return ideal_functions[0]

    '''
        Calculates the estimated values 
            Parameters:
                x_values(number[])
                alpha, beta (number): the coefficients of the function
            Returns: the estimated values as a list
    '''
    def calc_y_hat_values(self, x_values, alpha, beta):
        y_hat_values = []
        for x in x_values:
            y_hat = self.calc_y_hat(x, alpha, beta)
            y_hat_values.append(y_hat)
        return y_hat_values

    '''
        Calculates the sum of the squared deviations between the actual and the estimated values
            Parameters:
                y_values (number[]): the actual value
                y_hat (number[]): the estimated value
            Returns: the sum of the squared deviations
    '''

    def calc_y_deviation_squared_sum(self, y_values, y_hat_values):
        deviations = []
        for i in range(0, len(y_values)):
            deviation = self.calc_deviation(y_values[i], y_hat_values[i])
            deviations.append(deviation)
        return sum(deviations)
