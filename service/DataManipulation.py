from sklearn.linear_model import LinearRegression


class DataManipulation:
    """
        Calculates the linear regression coefficients for x and y
            Parameters:
                x (2D array)
                y (array)
        Returns: a LinearRegression model
    """
    def find_coefficients(self, x, y):
        model = LinearRegression()
        model.fit(x, y)
        return model

    """
        Calculates the estimated y value based on coefficients
            Parameters:
                x (number): the initial value
                alpha, beta (number): the coefficients of the linear regression
        Returns: the estimated y value
    """
    def calc_y_hat(self, x, alpha, beta):
        return alpha + (x * beta)

    """
        Calculates the squared deviation between the actual and the estimated value
            Parameters:
                y (number): the actual value
                y_hat (number): the estimated value
    """
    def calc_deviation(self, y, y_hat):
        return (y - y_hat) ** 2

