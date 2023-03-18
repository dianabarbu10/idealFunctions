# importing the modules
from bokeh.plotting import figure, show


def plot_data_train_vs_ideal(df_train, ideal_functions):
    no_of_y = len(df_train.columns)
    for i in range(1, no_of_y):
        # instantiating the figure object with a dynamic title
        graph_title = "Data graph for x and y{}".format(i)
        graph = figure(title=graph_title)
        # plotting the training data
        y_value = "y{}".format(i)
        graph.scatter(df_train.x, df_train[y_value], color="purple")

        # plotting the corresponding ideal function data
        graph.scatter(ideal_functions[i - 1].x_values, ideal_functions[i - 1].y_values, color="pink")
        # displaying the model
        show(graph)


def plot_data_ideal_vs_test(df_test):
    # instantiating the figure object
    graph_title = "Data graph for test vs ideal"
    graph = figure(title=graph_title)
    # plotting the test data
    graph.scatter(df_test.x, df_test.y, color="purple")
    # plotting the function estimated data with the chosen function
    graph.scatter(df_test.x, df_test.y_hat, color="pink")
    show(graph)
