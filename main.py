import DataQualityCheck
from service import DataVisualizer
from service.TestDataHandle import TestDataHandle
from service.TrainDataHandle import TrainDataHandle
from service.Database import Database

# initialize the reader
database = Database()

# initialize the data from the files
df_train = database.read_and_save('data/train.csv', 'train')
df_ideal = database.read_and_save('data/ideal.csv', 'ideal')
df_test = database.read('data/test.csv')

# check data quality
missing_data = False
DataQualityCheck.check_for_missing_data(df_train)
DataQualityCheck.check_for_missing_data(df_ideal)
DataQualityCheck.check_for_missing_data(df_test)

if not missing_data:
    # use the train data to find the ideal function for each set
    tran_data = TrainDataHandle()
    ideal_functions = tran_data.process(df_train, df_ideal)

    # compare the test data and pick the best ideal function
    test_data = TestDataHandle()
    test_data_results = test_data.process(df_test, ideal_functions)

    # visualize the training data vs the ideal function
    DataVisualizer.plot_data_train_vs_ideal(df_train, ideal_functions)
    # visualize the test data
    DataVisualizer.plot_data_ideal_vs_test(test_data_results)

    # save the test data to the database with the deviation and the picked ideal function
    test_data_results_2 = test_data_results[['x', 'y', 'Delta y', 'No. ideal func']].copy()
    database.save('test', test_data_results_2)




