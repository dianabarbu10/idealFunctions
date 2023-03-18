# check if the datasets are complete
def check_for_missing_data(data_frame):
    if data_frame.isnull().values.any():
        print('Missing data')
        print(data_frame.info())
        return True
    else:
        print('No data missing')
        return False
