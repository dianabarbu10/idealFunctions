import csv

import pandas as pd
import sqlalchemy as db


# create a class with a function for reading the files and saving them in an x and y variables
class Database:
    engine = None
    connection = None
    metadata = None

    def __init__(self):
        self
        self.engine = db.create_engine('sqlite:///ideal_functions.db', echo=False)
        self.connection = self.engine.connect()
        self.metadata = db.MetaData()

    '''
        Reads a document and saves the data to the database, then returns its content
            Parameters:
                document_name (string): the name of the document to read and parse
                table_name (string): the name of the table where the data should be saved
            Returns: a Pandas DataFrame of the content of the file
    '''
    def read_and_save(self, document_name, table_name):
        df = self.read(document_name)
        self.save(table_name, df)
        return df

    '''
        Reads a csv document and returns its content
            Parameters:
                document_name (string): the name of the document to read and parse
            Returns: a Pandas DataFrame of the content of the file
    '''
    def read(self, document_name):
        return pd.read_csv(document_name)

    '''
        Saves a DataFrame to the database
            Parameters:
                table_name (string): the name of the table where the data should be saved
                df (Pandas.DataFrame): the data to save to the database
            Returns: a Pandas DataFrame of the content of the file
    '''
    def save(self, table_name, df):
        df.to_sql(table_name, con=self.engine, if_exists='replace', index=False)

    '''
        Reads from the database and returns the data in the table
            Parameters:
                table_name (string): the name of the table from where the data should be read
            Returns: a Pandas DataFrame of the content of the table
    '''
    def read_from_database(self, table_name):
        df = pd.read_sql_table(table_name, self.connection)
        return df
