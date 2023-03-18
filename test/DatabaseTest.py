import unittest

import pandas

from service.Database import Database


class MyTestCase(unittest.TestCase):
    database = Database()

    def test_save(self):
        training_data = {'x': [1, 2, 3, 4], 'y1': [10, 20, 30, 40], 'y2': [100, 200, 300, 400]}
        df_train = pandas.DataFrame(training_data)
        self.database.save('test', df_train)
        result = self.database.read_from_database('test')
        self.assertEqual(df_train.x.values.all(), result.x.values.all())


if __name__ == '__main__':
    unittest.main()
