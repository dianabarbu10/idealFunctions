# idealFunctions
This program reads from 3 csv files: 'train', 'ideal', 'test'
The 'train' contains 4 x-y functions which are used to train an algorythm. 
Then, using linear regression, for each 'train' function we pick the best matching 'ideal' function out of 50 provided
We then use the 'test' data to pick the best 'ideal' function out of the 4, for each x-y pair in the 'test' file
The 'train' and 'ideal' are saved to the database
Each 'train' function and its corresponding 'ideal' function are displayed using Bokeh
The 'test' x-y values are saved to the database with the number of the ideal function chosen for each row, as well as the deviation.
The 'test' data and the ideal function results are displayed using Bokeh
