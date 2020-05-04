from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf
import time

################### start the clock ###################
start_time = time.time()

# Load dataset.
dftrain = pd.read_csv('/Users/georgedamoulakis/PycharmProjects/MachineLearning/2_CSV/train.csv') # training data
dfeval = pd.read_csv('/Users/georgedamoulakis/PycharmProjects/MachineLearning/2_CSV/eval.csv') # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

#print( dftrain.head() )

#print( dftrain.describe() )

#print( dftrain.shape )

#print( y_train.head() )

#plt.show( dftrain.age.hist(bins=20) )

# plt.show( dftrain.sex.value_counts().plot(kind='barh') )

# plt.show( dftrain['class'].value_counts().plot(kind='barh') )

# plt.show( pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive') )

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

#print(feature_columns)

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
# We create a linear estimtor by passing the feature columns we created earlier


linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

clear_output()  # clears consoke output
#print(f' the accuracy of the training is:', result['accuracy'])
# the result variable is simply a dict of stats about our model
#print(result)

#make predictions for every single point
#method dot predict

result = list( linear_est.predict(eval_input_fn) )
#print(result) #printed all the dictionaries
print('----------------------------------------------------------------------------------' )
print('The persons data :' )
print('----------------------------------------------------------------------------------' )
print(dfeval.loc[5]) #here we print to see the person that we are making our prediction to see if it makes
#sense that this guy survived or not
print('----------------------------------------------------------------------------------' )
a = round( ( (result[5]['probabilities'][1]) * 100 ), 2)
print(f' The model predicts that this person has this chance to survive: ', a , ' % ') #at the probabilies we get 2 numbers - one if the person survived
#print(result[0]['probabilities'][0]) #at the probabilies we get 2 numbers - one if the person DID NOT survived
print('----------------------------------------------------------------------------------' )
print(f' If the person actually survived shows 1 - if actually dead show 0 : ------->', y_eval.loc[5]) #here we print out the actual what happened to this person to see
print('----------------------------------------------------------------------------------' )



print("--- %s seconds ---" % (time.time() - start_time))
