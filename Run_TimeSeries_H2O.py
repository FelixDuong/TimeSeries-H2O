import numpy
import pandas
import h2o
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import time

from h2o.estimators.deeplearning import H2OAutoEncoderEstimator, H2ODeepLearningEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator

# Loading data
# load the dataset
dataframe = pandas.read_csv('Yields.csv', engine='python', skiprows=0, usecols=["15Y"])
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX = []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
    return numpy.array(dataX)


# reshape into X=t and Y=t+1
look_back = 12  # Match to look-back with created model
trainX = create_dataset(dataset, look_back)
# testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]

trainX = numpy.reshape(trainX, (trainX.shape[0], look_back))

# Determine number of forecast data
forecast_length = 60
# Initialise H2O server
h2o.init(nthreads=-1, max_mem_size='4G')
h2o.remove_all()
model = h2o.load_model('NN_15Y')
model.summary()
predictions = model.predict(h2o.H2OFrame(trainX))
# t = h2o.H2OFrame(trainX)
# Summary performance
# performance = model.model_performance(t)
# performance

# Starting the prediction procedure
predictions = predictions.as_data_frame(use_pandas=True)
predictions = numpy.array(predictions)
dataset = numpy.append(dataset, predictions[-1])
dataset = numpy.reshape(dataset, (dataset.shape[0], 1))
for n in numpy.arange(forecast_length):
    trainX = []
    trainX = create_dataset(dataset, look_back)
    trainX = numpy.array(trainX)
    trainX = numpy.reshape(trainX, (trainX.shape[0], look_back))
    predictions = model.predict(h2o.H2OFrame(trainX))
    time.sleep(1)
    predictions = predictions.as_data_frame(use_pandas=True)
    predictions = numpy.array(predictions)
    dataset = numpy.append(dataset, predictions[-1])
    dataset = numpy.reshape(dataset, (dataset.shape[0], 1))

dataset = scaler.inverse_transform(dataset)
predictions = scaler.inverse_transform(predictions)
# To plot whole predictions
plt.style.use('ggplot')
x = numpy.arange(0, len(predictions))
y = numpy.array(dataset[len(dataset) - len(predictions):, ])
x_smooth = numpy.linspace(x.min(), x.max(), len(predictions) * 20)
y_smooth = spline(x, y, x_smooth)
z_smooth = spline(x, predictions, x_smooth)
plt.plot(x_smooth, y_smooth)
plt.plot(x_smooth, z_smooth, 'g')

# To plot forecast length
# x = numpy.arange(len(dataset) - forecast_length - 1, len(dataset))
# y = dataset[len(dataset) - forecast_length - 1:, ]
# x_smooth = numpy.linspace(x.min(), x.max(), 500)
# y_smooth = spline(x, y, x_smooth)
# plt.plot(x_smooth, y_smooth, 'g')

print(dataset)
plt.show()
h2o.remove_all()
