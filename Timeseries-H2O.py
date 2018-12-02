import numpy
import pandas
import h2o
from sklearn.preprocessing import MinMaxScaler
import warnings

import matplotlib.pyplot as plt
from h2o.estimators.deeplearning import H2ODeepLearningEstimator, H2OAutoEncoderEstimator

"# Create model S = 0 ; Continuous training model S = 1"

S = 0
print('Model is in : ', S, '\n')
# load the dataset
dataframe = pandas.read_csv('Yield7Y', engine='python', skiprows=0, usecols=["Yield"])
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


# reshape into X=t and Y=t+1
look_back = 12
trainX, trainY = create_dataset(dataset, look_back)
# testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], look_back))
trainY = numpy.reshape(trainY, (trainX.shape[0], 1))
# testX = numpy.reshape(testX, (testX.shape[0], look_back, 1))
training_frame = numpy.concatenate((trainX, trainY), axis=1)

h2o.init(nthreads=-1, max_mem_size='4G')
h2o.remove_all()
# train = h2o.H2OFrame(training_frame, destination_frame='ts.hex')
training_frame = h2o.H2OFrame.from_python(training_frame)
training_frame, valid_frame = training_frame.split_frame(ratios=[0.9], seed=1234)
x = list(range(0, look_back))
y = look_back
plt.style.use('ggplot')
if S == 0:
    model = H2ODeepLearningEstimator(activation="Tanh", hidden=[100, 100],
                                     epochs=1200, overwrite_with_best_model=True, nfolds=5, score_training_samples=0,
                                     seed=777, train_samples_per_iteration=0)
    model.train(x=x, y=y, training_frame=training_frame, validation_frame=valid_frame)
    model.show()
    # print the auc for the cross-validated data
    # model.cross_validation_metrics_summary()
    # test_model = model.model_performance(valid_frame)
    sh = model.score_history()
    sh = pandas.DataFrame(sh)
    print(sh.columns)
    warnings.filterwarnings('ignore')
    sh.plot(x='duration', y=['training_deviance', 'validation_deviance'])
    plt.show()
    print("Saving model..>>>> ")
    path = h2o.save_model(model, force=True)
else:
    model = h2o.load_model('DL_5Y_0')
    model.train(x=x, y=y, training_frame=training_frame, validation_frame=valid_frame)
    predictions = model.predict(h2o.H2OFrame(trainX))
    model.show()
    sh = model.score_history()
    sh = pandas.DataFrame(sh)
    sh.plot(x='duration', y=['training_deviance', 'validation_deviance'])
    plt.show()
    print("Saving model..>>>> ")
    path = h2o.save_model(model, force=True)
h2o.remove_all()
