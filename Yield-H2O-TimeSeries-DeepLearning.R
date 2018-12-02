library(readr)
Yield5Y <- read_csv("Documents/Yield5Y.csv",
                    col_types = cols(Date = col_date(format = "%Y-%m-%d")))
Yield5Y
ts <- as.data.frame(as.numeric(Yield5Y$Yield))
dim(ts) ##476 rows
plot(ts[, 1]) ## generally increasing, with seasonality
lagpad <- function(x, k) {
  c(rep(NA, k), x)[1:length(x)]
}
# Set seq_length (lag) impact

lagging <- as.data.frame(matrix(0, nrow(ts), 12))
for (i in 1:12) {
  lagging[, i] <- lagpad(ts[, 1], i)
}
tsLagged <- cbind(ts, lagging, seq(1:nrow(ts)))
colnames(tsLagged) <-
  c("Yield",
    "l1",
    "l2",
    "l3",
    "l4",
    "l5",
    "l6",
    "l7",
    "l8",
    "l9",
    "l10",
    "l11",
    "l12",
    "Index")

library(h2o)
h <- h2o.init(nthreads = -1, max_mem_size = '4G')
## load data into cluster
tsHex <- as.h2o(tsLagged, destination_frame = 'ts.hex')
y = "Yield"
x = setdiff(names(tsHex), y)
## run deep learning against the time series: all but final year
dl <-
  h2o.deeplearning(
    x = x,
    y = y,
    training_frame = tsHex,
    model_id = "tsDL",
    epochs = 1000,
    hidden = c(50, 50)
  )
summary(dl)
# Save the DRF model to disk
# the model will be saved as "./folder_for_myDRF/myDRF"
h2o.saveModel(dl, path = "Documents/") 
# Load model

# Re-load the DRF model from disk model_id
tsDL_from_disk <- h2o.loadModel(path = "./Documents/tsDL")
#Predictions

dlP <- h2o.predict(dl, newdata = tsHex[nrow(tsLagged), ])

## Input length of prediction and make a loop for running all a single prediction in H2O

predictions = c()
predictions <- dlP[1, 1]
forcast_length = 12
for (n in 1:forcast_length) {
  
  lagging[nrow(lagging) + 1, ] = c(dlP[1, 1], lagging[nrow(lagging),-12])
  tsLagged[nrow(tsLagged) + 1, ] = c(0, lagging[nrow(lagging), ], nrow(tsLagged) + 1)
  tsHex <- as.h2o(tsLagged, destination_frame = 'ts.hex')
  dlP <- h2o.predict(dl, newdata = tsHex[nrow(tsLagged), ])
  predictions[n + 1] <- dlP[1, 1]
  Sys.sleep(1)
}
library(ggplot2)
tt_days <- seq(as.Date(max(Yield5Y$Date)), as.Date(max(Yield5Y$Date) + 12), "days")
forcast_Y <- data.frame(tt_days, predictions)
# Graph
ggplot(Yield5Y, aes(Date, Yield)) + geom_line() + geom_line(data = forcast_Y, aes(tt_days, predictions), color = 'red')


## quickly use forecast package to show what Arima will do
library(forecast)

fit <- stl(ts, s.window = "period")
plot(fit)
autoArima <-
  auto.arima(ts)
pAA <- forecast(autoArima, 12)
plot(pAA)
pAA$model$series