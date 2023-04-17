\name{PrasannaNN}
\alias{PrasannaNN}
\title{R6 class for training a neural network with backpropagation}
\description{
This R6 class implements a neural network with backpropagation for supervised learning tasks. It supports multiple hidden layers and can handle both binary and multi-class classification problems.
}
\usage{
NeuralNetwork$new(formula, hidden, data = list())
}
\arguments{
\item{formula}{
An object of class "formula" that describes the model. See the documentation for the "formula" function for details.
}
\item{hidden}{
An integer specifying the number of hidden nodes in each hidden layer.
}
\item{data}{
A data frame containing the training data. The default value is an empty list, in which case the model will need to be trained using the "train" method before predictions can be made.
}
}
\details{
The "initialize" method sets up the neural network by initializing the weights and biases, and the "fit" method performs forward propagation to compute the output for a given input. The "backpropagate" method performs backpropagation to update the weights and biases based on the error between the predicted output and the true output. The "predict" method returns the predicted class labels for a given input, and the "compute_loss" method computes the cross-entropy loss function for a given output.
}
\value{
An object of class "NeuralNetwork", which is an R6 object with the following methods:
\itemize{
\item initialize
\item fit
\item feedforward
\item backpropagate
\item predict
\item compute_loss
\item train
\item accuracy
\item sigmoid
}
}
\references{
No specific references are required for this package, but the methods used in this implementation are standard techniques in neural network and machine learning literature.
}
\author{
This package was created by Prasanna Rodrigues.
}
\note{
This package is provided "as is" and without any warranty.
}
\seealso{
Other packages for neural network and machine learning in R include "neuralnet", "caret", and "h2o".}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- Set up the model and train it on the iris dataset

library(caret)
library(R6)

Example 1

data(iris)
nn <- PrasannaNeuralNetwork$new(Species ~ ., hidden = 5, data = iris)
tr <- nn$train(iterations = 1000, trace = 0.001, learn_rate = .001)

##-- Make predictions
pred <- tr$predict()

##-- Confusion Matrix

table <- confusionMatrix(iris$Species, as.factor(pred))

}
