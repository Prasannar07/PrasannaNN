
PrasannaNeuralNetwork <- R6Class("PrasannaNeuralNetwork",
                         public = list(
                           X = NULL,  Y = NULL,
                           W1 = NULL, W2 = NULL,
                           output = NULL,
                           initialize = function(formula, hidden, data = list()) {
                             # Model and training data
                             model <- model.frame(formula, data = data)
                             self$X <- model.matrix(attr(model, 'terms'), data = model)
                             self$Y <- model.response(model)

                             # Dimensions
                             B1 <- ncol(self$X) # input dimensions (+ bias)
                             C <- length(unique(self$Y)) # number of classes
                             B2 <- hidden # number of hidden nodes (- bias)

                             # Initial weights and bias
                             self$W1 <- .001 * matrix(rnorm(B1 * B2), B1, B2)
                             self$W2 <- .001 * matrix(rnorm((B2 + 1) * C), B2 + 1, C)
                           },
                           fit = function(data = self$X) {
                             h <- self$sigmoid(data %*% self$W1)
                             score <- cbind(1, h) %*% self$W2
                             return(self$softmax(score))
                           },
                           feedforward = function(data = self$X) {
                             self$output <- self$fit(data)
                             invisible(self)
                           },
                           backpropagate = function(lr = 1e-2) {
                             h <- self$sigmoid(self$X %*% self$W1)
                             Y_id <- match(self$Y, sort(unique(self$Y)))

                             yhat_y <- self$output - (col(self$output) == Y_id) # E[y] - y
                             dW2 <- t(cbind(1, h)) %*% yhat_y

                             dh <- yhat_y %*% t(self$W2[-1, , drop = FALSE])
                             dW1 <- t(self$X) %*% (self$dsigmoid(h) * dh)

                             self$W1 <- self$W1 - lr * dW1
                             self$W2 <- self$W2 - lr * dW2

                             invisible(self)
                           },
                           predict = function(data = self$X) {
                             prob <- self$fit(data)
                             pred <- apply(prob, 1, which.max)
                             levels(self$Y)[pred]
                           },
                           compute_loss = function(prob = self$output) {
                             Y_id <- match(self$Y, sort(unique(self$Y)))
                             correct_logprob <- -log(prob[cbind(seq_along(Y_id), Y_id)])
                             sum(correct_logprob)
                           },
                           train = function(iterations = 1e4,
                                            learn_rate = 1e-2,
                                            tolerance = .01,
                                            trace = 100) {
                             for (i in seq_len(iterations)) {
                               self$feedforward()$backpropagate(learn_rate)
                               if (trace > 0 && i %% trace == 0)
                                 message('Iteration ', i, '\tLoss ', self$compute_loss(),
                                         '\tAccuracy ', self$accuracy())
                               if (self$compute_loss() < tolerance) break
                             }
                             invisible(self)
                           },
                           accuracy = function() {
                             predictions <- apply(self$output, 1, which.max)
                             predictions <- levels(self$Y)[predictions]
                             mean(predictions == self$Y)
                           },
                           sigmoid = function(x) 1 / (1 + exp(-x)),
                           dsigmoid = function(x) x * (1 - x),
                           softmax = function(x) exp(x) / rowSums(exp(x))
                         )
)


