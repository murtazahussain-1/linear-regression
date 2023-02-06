import matplotlib
import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


def LinearRegressionBuiltIn():
    df = pandas.read_csv('kc_house_train_data.csv')
    x = df.bedrooms  # This contains all the values of the input feature
    y = df.price  # This contains all the correct prices which we want the model to learn
    regressor = LinearRegression()
    # training the algorithm
    regressor.fit(x.values.reshape(-1, 1), y.values.reshape(-1, 1))
    # To retrieve the intercept:
    print('y-intercept = ', regressor.intercept_)
    # For retrieving the slope:
    print('slope = ', regressor.coef_)

    # X_train, X_test, y_train, y_test
    df = pandas.read_csv('kc_house_test_data.csv')
    X_test = df.bedrooms
    y_test = df.price
    y_pred = regressor.predict(X_test.values.reshape(-1, 1))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print(y_pred-y_test)


def ScatterPlotter(TheetaZero, TheetaOne):
    df = pandas.read_csv('kc_house_train_data.csv')

    x = df.sqft_living  # This contains all the values of the input feature
    y = df.price  # This contains all the correct prices which we want the model to learn
    plt.scatter(x, y)

    A = np.linspace(0, 15000, 15001)
    plt.plot(A, ((TheetaOne*A)+TheetaZero), 'black')

    plt.title("Size VS Price Comparison")
    plt.xlabel('Size in Sqft')
    plt.ylabel("Price of house")
    plt.show()


def TestingTheModel(TheetaZero, TheetaOne, Alpha):
    print('testing begins here')

    print('Opening Testing file...')
    df = pandas.read_csv('kc_house_test_data.csv')
    y_correct = df.price  # This contains the correct value of prices
    # This contains the correct value of sizes for correct prices
    x_correct = df.sqft_living

    y_predicted = TheetaZero+TheetaOne*x_correct
    # J(theeta)=(1/2m) Σ {h(xi)-y(i)}^2
    print('J(Theeta)=Cost= ', ((1/(2*(len(y_correct))))
          * sum((y_predicted-y_correct)**2)))

   # percentage error=  ( abs (  (y_predicted - y_correct)    ) / y_correct    ) * (    100/(Total Test Samples)     )
    print('%error = ', (np.mean(np.abs((y_predicted - y_correct) / y_correct)) * 100))


def LinearRegressionTrainingUsingGradientDescent():
    print('===training begins here===')
    print('Opening Training file...')
    df = pandas.read_csv('kc_house_train_data.csv')

    x = df.sqft_living  # This contains all the values of the input feature
    y = df.price  # This contains all the correct prices which we want the model to learn

    TheetaZero = -20000  # This is our initial y-intercept value.
    # After looking at the price vs sqft graph very carefully
    # I am assuming it to be a huge negative value
    # In this way we would have to do less iterations to reach the local minimum

    TheetaOne = 0  # This is our initial slope value

    Alpha = 0.0000001  # Our learning rate
    # After multiple iterations I can say that this rate works perfectly for our model

    Total = len(y)  # Now we have the total number of samples (m)

    temp = 1
    for j in range(100000):  # 0.1M Iterations
        TheetaZero = TheetaZero - \
            ((Alpha/Total)*sum((TheetaZero+(TheetaOne*x))-y))
        TheetaOne = TheetaOne - \
            ((Alpha/Total)*sum((TheetaZero+((TheetaOne*x))-y)*x))
        # This if condition prints iteration number after every 500 iterations
        if (j+1) == 500*temp:
            print('Iteration number ', j+1)
            print('0o=', TheetaZero, '01=', TheetaOne)
            temp = temp+1

    TestingTheModel(TheetaZero, TheetaOne, Alpha)
    # predicts the new values based on the test-set and gives us the cost function result and the percentage error

    ScatterPlotter(TheetaZero, TheetaOne)
    # gives us a graph of the training set values and our linear model


# =================[MAIN STARTS FROM HERE]===================
def main():
    LinearRegressionTrainingUsingGradientDescent()

main()
