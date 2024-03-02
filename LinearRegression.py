import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error as rmse
from data_generator import postfix

def lift(x):
    out = []
    for v in x:
        out.append(v)

    for i in range(0, len(x)):
        for j in range(i, len(x)):
            out.append(x[i]*x[j])

    return np.array(out)
    
def lift_dataset(xs):
    out = []
    for x in xs:
        out.append(lift(x))
    return out


# Number of samples
N = 1000

# Noise variance 
sigma = 0.01


# Feature dimension
d = 40


psfx = postfix(N,d,sigma) 
      

X = lift_dataset(np.load("X"+psfx+".npy"))
y = np.load("y"+psfx+".npy")

print(f"Dataset has n={len(X)} samples, each with d={len(X[0])} features, as well as {y.shape[0]} labels.")

samples = []
rmses = []

for i in range(10, 100, 10):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42)
    
    X_train = X_train[0: int(len(y_train) * (i / 100))]
    y_train = y_train [0: int(len(y_train) * (i / 100))]

    print("Randomly split dataset to %d training and %d test samples" % (len(X_train),len(X_test)))
    
    model = LinearRegression()
    
    print("Fitting linear model...",end="")
    model.fit(X_train, y_train)
    print(" done")
    
    
    # Compute RMSE on train and test sets
    rmse_train = rmse(y_train,model.predict(X_train))
    rmse_test = rmse(y_test,model.predict(X_test))
    
    print("Train RMSE = %f, Test RMSE = %f" % (rmse_train,rmse_test))
    
    
    print("Model parameters:")
    print("\t Intercept: %3.5f" % model.intercept_,end="")
    for i,val in enumerate(model.coef_):
        print(", β%d: %3.5f" % (i,val), end="")
    print("\n")

    samples.append(len(y_train))
    rmses.append(rmse_test)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(samples, rmses)
ax.set_xlabel('Sample size')
ax.set_ylabel('RMSE')

plt.show()
