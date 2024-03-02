import numpy as np
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.linear_model import Lasso
from sklearn.metrics import root_mean_squared_error as rmse
import matplotlib.pyplot as plt
from data_generator import postfix


# Number of samples
N = 1000

# Noise variance 
sigma = 0.01


# Feature dimension
d = 40

psfx = postfix(N,d,sigma) 
      

X = np.load("X"+psfx+".npy")
y = np.load("y"+psfx+".npy")

print("Dataset has n=%d samples, each with d=%d features," % X.shape,"as well as %d labels." % y.shape[0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)

print("Randomly split dataset to %d training and %d test samples" % (X_train.shape[0],X_test.shape[0]))


alphas = []
rmses = []
devs = []

def do_lasso(alpha, kfold):
    model = Lasso(alpha = alpha)
    
    if kfold:
        cv = KFold(
                n_splits=5, 
                random_state=42,
                shuffle=True
                )
    
    
        scores = cross_val_score(
                model, X_train, y_train, cv=cv,scoring="neg_root_mean_squared_error")
        
        
        print("Cross-validation RMSE for α=%f : %f ± %f" % (alpha,-np.mean(scores),np.std(scores)) )

        return (-np.mean(scores), np.std(scores))
    else: 
    
        print("Fitting linear model over entire training set...",end="")
        model.fit(X_train, y_train)
        print(" done")
        
        
        # Compute RMSE
        rmse_train = rmse(y_train,model.predict(X_train))
        rmse_test = rmse(y_test,model.predict(X_test))

        print("Train RMSE = %f, Test RMSE = %f" % (rmse_train,rmse_test))
        for i,val in enumerate(model.coef_):
            if abs(val) > 0.001:
                print(", β%d: %3.5f" % (i,val), end="")
        print(f", c={model.intercept_}\n")
        return rmse_test

for alpha in np.logspace(-10, 10, num=200, base=2):
    alphas.append(alpha)
    rms, dev= do_lasso(alpha, True)
    rmses.append(rms)
    devs.append(dev)

rmses = np.array(rmses)
devs = np.array(devs)

best_alpha = alphas[np.argmin(rmses)]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(alphas, rmses)
ax.fill_between(alphas, (rmses-devs), (rmses+devs), color='b', alpha=.1)
ax.set_xscale('log', base=2)
ax.set_xlabel('Alpha')
ax.set_ylabel('RMSE')

good_rmse = do_lasso(best_alpha, False)
print(f"Best alpha is {best_alpha}, with an RMSE of {good_rmse}")

plt.show()



