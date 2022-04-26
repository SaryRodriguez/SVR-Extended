# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 19:04:30 2021

@author: rodrigsa
"""

#Import Libraries
import matplotlib.pyplot as plt #visualization
import numpy as np #mathematical functions
from sklearn.preprocessing import StandardScaler #standardize features
from sklearn.model_selection import train_test_split #split train and test 
from sklearn.datasets import load_boston 
import pandas as pd #data analysis and manipulation
from pathlib import Path #manipulate file system paths
#pip install --user cvxpy==1.1.11
import cvxpy as cp #convex optimization problems
from bayes_opt import BayesianOptimization #pip install bayesian-optimization
from bayes_opt import SequentialDomainReductionTransformer #bounds of the problem can be panned and zoomed dynamically in an attempt to improve convergence.
from bayes_opt import UtilityFunction
from bayes_opt.logger import JSONLogger #save to and load progress from files
from bayes_opt.event import Events
from gplearn.genetic import SymbolicTransformer #Genetic programming generate new non-linear features automatically
from sklearn.metrics import r2_score, mean_absolute_error 

#%%SVR Extended library
class SVRExtended_cvxpy:
     
    """
    SVR based Elastic Net regularization. 
    
        -- Parameter --
            C: determines the number of points that contribute to creation of the boundary. 
               (Default = 0.1)
               The bigger the value of C, the lesser the points that the model will consider.
               
            epsilon: defines the maximum margin in the feature space (Default = 0.1).
                        The bigger its value, the more general~underfitted the model is.
                        Must be defined with a value between zero and one.
                        
            lamda: controls the implication of the weighted Elastic Net regularization.
                        
            kernel: name of the kernel that the model will use. Written in a string format.
                    (Default = "linear"). 
        
                    acceptable parameters: 
                        "linear", "poly", "polynomial", "rbf", 
                        "laplacian", "cosine".
        
                    for more information about individual kernels, visit the 
                    sklearn pairwise metrics affinities and kernels user guide.
                    
                    https://scikit-learn.org/stable/modules/metrics.html
            
            Specific kernel parameters: 
        --Methods--
            fit(X, y): Learn from the data. Returns self.
            predict(X_test): Predicts new points. Returns X_test labels.
            coef_(): Returns alpha support vectors (sv) coefficient, X sv, and b.
            For more information about each method, visit specific documentations.
            
        --Example-- 
            ## Call the class
            >>> from SVRExtended_Library import SVRExtended_cvxpy
            ...
            ## Initialize the SVR object with custom parameters
            >>> model = SVRExtended_cvxpy(C = 10, kernel = "rbf", gamma = 0.1)
            ...
            ## Use the model to fit the data
            >>> fitted_model = model.fit(X, y)
            ...
            ## Predict with the given model
            >>> y_prediction = fitted_model.predict(X_test)
            ...
            ## e.g
            >>> print(y_prediction)
            np.array([12.8, 31.6, 16.2, 90.5, 28, 1, 49.7])
    
    """
    
    def __init__(self, C = 0.1, epsilon = 0.01, kernel = "linear", lamda = 0.2, **kernel_param):
        import cvxpy as cp
        import numpy as np
        from sklearn.metrics.pairwise import pairwise_kernels
        from sklearn.utils import check_X_y, check_array 
        self.cp = cp
        self.C = C
        self.epsilon = epsilon
        self.lamda = lamda
        self.kernel = kernel
        self.pairwise_kernels = pairwise_kernels
        self.kernel_param = kernel_param
        self.check_X_y = check_X_y
        self.check_array = check_array
        
        """ 
        Computes coefficients for new data prediction.
        
            --Parameters--
                X: nxm matrix that contains all data points
                   components. n is the number of points and
                   m is the number of features of each point.
                   
                y: nx1 matrix that contains labels for all
                   the points.
            
            --Returns--
                self, containing all the parameters needed to 
                compute new data points.
        """
        
    def fit(self, X, y):
        X, y = self.check_X_y(X, y)
        # hyperparameters
        C = self.C 
        epsilon =  self.epsilon
        lamda = self.lamda
        kernel = self.kernel
        pairwise_kernels = self.pairwise_kernels
        cp = self.cp
        # Useful parameters
        ydim = y.shape[0]
        onev = np.ones((ydim,1))
        
        # Matrices for the optimizer
        K = pairwise_kernels(X, X, metric = kernel, **self.kernel_param)
        A = onev.T
        b = 0
        G = np.concatenate((np.identity(ydim), -np.identity(ydim)))
        h_ = np.concatenate((C*np.ones(ydim), C*np.ones(ydim))); 
        h = h_.reshape(-1, 1)
        
        # loss function and constraints
        beta = cp.Variable((ydim,1))
        Ev = (epsilon*onev.T)

        min_fun = (1/2)*cp.quad_form(beta, K) - y.T @ beta + lamda*((1-Ev) @ cp.abs(beta) + Ev/2 @ beta**2)
        objective = cp.Minimize(min_fun)
        constraints = [A @ beta == b, G @ beta <= h]
        
        # Solver and solution
        prob = cp.Problem(objective,constraints)
        result = prob.solve()
        
        # support vectors
        beta_1 = np.array(beta.value)
        indx = abs(beta_1) > 5e-3
        beta_sv = beta_1[indx]
        x_sv = X[indx[:,0],:]
        y_sv = y[indx[:,0]]
        
        # get w_phi and b
        k_sv = pairwise_kernels(x_sv, x_sv, metric = kernel, **self.kernel_param)
        cons = np.where(beta_sv >= 0, -epsilon, epsilon)
        
        w_phi = beta_sv @ k_sv
        b = np.mean((cons - w_phi + y_sv)); self.b = b
        self.beta_sv = beta_sv; self.x_sv = x_sv
        return self
        
    def predict(self, X_):
        """Predicts new labels for a given set of new 
           independent variables (X_test).
           
           --Parameters--
               X_test: nxm matrix containing all the points that 
                       will be predicted by the model.
                       n is the number of points. m represents the
                       number of features/dimensions of each point.
            
           --Returns--
               a nx1 vector containing the predicted labels for the 
               input variables.
                
        """
        X_ = self.check_array(X_)
        k_test = self.pairwise_kernels(self.x_sv, X_, metric = self.kernel, **self.kernel_param)
        w_phi_test = self.beta_sv @ k_test
        predict = w_phi_test + self.b
        return predict
    
    def coef_(self):
        """--Returns--
                - Dual Coefficients
                - Support vectors
                - intercept
        """
        return self.beta_sv, self.x_sv, self.b
    
#%%Load Data
boston = load_boston()

# Initializing the dataframe
data = pd.DataFrame(boston.data)

#Adding the feature names to the dataframe
data.columns = boston.feature_names

#Adding target variable to dataframe
data['PRICE'] = boston.target

# Spliting target variable and independent variables
X = data.drop(['PRICE'], axis = 1)
y = data['PRICE']

# Splitting to training and testing data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 4)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%%function bayes opt
def opt_bas(C, epsilon, lamda, gamma):
    
    # parameters
    hyperparameters = {
        'kernel' : "rbf",
        'C' : C, 
        'epsilon' : epsilon, 
        'lamda' : lamda,
        'gamma' : gamma,
    }
    
    # fit and predict
    model = SVRExtended_cvxpy(**hyperparameters).fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    # rescale
    #y_pred = scaler1.inverse_transform(predict.reshape(-1, 1)).reshape(-1)
    
    # get score
    metric = mean_absolute_error(y_test, y_pred)
    
    
    return -metric

#%%
class newJSONLogger(JSONLogger):

      def __init__(self, path):
            self._path=None
            super(JSONLogger, self).__init__()
            self._path = path if path[-5:] == ".json" else path + ".json"

#%%
# Bounded region of parameter space

pbounds = {'C': (300, 600), 'epsilon': (0.04, 1), 'lamda': (0.01, 0.3), 'gamma': (0.04, 0.15)}

# Bayes optimizer instantiation
optimizer = BayesianOptimization(f=opt_bas, 
                                 pbounds=pbounds, 
                                 random_state=1, verbose=2, 
#                                  bounds_transformer=bounds_transformer
                                )

# keep data
#log_path = Path().resolve() / "Logs" / "Boston_leo.json"
#logger = newJSONLogger(path = str(log_path))
#optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

#%%Run bayesian optimization
optimizer.maximize(init_points=4, n_iter=400)

#%%see results
print(optimizer.max)

#%%
{'target': 0.8995199974032354, 'params': {'C': 159.860284875011, 'epsilon': 0.053540006253200274, 'gamma': 0.09010788460383506, 'lamda': 0.10179434000470647}}

#%%
from sklearn import metrics

# Create a SVM Regressor
model = SVRExtended_cvxpy(
    kernel = "rbf", 
    C = 180.923857, 
    epsilon = 0.210641, 
    gamma = 0.069232, 
    lamda = 0.101813
)
# Train the model using the training sets 
model.fit(X_train,y_train);
# Model prediction on train data
y_pred = model.predict(X_train)
# Predicting Test data with the model
y_test_pred = model.predict(X_test)
# Model Evaluation
acc_svm = metrics.r2_score(y_test, y_test_pred)
print('SVR Extended')
print('R^2:', acc_svm)
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_test, y_test_pred))
print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
print('MAPE',np.mean(np.abs(( y_test - y_test_pred)/y_test))*100)

#%%# Visualizing the differences between actual prices and predicted values
y_pred = model.predict(X_test)
plt.scatter(y_test, y_pred)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()

#%%
# Checking residuals
plt.scatter(y_pred,y_test-y_pred)
plt.title("Predicted vs residuals")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()


#%%
def calculate_residuals(model, features, label):
    """
    Creates predictions on the features with the model and calculates residuals
    """
    predictions = model.predict(features)
    df_results = pd.DataFrame({'Actual': label, 'Predicted': predictions})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])
    
    return df_results
#%%
import seaborn as sns
def linear_assumption(model, features, label):
    """
    Linearity: Assumes that there is a linear relationship between the predictors and
               the response variable. If not, either a quadratic term or another
               algorithm should be used.
    """
    print('Assumption 1: Linear Relationship between the Target and the Feature', '\n')
        
    print('Checking with a scatter plot of actual vs. predicted.',
           'Predictions should follow the diagonal line.')
    
    # Calculating residuals for the plot
    df_results = calculate_residuals(model, features, label)
    
    # Plotting the actual vs predicted values
    sns.lmplot(x='Actual', y='Predicted', data=df_results, fit_reg=False, size=7)
        
    # Plotting the diagonal line
    line_coords = np.arange(df_results.min().min(), df_results.max().max())
    plt.plot(line_coords, line_coords,  # X and y points
             color='darkorange', linestyle='--')
    plt.title('Actual vs. Predicted')
    plt.show()    
    
#%%
linear_assumption(model, X_test, y_test)

#%%

def normal_errors_assumption(model, features, label, p_value_thresh=0.05):
    """
    Normality: Assumes that the error terms are normally distributed. If they are not,
    nonlinear transformations of variables may solve this.
               
    This assumption being violated primarily causes issues with the confidence intervals
    """
    from statsmodels.stats.diagnostic import normal_ad
    print('Assumption 2: The error terms are normally distributed', '\n')
    
    # Calculating residuals for the Anderson-Darling test
    df_results = calculate_residuals(model, features, label)
    
    print('Using the Anderson-Darling test for normal distribution')

    # Performing the test on the residuals
    p_value = normal_ad(df_results['Residuals'])[1]
    print('p-value from the test - below 0.05 generally means non-normal:', p_value)
    
    # Reporting the normality of the residuals
    if p_value < p_value_thresh:
        print('Residuals are not normally distributed')
    else:
        print('Residuals are normally distributed')
    
    # Plotting the residuals distribution
    plt.subplots(figsize=(12, 6))
    plt.title('Distribution of Residuals')
    sns.distplot(df_results['Residuals'])
    plt.show()
    
    print()
    if p_value > p_value_thresh:
        print('Assumption satisfied')
    else:
        print('Assumption not satisfied')
        print()
        print('Confidence intervals will likely be affected')
        print('Try performing nonlinear transformations on variables')
        
#%%
normal_errors_assumption(model, X_test, y_test)

#%%

def autocorrelation_assumption(model, features, label):
    """
    Autocorrelation: Assumes that there is no autocorrelation in the residuals. If there is
                     autocorrelation, then there is a pattern that is not explained due to
                     the current value being dependent on the previous value.
                     This may be resolved by adding a lag variable of either the dependent
                     variable or some of the predictors.
    """
    from statsmodels.stats.stattools import durbin_watson
    print('Assumption 4: No Autocorrelation', '\n')
    
    # Calculating residuals for the Durbin Watson-tests
    df_results = calculate_residuals(model, features, label)

    print('\nPerforming Durbin-Watson Test')
    print('Values of 1.5 < d < 2.5 generally show that there is no autocorrelation in the data')
    print('0 to 2< is positive autocorrelation')
    print('>2 to 4 is negative autocorrelation')
    print('-------------------------------------')
    durbinWatson = durbin_watson(df_results['Residuals'])
    print('Durbin-Watson:', durbinWatson)
    if durbinWatson < 1.5:
        print('Signs of positive autocorrelation', '\n')
        print('Assumption not satisfied')
    elif durbinWatson > 2.5:
        print('Signs of negative autocorrelation', '\n')
        print('Assumption not satisfied')
    else:
        print('Little to no autocorrelation', '\n')
        print('Assumption satisfied')
        
#%%
autocorrelation_assumption(model, X_test, y_test)

#%%

def homoscedasticity_assumption(model, features, label):
    """
    Homoscedasticity: Assumes that the errors exhibit constant variance
    """
    print('Assumption 5: Homoscedasticity of Error Terms', '\n')
    
    print('Residuals should have relative constant variance')
        
    # Calculating residuals for the plot
    df_results = calculate_residuals(model, features, label)

    # Plotting the residuals
    plt.subplots(figsize=(12, 6))
    ax = plt.subplot(111)  # To remove spines
    plt.scatter(x=df_results.index, y=df_results.Residuals, alpha=0.5)
    plt.plot(np.repeat(0, df_results.index.max()), color='darkorange', linestyle='--')
    ax.spines['right'].set_visible(False)  # Removing the right spine
    ax.spines['top'].set_visible(False)  # Removing the top spine
    plt.title('Residuals')
    plt.show()  
    
#%%
homoscedasticity_assumption(model, X_test, y_test)

#%%
def calculate_residuals(model, features, label):
    """
    Creates predictions on the features with the model and calculates residuals
    """
    predictions = model.predict(features)
    df_results = pd.DataFrame({'Actual': label, 'Predicted': predictions})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])
    
    return df_results
#%%

df_results = calculate_residuals(model, X_test, y_test)

#%%
from scipy.stats import norm
import matplotlib.pyplot as plt
mu, std = norm.fit(df_results.Residuals)

#plot histogram
plt.hist(df_results.Residuals, bins=25, density=True, alpha=0.6, color='b')

#plot pdf
xmin, xmax=plt.xlim()
x=np.linspace(xmin, xmax, 100)
p = norm.pdf(x,mu,std)
plt.plot(x,p,'k',linewidth=2)
title="Deistribution of Residuals"
plt.title(title)
plt.show

#%%kaggle comparison
from sklearn import metrics

# Import library for Linear Regression
from sklearn.linear_model import LinearRegression
# Create a Linear regressor
lm = LinearRegression()

# Train the model using the training sets 
lm.fit(X_train, y_train);
# Model prediction on train data
y_pred = lm.predict(X_train)

# Predicting Test data with the model
y_test_pred = lm.predict(X_test)

# Model Evaluation
acc_linreg = metrics.r2_score(y_test, y_test_pred)
print('Linear Regression')
print('R^2:', acc_linreg)
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_test, y_test_pred))
print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
print('MAPE',np.mean(np.abs(( y_test - y_test_pred)/y_test))*100)


#%%Random Forest
from sklearn.ensemble import RandomForestRegressor
# Create a Random Forest Regressor
reg = RandomForestRegressor(random_state = 1, max_depth = 14, n_estimators = 66)

# Train the model using the training sets 
reg.fit(X_train, y_train);
# Model prediction on train data
y_pred = reg.predict(X_train)

# Predicting Test data with the model
y_test_pred = reg.predict(X_test)
# Model Evaluation
acc_rf = metrics.r2_score(y_test, y_test_pred)
print('Sklearn Random Forest')
print('R^2:', acc_rf)
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_test, y_test_pred))
print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
print('MAPE',np.mean(np.abs(( y_test - y_test_pred)/y_test))*100)

#%%XGBoost
# Import XGBoost Regressor
from xgboost import XGBRegressor
#Create a XGBoost Regressor
reg = XGBRegressor(objective = "reg:squarederror", random_state = 1, eta = 0.159973, max_depth = 5, n_estimators = 64)

# Train the model using the training sets 
reg.fit(X_train, y_train);
# Model prediction on train data
y_pred = reg.predict(X_train)

#Predicting Test data with the model
y_test_pred = reg.predict(X_test)
# Model Evaluation
acc_xgb = metrics.r2_score(y_test, y_test_pred)
print('XGBoost Regressor')
print('R^2:', acc_xgb)
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_test, y_test_pred))
print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
print('MAPE',np.mean(np.abs(( y_test - y_test_pred)/y_test))*100)

#%%&SVR
# Creating scaled set to be used in model to improve our results
from sklearn.preprocessing import StandardScaler
# Import SVM Regressor
from sklearn import svm
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Create a SVM Regressor
reg = svm.SVR(C = 74.888336, epsilon = 0.156926, gamma = 0.060984)
# Train the model using the training sets 
reg.fit(X_train, y_train);
# Model prediction on train data
y_pred = reg.predict(X_train)
# Predicting Test data with the model
y_test_pred = reg.predict(X_test)
# Model Evaluation
acc_svm = metrics.r2_score(y_test, y_test_pred)
print('Sklearn SVR')
print('R^2:', acc_svm)
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_test, y_test_pred))
print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
print('MAPE',np.mean(np.abs(( y_test - y_test_pred)/y_test))*100)
