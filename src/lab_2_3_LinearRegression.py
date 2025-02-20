# Import here whatever you may need
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class LinearRegressor:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit_simple(self,X,y):
        #Me aseguro de que X sea un array 1D
        if np.ndim(X) > 1:
            X = X.flatten()

        #Calculo la pendiente utilizando la fórmula: pendiente = cov(X,y)/var(X)
        numerator = np.sum((X - np.mean(X))*(y - np.mean(y)))
        denominator = np.sum((X - np.mean(X))**2)
        slope = numerator/denominator

        #Calculo el intercepto
        intercept = np.mean(y) - slope*np.mean(X)

        self.coefficients = np.array([slope])
        self.intercept = intercept

    def fit_multiple(self,X,y):
        #Agrego una columna de 1's para el intercepto y calculo los coeficientes
        X_augmented = np.column_stack((np.ones(X.shape[0]), X))
        beta = np.linalg.inv(X_augmented.T.dot(X_augmented)).dot(X_augmented.T.dot(y))

        self.intercept = beta[0]
        self.coefficients = beta[1:]

    def predict(self, X):
        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")

        if np.ndim(X) == 1:
            # TODO: Predict when X is only one variable
            predictions = self.intercept + self.coefficients[0] * X
        else:
            predictions = self.intercept + X.dot(self.coefficients)
        return predictions


def evaluate_regression(y_true,y_pred):
    #Calculo el r cuadrado, el rmse y el mae
    r_squared = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}


def sklearn_comparison(x,y,linreg):
    #Importo LinearRegression y me aseguro de que x sea un array de dos dimensiones
    from sklearn.linear_model import LinearRegression
    x_reshaped = x.reshape(-1, 1)

    #Creao y ejecuto el modelo de scikit-learn
    sklearn_model = LinearRegression()
    sklearn_model.fit(x_reshaped, y)

    #Comparo los coeficientes e interceptos
    print("Custom Model Coefficient:", linreg.coefficients)
    print("Custom Model Intercept:", linreg.intercept)
    print("Scikit-Learn Coefficient:", sklearn_model.coef_[0])
    print("Scikit-Learn Intercept:", sklearn_model.intercept_)

    return {
        "custom_coefficient": linreg.coefficients,
        "custom_intercept": linreg.intercept,
        "sklearn_coefficient": sklearn_model.coef_[0],
        "sklearn_intercept": sklearn_model.intercept_,
    }


def anscombe_quartet():
    #Cargo el cuarteto de Anscombe
    anscombe = sns.load_dataset("anscombe")

    #Obtengo los identificadores únicos de los conjuntos de datos
    datasets = anscombe['dataset'].unique()
    models = {}
    results = {"R2": [], "RMSE": [], "MAE": []}

    for dataset in datasets:
        #Filtro los datos para el conjunto y creo una instancia del modelo de regresión lineal
        data = anscombe[anscombe['dataset'] == dataset]
        model = LinearRegressor()

        #Extraigo el predictor y la respuesta
        X = data['x'].values
        y = data['y'].values

        #Ajusto el modelo con regresión simple y hago las predicciones
        model.fit_simple(X, y)
        y_pred = model.predict(X)

       #Guardo el modelo e imprimo los coeficientes
        models[dataset] = model
        print(f"Dataset {dataset}: Coefficient: {model.coefficients}, Intercept: {model.intercept}")

        #Evalúo el rendimiento del modelo
        evaluation_metrics = evaluate_regression(y, y_pred)
        print(f"R2: {evaluation_metrics['R2']}, RMSE: {evaluation_metrics['RMSE']}, MAE: {evaluation_metrics['MAE']}")

        results["R2"].append(evaluation_metrics["R2"])
        results["RMSE"].append(evaluation_metrics["RMSE"])
        results["MAE"].append(evaluation_metrics["MAE"])

    return anscombe, datasets, models, results

# Ir al notebook para visualizar los resultados