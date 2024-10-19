from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from werkzeug.serving import run_simple
from sklearn.metrics import confusion_matrix

app = Flask(__name__)

# Global variables for model training
knn_data = None
knn_model = None
feature_columns = []
accuracy_matrix = None

@app.route('/')
def index():
    return render_template('index.html')
    


@app.route('/knn', methods=['GET', 'POST'])
def knn():
    global knn_model, knn_data
    accuracy = None
    prediction_result = None  # Variable to hold prediction result
    confusion_mat = None  # Variable to hold confusion matrix

    try:
        if request.method == 'POST':
            if 'dataset' in request.files:  # Handling dataset upload
                file = request.files['dataset']
                knn_data = pd.read_csv(file)
                # Assuming the last column is the target
                X = knn_data.iloc[:, :-1]
                y = knn_data.iloc[:, -1]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                knn_model = KNeighborsClassifier(n_neighbors=3)
                knn_model.fit(X_train, y_train)
                predictions = knn_model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                accuracy_matrix = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': predictions
                })
                confusion_mat = confusion_matrix(y_test, predictions)
                confusion_matrix_df = pd.DataFrame(confusion_mat)

                return render_template('knn.html', accuracy=accuracy, 
                                       accuracy_matrix=accuracy_matrix.to_html(classes='table table-striped'), 
                                       confusion_matrix=confusion_matrix_df.to_html(classes='table table-striped'),
                                       feature_columns=X.columns.tolist())
            else:  # Handling prediction form submission
                feature_values = []
                for column in knn_data.columns[:-1]:  # Exclude target column
                    feature_values.append(float(request.form[column]))
                prediction_result = knn_model.predict([feature_values])[0]  # Predict and get the first result
                return render_template('knn.html', accuracy=accuracy, 
                                       prediction_result=prediction_result, 
                                       confusion_matrix=None,
                                       feature_columns=knn_data.columns[:-1].tolist())

        return render_template('knn.html', accuracy=accuracy, confusion_matrix=None)

    except Exception as e:
        app.logger.error("Error in /knn route: %s", e)
        return render_template('error.html', message=str(e)), 500




linear_regression_model = None
linear_regression_data = None

@app.route('/linear_regression', methods=['GET', 'POST'])
def linear_regression():
    global linear_regression_model, linear_regression_data
    mse = None
    prediction_result = None  # Variable to hold prediction result

    try:
        if request.method == 'POST':
            if 'dataset' in request.files:  # Handling dataset upload
                file = request.files['dataset']
                linear_regression_data = pd.read_csv(file)
                # Assuming the last column is the target (dependent variable)
                X = linear_regression_data.iloc[:, :-1]
                y = linear_regression_data.iloc[:, -1]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                linear_regression_model = LinearRegression()
                linear_regression_model.fit(X_train, y_train)
                predictions = linear_regression_model.predict(X_test)
                mse = mean_squared_error(y_test, predictions)
                accuracy_matrix = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': predictions
                })
                return render_template('linear_regression.html', mse=mse, accuracy_matrix=accuracy_matrix.to_html(classes='table table-striped'), feature_columns=X.columns.tolist())
            else:  # Handling prediction form submission
                feature_values = []
                for column in linear_regression_data.columns[:-1]:  # Exclude target column
                    feature_values.append(float(request.form[column]))
                prediction_result = linear_regression_model.predict([feature_values])[0]  # Predict and get the first result
                return render_template('linear_regression.html', mse=mse, prediction_result=prediction_result, feature_columns=linear_regression_data.columns[:-1].tolist())

        return render_template('linear_regression.html', mse=mse)

    except Exception as e:
        app.logger.error("Error in /linear_regression route: %s", e)
        return render_template('error.html',message = str(e)), 500


if __name__ == '__main__':
    run_simple('0.0.0.0', 5000, app, use_reloader=False, use_debugger=True)
