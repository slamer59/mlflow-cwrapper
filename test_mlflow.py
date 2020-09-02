import mlflow.pyfunc
import os

# Define the model class
class AddN(mlflow.pyfunc.PythonModel):

    def __init__(self, n):
        self.n = n

    def predict(self, context, model_input):
        return model_input.apply(lambda column: column + self.n)

# Construct and save the model
model_path = "add_n_model"
add5_model = AddN(n=5)
mlflow.pyfunc.save_model(path=model_path, python_model=add5_model)
mlflow.pyfunc.log_model(artifact_path='mlruns', python_model=add5_model)

# Load the model in `python_function` format
loaded_model = mlflow.pyfunc.load_model(model_path)

# Evaluate the model
import pandas as pd
model_input = pd.DataFrame([range(10)])
model_output = loaded_model.predict(model_input)
assert model_output.equals(pd.DataFrame([range(5, 15)]))


