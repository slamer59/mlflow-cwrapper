import mlflow.pyfunc
import os
import example

# https://www.mlflow.org/docs/latest/models.html#example-creating-a-custom-add-n-model

# Define the model class
class ExampleModel(mlflow.pyfunc.PythonModel):

    def __init__(self, n):
        self.n = n

#    def load_context(self, context):
#        self.xgb_model.load_model(context.artifacts["py_wrap_model"])

    def predict(self, context, model_input):
        return example.fact(self.n)

# Construct and save the model
model_path = "py_wrapp_model"
py_wrap_model = ExampleModel(n=5)
py_lib_path = "_example.so"

artifacts = {
    "py_wrap_model": py_lib_path
}


mlflow.pyfunc.save_model(path=model_path, python_model=py_wrap_model,  artifacts=artifacts)
mlflow.pyfunc.log_model(artifact_path='mlruns', python_model=py_wrap_model, artifacts=artifacts)

# Load the model in `python_function` format
loaded_model = mlflow.pyfunc.load_model(model_path)

# Evaluate the model
import pandas as pd
model_input = pd.DataFrame([range(10)])
model_output = loaded_model.predict(model_input)



