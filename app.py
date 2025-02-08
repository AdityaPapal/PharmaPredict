from flask import Flask,request,render_template

from src.pipelines.prediction_pipeline import CustomData,PredictPipeline
from src.exception.exception import CustomException
from src.logging.logger import logging
import pandas as pd 
import numpy as np  


application = Flask(__name__)
app = application

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict_data():
    if request.method == "GET":
        return render_template("predict.html")
    else:
        try:
            data = CustomData(
                Age = int(request.form.get('Age')),
                Sex = request.form.get("Sex"),
                BP = request.form.get("BP"),
                Cholesterol = request.form.get("Cholesterol"),
                Na_to_K = float(request.form.get("Na_to_K"))
            )
            
            final_data= data.get_data_as_dataframe()
            predict_pipline = PredictPipeline()
            pred = predict_pipline.predict(final_data)
            result = pred

            if pred == ['drugX']:
                return render_template("results/drugX.html" ,drug=result )
            elif pred == ['drugY']:
                return render_template("results/drugY.html", drug=result)
            if pred == ['drugA']:
                return render_template("results/drugA.html", drug=result)
            elif pred == ['drugB']:
                return render_template("results/drugB.html", drug=result)
            elif pred == ['drugC']:
                return render_template("results/drugC.html", drug=result)
           # else:
            #     # Handle unexpected prediction values
                # result = pred
            
            # return render_template("results/drugA.html")
        
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            return render_template("error.html", error_message="An error occurred during prediction.")
            # if result == 0:
            #     return render_template("result.html", final_result="DurgX")
            # elif result == 1:
            #     return render_template("result.html", final_result="DrugY")
            # elif result == 2:
            #     return render_template("result.html", final_result="DrugA")
            # elif result == 3:
            #     return render_template("result.html", final_result="DrugB")
            # elif result == 4:
            #     return render_template("result.html", final_result="DrugC")
    # return render_template('input_form.html')

if __name__ == "__main__":
    app.run(debug=True)