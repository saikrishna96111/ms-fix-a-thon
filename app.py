from flask import Flask, jsonify, request, render_template
# from sklearn.externals import joblib
import traceback
import pickle
import pandas as pd
import numpy as np

# lr = joblib.load('model.pkl')


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

l=[63,1,3,145,233,1,0,150,0,2.3,0,0,1,1,63,1,3,145,233,1,0,150,0,2.3,0,0,1,1,1,1]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in l]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0],2)
    if(output=="1"):
        return render_template('index.html',prediction_text = "There is a possibility of occurence of heart disease")
    else:
        return render_template('index.html',prediction_text = "There is no possibility of occurence of heart disease")

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)


# def predict():
#     if lr:
#         try:
#             json_ = request.json
#             query = pd.get_dummies(pd.DataFrame(json_))
#             query = query.reindex(columns=model_columns, fill_value=0)

#             prediction = list(lr.predict(query))

#             return jsonify({'prediction': prediction})

#         except:

#             return jsonify({'trace': traceback.format_exc()})
#     else:
#         print ('Train the model first')
#         return ('No model here to use')

# if __name__ == '__main__':
#     try:
#         port = int(sys.argv[1]) # This is for a command-line argument
#     except:
#         port = 12345 # If you don't provide any port then the port will be set to 12345
#     lr = joblib.load(model_file_name) # Load "model.pkl"
#     print ('Model loaded')
#     model_columns = joblib.load(model_columns_file_name) # Load "model_columns.pkl"
#     print ('Model columns loaded')
#     app.run(port=port, debug=True)

if __name__ == '__main__':
    app.run(debug=True)
