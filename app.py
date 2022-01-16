import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle5
import pandas as pd
# 2. Create the app object
app = FastAPI()
pickle5_in = open("model.pkl","rb")
model=pickle5.load(pickle5_in)



# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_category(data:category):
    data = data.dict()
    category=data['category']

   # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction = model.wv.most_similar([[category]])
    
    return {
        'prediction': prediction
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5000)