# Predictive Analytics On Water Resources Data
A gated recurrent neural net (GRU) model for predicting monthly water demand for a locality based on selected factors - smart water meter’s previous consumption readings, monthly rainfall, temperature and population data collected for 25 months, the results are visualized on a website deployed with Flask.
The model aims to enhance the unmonitored water supply in the country, GRU outperformed the LSTM with an accuracy of 87.2%. 

The curated factors were selected based upon research on the correlation of parameters with the output parameter. The output is the predicted monthly water demand value in metric tonnes. The research project was implemented on real data obtained from the village of Malkhapur, India where smart meters for water monitoring have been installed in each household. The other data collected is also collected from this village.  

The images for the graphs obtained and pictures of our site visit are included in the 'static/img/' folder.  
Screenshots of our system after final deployment are included in the 'screenshots' folder.

Gru.py - The code for the GRU model,  
gru_model.pkl - The pickled version of the model

LSTM.py - The code for the LSTM model  
lstm_model.pkl - The pickled version of the model

predict_app.py - Here, the pickled models are read and the result is sent to indexFinal.html for display

indexFinal.html - The month and the village ward is chosen for prediction of its water demand, then the result is displayed here.

Please Note : For the webpages we have used Bootstrap templates. 
