# Deciphering the New York rental market

This repository contains the presentation, Jupyter workbooks and
Flask/D3/Javescript application I created for the fifth project in
Metis' Data Science program. It is a web application that, given
location, size, and features, predicts the rent for a New York
apartment. 

![Rent Report screeshot][RentReport.png]

I created this application to solve the problem where you are moving
in New York and need to know what you should expect to pay, how
different apartment and building features affect prices, and what is a
good value.

I built the application using a dataset of 95,000 listings I scraped
from RentHop.com from March 25 - 27, 2017. From each listing, I
extracted the location (longitude and latitude), size (number of
bedrooms and bathrooms), price, and apartment and building features.

I stored the data in a MongoDB database running on and AWS instance,
then used Python and the **scikit-learn** modules to test different
linear regression models. The different algorithms were close in
performance. For the application, I settled on an ElasticNet model
with log<sub>10</sub> price as the target variable.

The **app** folder contains the code for the **Rent Report**
application that connects to the MongoDB database a uses the listings
to predict rents. 

When you open Rent Report, it displays a heat map of all the listings
in the city (giving you a quick sense of the relative prices in
different areas), and a chart showing the distribution of all rents.

When you select an apartment size, it updates the chart to show the
distribution of prices for just those size apartment. 

When you enter an address or location, or click on an area of the map,
it updates the chart to reflect prices in that vicinity, and predicts
the rent for that specific size apartment in that specific area. From
there you can toggle feature buttons to explore how different building
and apartment features affect the prediction.
