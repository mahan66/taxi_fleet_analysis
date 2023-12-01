# Taxi Mobility Data Science Challenge

## Dataset
The dataset for this challenge contains mobility traces of approximately 500 taxi cabs in San Francisco collected over a span of around 30 days. You can access the dataset [here](https://github.com/PDXostc/rvi_big_data/blob/master/cabspottingdata.tar.gz). Each mobility trace file is formatted as [latitude, longitude, occupancy, time], where latitude and longitude are in decimal degrees, occupancy indicates whether a cab has a fare (1 = occupied, 0 = free), and time is in UNIX epoch format.

## Challenge Goals

### 1. CO2 Emission Reduction Calculation
The first goal is to calculate the potential for a yearly reduction in CO2 emissions caused by taxi cabs roaming without passengers. The challenge involves assuming a monthly transition rate of 15% from combustion engine-powered vehicles to electric vehicles. The emission rate for the average passenger vehicle is assumed to be about 404 grams of CO2 per mile.

### 2. Predictor for Taxi Hailing Locations
The second goal is to build a predictor for taxi drivers, forecasting the next place a passenger will hail a cab.

### 3. Bonus Question
As a bonus, you are encouraged to identify clusters of taxi cabs that are relevant from the taxi cab company's perspective.

## Instructions
- Explore the dataset and implement solutions for each of the defined goals.
- Provide clear explanations and code documentation.
- Feel free to use additional resources, libraries, or tools as needed.
- Bonus points for creative insights and visualizations.

Good luck with the challenge! If you have any questions or insights, please share them in the repository.
