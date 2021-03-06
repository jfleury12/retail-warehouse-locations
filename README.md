# Predicting Retail Warehouse Locations
Analysis on retail fulfillment centers and likely future candidates

### Project Members
   - <b>[Aaron Childress](https://github.com/achildress83)</b>
   - <b>[Justin Fleury](https://github.com/jfleury20)</b>
   
### Project Scope and Background
Logistics has become a hot area for investment across the supply chain, from startups disrupting the trucking space all the way down to the commercial real estate on which warehouses are built. Major retailers like Amazon and Walmart are ever expanding their warehouse footprint in competition with each other to get goods and services to customers as fast as possible and at the best price – this requires scale. Moreover, US warehouse square footage is expected to reach 15 billion by 2023.

The average age of a U.S. warehouse is 34 years, according to a survey by real estate services firm CBRE. And that likely won't cut it for a retail industry that's moving increasingly toward e-commerce and fulfilling online orders for customers same day. In surveying facilities across 56 markets throughout the country, CBRE found most warehouses built before about 2005 lack modern upgrades: Ceilings are low, flooring is uneven and space is tight.

New construction of facilities that retailers are increasingly hungry for has been hitting the West and South, including California's Inland Empire, Las Vegas, Phoenix and Atlanta. Industrial land plots of 5 to 10 acres, which typically house distribution centers for completing "last-mile" deliveries, watched their prices soar to more than $250,000 per acre by the end of 2017, up from roughly $200,000 the year before.

With this as the backdrop, our goal is to build a classification model to predict high probability zones for future warehouse locations.

### Project Goals
- Based on demographic characteristics, be able to determine which U.S. counties are likely to open major retail warehouses.

### Data
Our analysis included 2364 out of roughly 3000 counties in the United States. This provided a good combination of practicality and interpretability. Analysis was narrowed to Amazon and Walmart locations.

Data primarily sourced from:
- U.S. Census Bureau ACS Survey
- Zillow Home Value Index
- USDA Economic Research Service Surveys
- Publicly-available warehouse location data

### Methodology
- Using U.S. county-level data, construct a classification model to determine the probability of each county containing the target variable (warehouse presence).
- Demographic characteristics were examined for prediction such as: population density, median household income, unemployment rate, education level, net migration rate.
- Model utilized was logistic regression with SMOTE to alleviate class imbalance.
- False positive classifications were used a proxy to determine likely counties. 

### Project Links

#### Presentation
- Predicting_Retail_Warehouses.pdf

#### Technical Notebook
- notebooks/warehouse_model.ipynb

#### Working Notebooks
- Scraping/Cleaning Wareshouse Locations: notebooks/amazon_FC_locations.ipynb
- Census Data Cleaning: notebooks/data_cleaning_county.ipynb 
- Exploratory Data Analysis: notebooks/exploratory_data_analysis.ipynb
