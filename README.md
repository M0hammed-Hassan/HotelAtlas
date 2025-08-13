# HotelAtlas
**Welcome to HotelAtlas, where Machine Learning meets your next vacation spot.**

In this project, i’m unleashing the full power of Machine Learning magic, both Supervised and Unsupervised, plus some SQL wizardry, all to decode the mysteries hidden in a massive hotel dataset.

I guess now you know why i named it HotelAtlas as our journey needs to dive into the world of math and many different algorithms so we need an atlas.

The dataset consists of approximately 85,000 records detailing various hotel attributes.

This project includes the following key components:

- Conducting thorough Exploratory Data Analysis (EDA) and data cleaning to ensure data quality and consistency.

- Applying unsupervised learning methods to segment hotels into distinct clusters based on their features and amenities.

- Building supervised learning models to predict hotel ratings effectively.

- Utilizing SQL queries for data aggregation, ranking, and enrichment to support deeper analysis.

Through HotelAtlas, the goal is to provide actionable insights into customer segments and customers stuff using into the hotels using machine learning and data engineering practices.

## Project Structure
<pre>
HotelAtlas/
│
├── datasets
│   ├── row_hotels_dataset.csv
|
├── jupyter_notebooks/
│   ├── EDA_hotels.ipynb
|
├── README.md    
|
├── LICENCE
|
├── requirements.txt
|
├── .gitignore
</pre>

## Get Started
### 1. Python & Environment Setup
- I suggest using python >= 3.10
- Create your conda environment
    ```bash
    $ conda create --name env-name python==3.10
    ```
- Activate your environment
    ```bash
    $ conda activate env-name
    ```
### 2. Install Dependencies
```bash
$ pip install -r requirements.txt
```

## Hotel EDA Report

This notebook contains an **Exploratory Data Analysis (EDA)** on hotel-related data, aiming to uncover patterns, and insights that can support decision-making and further modeling.

#### Objectives
- Understand the structure and quality of the dataset.
- Identify missing values and data inconsistencies.
- Explore patterns in data features.
- Analyze customer segemnts.
- Visualize key relationships between features.

### Results
- **Cleaned version of dataset** that can be used for different machine learning approaches. 