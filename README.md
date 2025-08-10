# Waterloo Student Rental Predictor  
## Introduction
Welcome to the Waterloo Student Rental Predictor! This project helps students make smarter rental decisions by predicting fair prices and flagging unusual listings in the Waterloo student housing market.

Built on real [Bamboo Housing](https://bamboohousing.ca) data, enhanced with synthetic samples and custom features reflecting student needs, it uses machine learning models for price prediction and anomaly detection.

The system includes a fast API and an easy-to-use web frontend, deployed on modern cloud infrastructure.
If you’re a student hunting for a good deal, this project provides clear, actionable insights with a clean and intuitive user experience.
## Step-by-step Guide  

1. **Data Collection**  
   Scrape rental listings from Bamboo Housing, collecting key data such as price, address, description, bedrooms, bathrooms, lease type, and gender restrictions.  

2. **Synthetic Data Generation**  
   Use ChatGPT-5 to generate realistic synthetic listings that mirror patterns in the original dataset.  

3. **Feature Engineering & Preprocessing**  
   Create student-specific features (e.g., pet-friendly flags, square footage estimates), encode categorical variables, and normalize numerical fields.  

4. **Anomaly Detection**  
   Train and tune an Isolation Forest model on combined real and synthetic data to detect unusual listings.  

5. **Price Prediction**  
   Develop an XGBoost regression model optimized with feature selection and hyperparameter tuning to predict rental prices.  

6. **API Development**  
   Build a FastAPI backend to process listings, handle missing data, run anomaly checks, and output price predictions with explanations.  

7. **Deployment**  
   Containerize the API using Docker, deploy to Render.com, and host the frontend on GitHub Pages.  

8. **Frontend**  
   Create a responsive web form to submit listings and display results, including predicted price, price status, anomaly flags, and detailed feedback.

## Problem Definition  

Finding affordable, fairly priced housing near campus is a big challenge for many students, especially international or out-of-town students new to the area. Without good info or tools, it’s easy to overpay or rent places that don’t meet expectations.  

For example, when I first came to Waterloo as an international student, I struggled to find decent off-campus housing. I paid way above market price for a semi-furnished room without proper amenities.This and many other similar experiences inspired this project.  

Other strugles students face:  
- Comparing rental prices across neighborhoods and types.  
- Spotting fraudulent or poor-quality listings.  
- Dealing with seasonal price changes and demand.

## Data Gathering

One of the biggest challenges when looking for off-campus housing is avoiding scams and ensuring that the people you’re dealing with are genuine. Bamboo Housing solves this by requiring student verification for all users, as you can only post or book listings if you’re a verified student.

This means no random strangers posting fake properties or inflated prices and lastly, its a safe community where both tenants and landlords are verified members of the student population.

URL Scraped for 7500 listings: https://bamboohousing.ca/homepage

## Feature Engineering with Synthetic Rental Data
To enhance my dataset for training, I used ChatGPT Model 5 to generate synthetic rows. These rows were created by analyzing patterns, ranges, and distributions from the original real-world Waterloo housing data, specifically features like location, property type, bedroom/bathroom count, amenities, and pricing trends.

The synthetic dataset was designed to mimic real listings and by augmenting the dataset to a total of 250,000 rows, I was able to train more robust and accurate machine learning models for rent prediction and anomaly detection.
The following ChatGPT prompt was used in model 5 to generate the synthetic data:
```
## Synthetic Data Generation for Waterloo Student Rentals

You are given a scraped dataset of rental listings from **Bamboo Housing** for **Waterloo, Ontario** — a student-heavy city.  

Your task is to:

1. Understand & Analyze the Original Data
- Learn the structure, column meanings, value ranges, and data types from the scraped dataset.
- Identify trends such as:
  - Typical rental price ranges by room type and lease type.
  - Seasonal listing patterns.
  - Common description styles and keywords.
  - Bed/Bath distributions and their price relationships.
  - Gender restrictions in shared rentals.
  - Detect correlations (e.g., price vs. Bed_Bath, furnished vs. unfurnished, lease type vs. price).

2. Generate Synthetic Rows Following the Same Patterns
- Create enough synthetic rows to reach a total of **15,000 rows** (including the scraped data).
- Keep data statistically consistent with the scraped dataset’s trends.
- Maintain realistic natural language in `Description`, following the tone and style of actual listings.
- Ensure plausible `Address` and neighborhoods for Waterloo (Northdale, Beechwood, Uptown, Laurelwood, etc.).

3. Final Dataset Columns
The final dataset contains only:
- `Price` — numeric, monthly rent in CAD
- `Address` — realistic street or neighborhood name
- `Description` — short listing text
- `Bed_Bath` — e.g., `"3 Bed / 2 Bath"`
- `Lease_Type` — e.g., `8-month`, `12-month`, `4-month`
- `Room_Info` — e.g., `single room`, `shared`, `ensuite`, `bachelor`
- `Gender` — `Male`, `Female`, `Coed`
- `Scraped_At` — `YYYY-MM-DD`

4. Realism Rules
- Keep pricing ranges and seasonal demand spikes in line with the original dataset.
- Reflect student market patterns:
  - Higher prices near University of Waterloo and Wilfrid Laurier University.
  - Cheaper per-room rates in shared houses further from campus.
- Match the ratio of furnished vs. unfurnished, lease types, and room configurations seen in the scraped data.

5. Engineered additional features for richer modeling:  
  - Pet_Friendly — Assigned with a ratio of 25–35% based on CMHC reports for student rentals.  
  - Furnished — Set to 80–95%, reflecting high furnishing rates in student-heavy markets like Waterloo.  
  - Parking_Included — Assigned with a 50–60% ratio, based on city rental permit studies for student housing.  
  - Internet_Available — Set to 90–95% availability, as internet is nearly always included in student-oriented rentals.  
  - Square_Footage — Assigned based on average square footage for a standard room, with variations to reflect real-world ranges.  
```

## Data Preparation and Exploratory Data Analysis

**Data Preprocessing**
   - **One-Hot Encoding**: Converted categorical variables into one-hot encoded vectors for machine learning compatibility.
   - **Boolean Conversion**: Transformed boolean columns into integer format (0 and 1) for consistency.
   - **Feature Scaling**: Applied feature scaling techniques (e.g., MinMaxScaler or StandardScaler) to normalize numerical values.

**Exploratory Data Analysis (EDA)**
   - **Feature Comparison**: Compared column-level values to check for inconsistencies or anomalies.
   - **Correlation Analysis**: Generated correlation matrices to identify relationships between features and target variables.
   - **Distribution Analysis**: Visualized feature distributions using histograms, KDE plots, and box plots to understand data spread and detect outliers.

