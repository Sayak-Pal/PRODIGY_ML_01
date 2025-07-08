---

# 🏡 House Price Prediction using Linear Regression

This repository contains a machine learning project developed as part of the **Prodigy Infotech Internship**. The aim of the project is to **predict housing prices** based on features like **square footage**, **number of bedrooms**, and **bathrooms** using a **Linear Regression model**.

---

## 📌 Project Description

Most home buyers consider only a few visible aspects when thinking of their dream house. However, the actual price is influenced by a combination of many variables. This project explores housing price prediction using the **Ames Housing Dataset**, which includes **79 explanatory variables** describing nearly every aspect of residential homes in Ames, Iowa.

In this task, we simplify the problem by focusing on:

* Square Footage (`GrLivArea`)
* Number of Bedrooms (`BedroomAbvGr`)
* Number of Bathrooms (`FullBath`)

---

## 📂 Dataset

* **Source:** [Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
* **Target Variable:** `SalePrice`
* **Features Used:**

  * `GrLivArea`: Above-ground living area (in square feet)
  * `BedroomAbvGr`: Number of bedrooms above ground
  * `FullBath`: Number of full bathrooms

---

## ⚙️ Technologies Used

* Python 3.x
* Jupyter Notebook
* NumPy & Pandas – data manipulation
* Seaborn & Matplotlib – data visualization
* Scikit-learn – model training and evaluation

---

## 🧠 Concepts Learned

* Exploratory Data Analysis (EDA)
* Feature selection and correlation
* Linear Regression modeling
* Model evaluation using **R² Score** and **Root Mean Squared Error (RMSE)**

---

## 📊 Results

The linear regression model was able to capture basic trends in the data and provided an interpretable baseline for housing price prediction. While limited in complexity, this model lays the groundwork for future enhancements using more features and advanced algorithms.

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/house-price-prediction.git
   cd house-price-prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:

   ```bash
   jupyter notebook House_Price_Prediction.ipynb
   ```

---

## 📁 File Structure

```
├── House_Price_Prediction.ipynb     # Main project notebook
├── data/                            # Dataset CSV files (optional, ignored in .gitignore)
├── README.md                        # Project description
├── requirements.txt                 # Python dependencies
```

---

## 🔮 Future Improvements

* Use all 79 features with regularization techniques (Ridge, Lasso)
* Apply advanced models like Random Forest, Gradient Boosting
* Hyperparameter tuning and cross-validation

---

## 🏷️ Tags

`#MachineLearning` `#LinearRegression` `#AmesHousing` `#Python` `#ProdigyInfotech` `#InternshipProject` `#DataScience`

---

