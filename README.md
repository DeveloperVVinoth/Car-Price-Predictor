# 🚗 Car Price Predictor

This project uses machine learning to predict the **selling price of used cars** based on features like car age, fuel type, seller type, transmission, and kilometers driven.

## 🔍 Dataset

The dataset used is `car data.csv`, which contains historical car sales data with the following columns:
- `Car_Name`
- `Year`
- `Selling_Price`
- `Present_Price`
- `Kms_Driven`
- `Fuel_Type`
- `Seller_Type`
- `Transmission`
- `Owner`

## 🧠 Model

We use a **Random Forest Regressor** from `scikit-learn` to predict the selling price.

## 📁 Project Structure

CarPricePredictor/
├── app.py # Streamlit app
├── car data.csv # Dataset
├── car_price_model.pkl # Trained ML model
├── car_price_encoder.pkl # LabelEncoder for Fuel_Type, Seller_Type, Transmission
├── train_model.py # Model training script
├── requirements.txt # Python dependencies
└── README.md # Project info

## 🚀 How to Run

### 1. Install Dependencies
pip install -r requirements.txt

2. Train the Model (Optional)
python train_model.py

3. Run the App
streamlit run app.py

📊 Features Used
Present_Price
Kms_Driven
Fuel_Typ
Seller_Type
Transmission
Owner
Car_Age (calculated from Year)

🧪 Example Input

Feature	Value
Present_Price	5.59
Kms_Driven	27000
Fuel_Type	Petrol
Seller_Type	Dealer
Transmission	Manual
Owner	0
Car_Age	8

✅ Output
Predicted Selling Price in Lakhs ₹

📸 App Preview

🛠️ Built With
Python 🐍
Streamlit 📊
scikit-learn 🤖
pandas, numpy 📉

🙋‍♂️ Author
Vinoth Viveki — GitHub Profile

Let me know once you’ve saved this or if you'd like to include a screenshot in the repo too. ✅