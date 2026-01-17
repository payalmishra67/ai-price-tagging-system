from flask import Flask, render_template, request
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# -----------------------------
# TRAINING DATA (Synthetic but realistic)
# Features: [stock, demand, past_sales]
# -----------------------------
X = np.array([
    [100, 20, 50],
    [80, 30, 70],
    [60, 50, 100],
    [40, 70, 150],
    [30, 80, 200],
    [20, 90, 250],
    [10, 95, 300]
])

# Target prices (₹)
y = np.array([
    5000,
    7000,
    12000,
    18000,
    22000,
    26000,
    30000
])

# -----------------------------
# FEATURE SCALING
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# MODEL TRAINING
# -----------------------------
model = LinearRegression()
model.fit(X_scaled, y)

# -----------------------------
# ROUTE
# -----------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    price = None

    if request.method == 'POST':
        stock = int(request.form['stock'])
        demand = int(request.form['demand'])
        sales = int(request.form['sales'])

        # Scale input
        input_data = scaler.transform([[stock, demand, sales]])

        prediction = model.predict(input_data)
        price = round(prediction[0], 2)

        # -----------------------------
        # BUSINESS RULE (Price Limits)
        # -----------------------------
        if price < 1000:
            price = 1000
        elif price > 30000:
            price = 30000

    return render_template('index.html', price=price)


if __name__ == '__main__':
    app.run(debug=True)
