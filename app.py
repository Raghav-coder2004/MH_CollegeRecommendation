from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Database setup
def init_db():
    conn = sqlite3.connect("coldb.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS crs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )""")
    conn.commit()
    conn.close()

init_db()

# Load dataset
data = pd.read_csv("MHNew_Cleaned.csv")

# Encode categorical columns
label_encoders = {}
for column in ["Category", "Branch", "College Name", "Location", "College_Status"]:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Features and target
X = data[["MHTCET Score", "12th Score", "Category", "Branch", "Location", "College_Status"]]
y = data["College Name"]

# Train the model
clf = GradientBoostingClassifier()
clf.fit(X, y)

# Extract class labels
categories = label_encoders["Category"].classes_
branches = label_encoders["Branch"].classes_
locations = label_encoders["Location"].classes_
college_statuses = label_encoders["College_Status"].classes_

@app.route("/")
def home():
    return redirect(url_for("login"))  # Ensure login is the first page

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        if not email or not password:
            flash("Please enter both email and password", "danger")
            return redirect(url_for("login"))

        conn = sqlite3.connect("coldb.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM crs WHERE email = ? AND password = ?", (email, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session["user"] = email
            return redirect(url_for("college_form"))
        else:
            flash("Invalid credentials!", "danger")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        if not email or not password:
            flash("Please enter both email and password", "danger")
            return redirect(url_for("register"))

        conn = sqlite3.connect("coldb.db")
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO crs (email, password) VALUES (?, ?)", (email, password))
            conn.commit()
            flash("Account created successfully! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already exists!", "danger")
        finally:
            conn.close()
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/college-form", methods=["GET", "POST"])
def college_form():
    if "user" not in session:
        return redirect(url_for("login"))  # Prevent access if not logged in
    return render_template("index.html", categories=categories, branches=branches, 
                           locations=locations, college_statuses=college_statuses)

@app.route("/recommend", methods=["POST"])
def recommend():
    if "user" not in session:
        return redirect(url_for("login"))
    
    mhcet_score = float(request.form["mhcet"])
    twelfth_score = float(request.form["twelfth"])
    category = request.form["category"]
    selected_branches = request.form.getlist("branches")
    selected_locations = request.form.getlist("locations")
    selected_college_statuses = request.form.getlist("college_statuses")

    if not selected_branches or not selected_locations or not selected_college_statuses:
        flash("Error: Please select at least one option for Branch, Location, and College Status!", "danger")
        return redirect(url_for("college_form"))

    predictions = []
    for branch in selected_branches:
        for location in selected_locations:
            for college_status in selected_college_statuses:
                input_data = np.array([[mhcet_score, twelfth_score,
                                        label_encoders["Category"].transform([category])[0],
                                        label_encoders["Branch"].transform([branch])[0],
                                        label_encoders["Location"].transform([location])[0],
                                        label_encoders["College_Status"].transform([college_status])[0]]])

                prediction = clf.predict(input_data)[0]
                college_name = label_encoders["College Name"].inverse_transform([prediction])[0]
                predictions.append((college_name, branch, location, college_status))

    recommendations = pd.DataFrame(predictions, columns=["College Name", "Branch", "Location", "College Status"]).drop_duplicates()
    
    return render_template("recommendation.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
