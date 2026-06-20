from flask import Flask, request, render_template, redirect, url_for, session
import tensorflow as tf
import cv2
import numpy as np
import os
import uuid
import mysql.connector
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)
app.secret_key = "supersecretkey"

model = tf.keras.models.load_model("model/calsignet.h5")

UPLOAD_FOLDER = "static/uploads"
ENROLL_FOLDER = "enrolled_signatures"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ENROLL_FOLDER, exist_ok=True)


db = mysql.connector.connect(  
    host="localhost",
    user="root",
    password="",
    database="signature_db",
    autocommit=True
)

cursor = db.cursor(buffered=True)

def check_db_connection():
    """Connection cut aagi iruntha thirumba connect pannum logic"""
    if not db.is_connected():
        db.reconnect(attempts=3, delay=1)

def preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return img.reshape(1, 128, 128, 1)

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        mode = request.form.get("mode")
        user_id = request.form.get("user_id")
        file = request.files["signature"]

        filename = f"{uuid.uuid4()}.png"
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_path)

        user_folder = os.path.join(ENROLL_FOLDER, user_id)

    
        check_db_connection()

        if mode == "enroll":
            os.makedirs(user_folder, exist_ok=True)
            save_path = os.path.join(user_folder, filename)
            os.rename(temp_path, save_path)

            cursor.execute(
                "INSERT INTO users (user_id, signature_path) VALUES (%s, %s)",
                (user_id, save_path)
            )
            result = "✅ Signature Enrolled & Stored in Database"

        elif mode == "verify":
            if not os.path.exists(user_folder):
                result = "❌ No enrolled signature found for this user"
            else:
                enrolled_files = os.listdir(user_folder)
                if len(enrolled_files) == 0:
                    result = "❌ No enrolled signature found"
                else:
                    enroll_path = os.path.join(user_folder, enrolled_files[0])
                    stored = cv2.imread(enroll_path, 0)
                    uploaded = cv2.imread(temp_path, 0)

                    stored = cv2.resize(stored, (128, 128))
                    uploaded = cv2.resize(uploaded, (128, 128))

                    similarity, _ = ssim(stored, uploaded, full=True)

                    if similarity > 0.85:
                        status = "Genuine"
                        result_text = "✅ Genuine (Matched with Enrolled Signature)"
                        score = float(similarity)
                    else:
                        img = preprocess(temp_path)
                        prediction = model.predict(img)
                        score = float(prediction[0][0])

                        if score >= 0.60:
                            status = "Genuine"
                            result_text = "✅ Genuine"
                        elif score >= 0.35:
                            status = "Suspicious"
                            result_text = "⚠️ Suspicious"
                        else:
                            status = "Forged"
                            result_text = "❌ Forged"

                    
                    check_db_connection()
                    cursor.execute(
                        "INSERT INTO verification_history (user_id, score, result) VALUES (%s, %s, %s)",
                        (user_id, score, status)
                    )
                    result = f"{result_text} (Score: {score:.2f})"

        
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return render_template("index.html", result=result)

@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    error = ""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        check_db_connection()
        cursor.execute(
            "SELECT * FROM admins WHERE username=%s AND password=%s",
            (username, password)
        )
        admin = cursor.fetchone()

        if admin:
            session["admin_logged_in"] = True
            return redirect(url_for("dashboard"))
        else:
            error = "Invalid Admin Credentials"

    return render_template("admin_login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("admin_login"))

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))

    records = []
    if request.method == "POST":
        user_id = request.form.get("user_id")
        check_db_connection()
        cursor.execute(
            "SELECT user_id, score, result, verified_at FROM verification_history WHERE user_id = %s ORDER BY verified_at DESC",
            (user_id,)
        )
        records = cursor.fetchall()

    return render_template("dashboard.html", records=records)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)