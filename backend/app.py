import os
import base64
import mysql.connector
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from deepface import DeepFace
from deepface.commons import functions # For alignment (kept for potential future DeepFace helpers)
from datetime import datetime
from PIL import Image
import io
import uuid # Added for unique filename generation
# Removed Keras/TensorFlow/CV2 imports as we are reverting to DeepFace

# ------------------------------------------------
# NEW IMPORTS FOR QR & PDF GENERATION
# ------------------------------------------------
import qrcode
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch, mm
from reportlab.platypus import Image as PDFImage

# ------------------------------------------------
# NEW IMPORTS FOR OTP & GOOGLE AUTH
# ------------------------------------------------
import smtplib
from email.mime.text import MIMEText
import random
import string
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

# ------------------------------------------------
# FOLDER SETUP
# ------------------------------------------------
TEMP_DIR = "data/temp_uploads"
IMG_DIR = "data/db_images"

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# ------------------------------------------------
# MySQL Config (uses config.py)
# ------------------------------------------------
import config

def get_db():
    return mysql.connector.connect(
        host=config.MYSQL_HOST,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        database=config.MYSQL_DB
    )


# ------------------------------------------------
# FLASK APP
# ------------------------------------------------
app = Flask(__name__)
CORS(app)

# Serve images from folder
@app.route("/images/<filename>")
def serve_image(filename):
    return send_from_directory(IMG_DIR, filename)


# ------------------------------------------------
# HELPERS
# ------------------------------------------------
def save_base64_image(base64_data, folder, filename_prefix="img"):
    """Convert base64 encoded image to JPG file in local folder."""
    if "," in base64_data:
        base64_data = base64.split(",")[1]

    img_bytes = base64.b64decode(base64_data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    filename = f"{filename_prefix}_{datetime.now().timestamp()}.jpg"
    filepath = os.path.join(folder, filename)
    
    img.save(filepath, "JPEG")
    return filepath


# ------------------------------------------------
# SUBMIT REPORT (GET DETAILS)
# ------------------------------------------------
@app.route("/api/report/<int:report_id>", methods=["GET"])
def get_report_details(report_id):
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM reports WHERE id = %s", (report_id,))
        report = cursor.fetchone()
        cursor.close()
        conn.close()

        if not report:
            return jsonify({"success": False, "message": "Report not found"}), 404

        return jsonify({
            "success": True,
            "report": {
                "id": report["id"],
                "user_id": report["user_id"],
                "name": report["name"],
                "age": report["age"],
                "gender": report["gender"],
                "location": report["location"],
                "last_seen_date": str(report["last_seen_date"]),
                "description": report["description"],
                "status": report["status"],
                "filed_at": str(report["filed_at"]),
                "photoPath": f"http://127.0.0.1:5000/images/{report['image_path']}",
                "phone": report.get("phone")
            }
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ------------------------------------------------
# AI SEARCH (DEEPFACE SIMILARITY RESTORED)
# ------------------------------------------------
@app.route("/api/search", methods=["POST"])
def search_face():
    data = request.json
    base64_image = data.get("image")

    if not base64_image:
        return jsonify({"success": False, "message": "No image provided."}), 400

    # Save temporary query image
    query_path = save_base64_image(base64_image, TEMP_DIR, "query")

    try:
        # 1. REMOVED: query_path = functions.align_face(img_path=query_path) 
        # The crash was caused by deepface.commons.functions not having 'align_face'.
        # Alignment is handled internally by DeepFace.find(enforce_detection=True).

        # 2. Delete DeepFace representations cache to force re-indexing
        # NOTE: DeepFace creates a cache file named after the model/distance (e.g., representations_Facenet512_cosine.pkl).
        # We delete the VGG-Face one just in case, but rely on DeepFace to recreate the correct one.
        rep_file = os.path.join(IMG_DIR, "representations_vgg_face.pkl")
        if os.path.exists(rep_file):
            try:
                os.remove(rep_file)
                print("Deleted DeepFace cache to ensure up-to-date index.")
            except Exception as e:
                print("Could not delete DeepFace cache:", e)

        # 3. Run DeepFace search using Facenet512 (better accuracy than VGG-Face)
        results = DeepFace.find(
            img_path=query_path,
            db_path=IMG_DIR,
            model_name="Facenet512",
            detector_backend="opencv",
            distance_metric="cosine",
            enforce_detection=False # <<< CHANGED TO FALSE TO PREVENT CRASH
        )

        matches = []
        if isinstance(results, list) and len(results) > 0 and not results[0].empty:
            df = results[0]
            for _, row in df.iterrows():
                img_path = str(row["identity"]).replace("\\", "/")
                
                # Dynamically find the distance column (e.g., Facenet512_cosine)
                distance_col = [c for c in df.columns if "cosine" in c.lower() or "distance" in c.lower()]
                distance = float(row[distance_col[0]]) if distance_col else 1.0

                # Improved Accuracy Threshold (0.55 is a safer default for Facenet512 cosine)
                CONFIDENCE_THRESHOLD = 0.55 
                
                if distance <= CONFIDENCE_THRESHOLD:
                    # Cosine distance: 0 = perfect match, 1 = no match. Similarity = 1 - distance.
                    similarity = round((1 - distance) * 100, 2) 
                    filename_only = os.path.basename(img_path)

                    # Lookup in DB by filename only
                    conn = get_db()
                    cursor = conn.cursor(dictionary=True)
                    cursor.execute("SELECT * FROM reports WHERE image_path = %s", (filename_only,))
                    report = cursor.fetchone()
                    cursor.close()
                    conn.close()

                    if report:
                        matches.append({
                            "name": report["name"],
                            "reportId": report["id"],
                            "similarity": similarity,
                            "status": report["status"],
                            "photoPath": report["image_path"]
                        })

        return jsonify({"success": True, "matches": matches})

    except Exception as e:
        print("Search error:", e)
        # Handle this gracefully.
        if "face could not be detected" in str(e):
             # This message should no longer be reached if enforce_detection=False, but kept as a fallback.
             return jsonify({"success": False, "message": "Face detection failed internally. Try a clearer image."}), 400
        return jsonify({"success": False, "message": str(e)}), 500

    finally:
        # Clean up temp query file
        try:
            if os.path.exists(query_path):
                os.remove(query_path)
        except Exception as e:
            print("Could not remove query image:", e)


# ------------------------------------------------
# GET ALL USER REPORTS
# ------------------------------------------------
@app.route("/api/user-reports/<int:user_id>", methods=["GET"])
def get_user_reports(user_id):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT id, name, age, gender, location, last_seen_date, 
                description, image_path, status, filed_at
        FROM reports
        WHERE user_id = %s
        ORDER BY filed_at DESC
    """, (user_id,))

    reports = cursor.fetchall()

    # Process reports to include full URL for photoPath
    for report in reports:
        if report["image_path"]:
            report["photoPath"] = f"http://127.0.0.1:5000/images/{os.path.basename(report['image_path'])}"

    cursor.close()
    conn.close()

    return jsonify({"success": True, "reports": reports})


# ------------------------------------------------
# UPDATE STATUS (FOUND / PENDING)
# ------------------------------------------------
@app.route("/api/update-status", methods=["POST"])
def update_status():
    data = request.json
    report_id = data["reportId"]
    status = data["status"]

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("UPDATE reports SET status=%s WHERE id=%s", (status, report_id))
    conn.commit()

    cursor.close()
    conn.close()

    return jsonify({"success": True, "message": "Status updated"})


# ------------------------------------------------
# DELETE REPORT
# ------------------------------------------------
@app.route("/api/delete-report/<int:report_id>", methods=["DELETE"])
def delete_report(report_id):
    conn = get_db()
    cursor = conn.cursor()

    # get filename
    cursor.execute("SELECT image_path FROM reports WHERE id=%s", (report_id,))
    row = cursor.fetchone()
    
    # Handle both tuple (from standard cursor) and dict (from dictionary cursor) just in case, 
    # though standard cursor returns tuples.
    if row:
        filename = None
        if isinstance(row, dict):
            filename = row.get('image_path')
        elif isinstance(row, (list, tuple)):
            filename = row[0]
            
        if filename:
            filepath = os.path.join(IMG_DIR, filename)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as e:
                    print("Could not remove file:", e)

    cursor.execute("DELETE FROM reports WHERE id=%s", (report_id,))
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({"success": True, "message": "Report deleted"})


# ------------------------------------------------
# GET USER PROFILE
# ------------------------------------------------
@app.route("/api/user/<int:user_id>", methods=["GET"])
def get_user(user_id):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT id, name, email, created_at FROM users WHERE id=%s", (user_id,))
    user = cursor.fetchone()

    cursor.close()
    conn.close()

    if user:
        return jsonify({"success": True, "user": user})
    return jsonify({"success": False, "message": "User not found"})


# ------------------------------------------------
# ADMIN: GET ALL REPORTS
# ------------------------------------------------
@app.route("/api/admin/all-reports", methods=["GET"])
def admin_all_reports():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT reports.*, users.name AS user_name, users.email AS user_email
        FROM reports
        JOIN users ON reports.user_id = users.id
        ORDER BY reports.filed_at DESC
    """)

    reports = cursor.fetchall()

    cursor.close()
    conn.close()

    return jsonify({"success": True, "reports": reports})


# ------------------------------------------------
# ADD NEW OTP + GOOGLE LOGIN ENDPOINTS HERE
# ------------------------------------------------

# ------------------------------------------------
# HELPER: Generate & Send OTP
# ------------------------------------------------
def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

def send_email_otp(to_email, otp):
    try:
        msg = MIMEText(f"Your FindMe verification OTP is: {otp}")
        msg['Subject'] = "FindMe OTP Verification"
        msg['From'] = config.SMTP_EMAIL
        msg['To'] = to_email

        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(config.SMTP_EMAIL, config.SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print("Email error:", e)
        return False

# ------------------------------------------------
# API: Send OTP
# ------------------------------------------------
@app.route("/api/send-otp", methods=["POST"])
def send_otp_route():
    data = request.json
    email = data.get("email")

    if not email:
        return jsonify({"success": False, "message": "Email is required"}), 400

    otp = generate_otp()

    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    try:
        # Check if user exists
        cursor.execute("SELECT id FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()

        # If not exists → create placeholder user for OTP registration
        if not user:
            cursor.execute("""
                INSERT INTO users (email, password)
                VALUES (%s, NULL)
            """, (email,))
            conn.commit()

        # Now update OTP
        cursor.execute("""
            UPDATE users SET otp=%s, otp_expires=NOW() + INTERVAL 5 MINUTE
            WHERE email=%s
        """, (otp, email))
        conn.commit()

        send_email_otp(email, otp)
        return jsonify({"success": True, "message": "OTP sent to email"})
    
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
    
    finally:
        cursor.close()
        conn.close()

# ------------------------------------------------
# API: Verify OTP for Registration
# ------------------------------------------------
@app.route("/api/verify-register-otp", methods=["POST"])
def verify_register_otp():
    data = request.json
    email = data["email"]
    name = data["name"]
    password = data["password"]
    otp = data["otp"]

    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT * FROM users 
        WHERE email=%s AND otp=%s AND otp_expires > NOW()
    """, (email, otp))

    user = cursor.fetchone()

    if user:
        cursor.execute("""
            UPDATE users
            SET name=%s, password=%s, otp=NULL, otp_expires=NULL
            WHERE email=%s
        """, (name, password, email))
        conn.commit()

        return jsonify({"success": True, "message": "Registration successful"})

    return jsonify({"success": False, "message": "Invalid or expired OTP"})

# ------------------------------------------------
# API: Verify OTP for Login
# ------------------------------------------------
@app.route("/api/verify-login-otp", methods=["POST"])
def verify_login_otp():
    data = request.json
    email = data["email"]
    otp = data["otp"]

    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT * FROM users WHERE email=%s 
        AND otp=%s AND otp_expires > NOW()
    """, (email, otp))

    user = cursor.fetchone()

    if user:
        return jsonify({
            "success": True,
            "message": "Login successful",
            "userId": user["id"],
            "email": user["email"],
            "name": user["name"]
        })

    return jsonify({"success": False, "message": "Invalid or expired OTP"})

# ------------------------------------------------
# API: Google Sign-In
# ------------------------------------------------
@app.route("/api/google-auth", methods=["POST"])
def google_auth():
    data = request.json
    token = data["token"]

    try:
        # verify google token
        google_user = id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            config.GOOGLE_CLIENT_ID
        )

        email = google_user["email"]
        name = google_user["name"]
        google_id_val = google_user["sub"]

        conn = get_db()
        cursor = conn.cursor(dictionary=True)

        # check if exist
        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()

        if not user:
            cursor.execute("""
                INSERT INTO users (name, email, google_id)
                VALUES (%s, %s, %s)
            """, (name, email, google_id_val))
            conn.commit()

            cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
            user = cursor.fetchone()

        return jsonify({
            "success": True,
            "userId": user["id"],
            "email": user["email"],
            "name": user["name"]
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

# ------------------------------------------------
# REPORT SUBMISSION ENDPOINT
# ------------------------------------------------
@app.route("/api/report", methods=["POST"])
def file_report():
    try:
        user_id = request.form.get("userId")
        name = request.form.get("name")
        age = request.form.get("age")
        gender = request.form.get("gender")
        location = request.form.get("location")
        last_seen_date = request.form.get("lastSeenDate")
        description = request.form.get("description")
        phone = request.form.get("phone")
        photo = request.files.get("photo")

        if not user_id or not name or not photo:
            return jsonify({"success": False, "message": "Missing required fields"}), 400

        # Save the image with a unique filename
        filename = f"{uuid.uuid4().hex}_{secure_filename(photo.filename)}"
        save_path = os.path.join(IMG_DIR, filename)
        photo.save(save_path)

        # Insert into DB saving only the filename (not the full system path)
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO reports (user_id, name, age, gender, location, last_seen_date, description, image_path, phone)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (user_id, name, age, gender, location, last_seen_date, description, filename, phone))
        conn.commit()
        cursor.close()
        conn.close()

        # Remove DeepFace cache so new identity will be indexed next search
        rep_file = os.path.join(IMG_DIR, "representations_vgg_face.pkl")
        if os.path.exists(rep_file):
            try:
                os.remove(rep_file)
                print("DeepFace cache deleted after new report submission.")
            except Exception as e:
                print("Warning: could not remove representations file:", e)

        return jsonify({"success": True, "message": "Report filed successfully!"})

    except Exception as e:
        print("Report error:", e)
        return jsonify({"success": False, "message": "Server error while filing report"}), 500

# ------------------------------------------------
# API: Generate QR Code
# ------------------------------------------------
@app.route("/api/report-qr/<int:report_id>", methods=["GET"])
def generate_qr(report_id):
    try:
        # URL to open when scanned (your frontend file)
        url = f"http://127.0.0.1:5500/findme_app.html#report-details?id={report_id}"

        qr_img = qrcode.make(url)
        
        # Save QR to a static file so it can be served via URL
        qr_filename = f"qr_{report_id}.png"
        qr_save_path = os.path.join(IMG_DIR, qr_filename)
        qr_img.save(qr_save_path, format="PNG")
        
        # Path for the JSON response
        qr_path = f"/images/{qr_filename}"

        return jsonify({
            "success": True,
            "qr_url": f"http://127.0.0.1:5000{qr_path}",
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ------------------------------------------------
# API: Report Public Link
# ------------------------------------------------
@app.route("/api/report-url/<int:report_id>")
def report_public_link(report_id):
    return jsonify({
        "url": f"http://127.0.0.1:5000/report/{report_id}"
    })

# ------------------------------------------------
# API: Generate PDF Report (Simple)
# ------------------------------------------------
@app.route("/api/report-pdf/<int:report_id>")
def generate_report_pdf(report_id):
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM reports WHERE id=%s", (report_id,))
        r = cursor.fetchone()

        cursor.close()
        conn.close()

        if not r:
            return jsonify({"success": False, "message": "Report not found"}), 404

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)

        y = 750

        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, y, "Missing Person Report")
        y -= 40

        c.setFont("Helvetica", 12)

        for key in ["name", "age", "gender", "location", "last_seen_date", "description", "status"]:
            c.drawString(50, y, f"{key.capitalize()}: {r.get(key, '')}")
            y -= 20

        c.save()
        buffer.seek(0)

        return send_file(buffer, download_name="report.pdf", as_attachment=True)

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ---------------------------------------------------------
#       PDF A4 POSTER GENERATION (FINAL FIXED VERSION)
# ---------------------------------------------------------
@app.route("/api/report-poster/<int:report_id>")
def report_poster(report_id):
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM reports WHERE id=%s", (report_id,))
        report = cursor.fetchone()
        cursor.close()
        conn.close()

        if not report:
            return jsonify({"success": False, "message": "Report not found"}), 404

        # correct image location
        img_path = os.path.join(IMG_DIR, os.path.basename(report["image_path"]))
        if not os.path.exists(img_path):
            return jsonify({"success": False, "message": "Image Not Found On Server"}), 500

        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=A4)

        # -------- POSTER HEADER --------
        pdf.setFont("Helvetica-Bold", 34)
        pdf.drawCentredString(300, 800, "FINDME – Missing Person Alert")

        # -------- PERSON PHOTO --------
        pdf.drawImage(img_path, 150, 500, width=260, height=260)

        # -------- DETAILS TEXT --------
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(50, 460, f"Name: {report['name']}")
        pdf.drawString(50, 435, f"Age: {report['age']}")
        pdf.drawString(50, 410, f"Gender: {report['gender']}")
        pdf.drawString(50, 385, f"Phone: {report.get('phone','Not Available')}")

        pdf.setFont("Helvetica", 14)
        pdf.drawString(50, 360, f"Last Seen: {report['last_seen_date']}")
        pdf.drawString(50, 340, f"Location: {report['location']}")
        pdf.drawString(50, 318, f"Description: {report['description']}")

        # -------- QR CODE CREATION --------
        report_link = f"http://127.0.0.1:5000/report/{report_id}"
        qr_path = f"qr_{report_id}.png"
        qrcode.make(report_link).save(qr_path)

        pdf.setFont("Helvetica-Bold", 15)
        pdf.drawString(50, 285, "Scan QR to open report online:")
        pdf.drawImage(qr_path, 50, 110, width=160, height=160)

        os.remove(qr_path)  # cleanup after saving

        pdf.save()
        buffer.seek(0)

        return send_file(buffer,
                         as_attachment=True,
                         download_name=f"FindMe_Report_{report['name']}.pdf",
                         mimetype="application/pdf")

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ------------------------------------------------
# RUN SERVER
# ------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)