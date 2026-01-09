# Find Me! – AI Missing Person Identification System

An AI-powered web application designed to help identify missing persons using facial recognition and deep learning. The system compares uploaded images against a large dataset to find potential matches and assist in faster identification.

---

## 🔍 Problem Statement
Finding missing persons is often a slow and manual process. This project automates image-based identification using AI to improve accuracy, speed, and reliability.

---

## 🚀 Features
- Upload and verify images of missing persons  
- AI-powered facial recognition using DeepFace  
- High-dimensional face embedding comparison  
- Secure user authentication using OTP (Gmail SMTP)  
- REST API–based backend built with Flask  
- Real-time match detection from the database  

---

## 🧠 Tech Stack

**Backend**
- Python  
- Flask  
- DeepFace  
- MySQL  

**Frontend**
- HTML  
- CSS  
- JavaScript  

**Other Tools**
- Gmail SMTP  
- Git & GitHub  

---

## 🏗️ System Architecture
1. User uploads an image through the web interface  
2. Image is processed using DeepFace  
3. Face embeddings (numerical vectors) are generated  
4. Embeddings are compared against stored database images  
5. Best matching results are returned  
6. OTP-based authentication ensures secure access  

---

## 🧪 How It Works
- The system extracts facial features using DeepFace  
- Each face is converted into a numerical embedding  
- Embeddings are compared against a dataset of stored faces  
- A similarity score determines the closest match  

---

## 📂 Project Structure

Find-Me-AI-Missing-Person-System
│
├── app.py # Main application entry point
├── templates/ # HTML files
├── static/ # CSS, JS, and images
├── database/ # Database connection and scripts
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## ▶️ How to Run

1. Clone the repository  
git clone https://github.com/Pradhyumnajain23/Find-Me-AI-Missing-Person-System.git

2. Install dependencies  
pip install -r requirements.txt

3. Run the application  
python app.py

4. Open in browser  
http://localhost:5000

---

## 🎯 Use Case
This system can be used by:
- NGOs working with missing persons  
- Law enforcement agencies  
- Missing person organizations to automate and speed up searches  

---

## 👨‍💻 Author
**Pradyumna J Jain**  
MCA Student – 2026  
GitHub: https://github.com/Pradhyumnajain23
