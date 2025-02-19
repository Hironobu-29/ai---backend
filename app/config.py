import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/ai_receptionist")

    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    FACE_FOLDER = os.getenv(UPLOAD_FOLDER, "faces")
    ID_FOLDER = os.getenv(UPLOAD_FOLDER, "ids")