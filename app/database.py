from flask_pymongo import PyMongo
from .config import Config

mongo = PyMongo()
customers = None

def init_db(app):
    global customers
    app.config["MONGO_URI"] = Config.MONGO_URI
    mongo.init_app(app)
    customers = mongo.db.customers

from pymongo import MongoClient

# MongoDB �ɐڑ�
client = MongoClient("mongodb+srv://root:123@cluster0.zwdmz.mongodb.net/ai-receptionist?retryWrites=true&w=majority&appName=Cluster0")
db = client["ai-receptionist"]
collection = db["ocr_data"]

def save_extracted_text(user_id, text):
    """ ���o�����e�L�X�g�� MongoDB �ɕۑ� """
    collection.insert_one({"user_id": user_id, "text": text})

def get_saved_texts():
    """ �ۑ����ꂽ�f�[�^���擾 """
    return list(collection.find({}, {"_id": 0, "user_id": 1, "text": 1}))
