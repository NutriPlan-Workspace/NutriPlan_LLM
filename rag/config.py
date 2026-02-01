import os

os.environ["OPENAI_BASE_URL"] = "http://localhost:8001/v1"

# MongoDB Config
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "test"
COLLECTION_NAME = "foods"

# Model Config
MODEL_NAME = "intfloat/multilingual-e5-small"

