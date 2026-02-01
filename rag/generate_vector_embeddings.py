import os
import time
import json
from datetime import datetime
import torch
from pymongo import MongoClient
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

# ============================================
# CONFIGURATION
# ============================================
try:
    from config import MONGODB_URI, DB_NAME, COLLECTION_NAME, MODEL_NAME
    from constants import CATEGORY_LABELS
except ImportError:
    # Handle execution from parent directory
    from .config import MONGODB_URI, DB_NAME, COLLECTION_NAME, MODEL_NAME
    from .constants import CATEGORY_LABELS

# ============================================
# CATEGORY MAPPING
# ============================================
# CATEGORY_LABELS imported from constants.py

# ============================================
# TEXT GENERATION FUNCTION
# ============================================
def create_embedding_text(food_doc):
    """
    Tạo rich text content từ food document để embedding

    Kết hợp các trường quan trọng:
    - Name (tên món)
    - Description (mô tả)
    - Categories (danh mục)
    - Directions (hướng dẫn)
    - Properties (thuộc tính: meal type, dietary tags, cooking method)
    - Major ingredients (nguyên liệu chính)
    """
    parts = []

    # 1. Name (quan trọng nhất)
    name = food_doc.get('name', '')
    if name:
        parts.append(f"Name: {name}")

    # 2. Description
    desc = food_doc.get('description')
    if desc and str(desc) not in ['nan', 'NaN', '']:
        parts.append(f"Description: {desc}")

    # 3. Categories
    categories = food_doc.get('categories', [])
    if categories and isinstance(categories, list):
        category_names = [CATEGORY_LABELS.get(cat, '') for cat in categories]
        category_names = [c for c in category_names if c]  # Remove empty strings
        if category_names:
            parts.append(f"Categories: {', '.join(category_names)}")

    # 4. Directions (cách làm)
    directions = food_doc.get('directions', [])
    if directions and isinstance(directions, list):
        # Ghép các bước, giới hạn độ dài
        directions_text = ' '.join(directions)[:500]
        parts.append(f"Instructions: {directions_text}")

    # 5. Properties
    prop = food_doc.get('property', {})

    # Meal types
    meal_types = []
    if prop.get('isBreakfast'): meal_types.append('breakfast')
    if prop.get('isLunch'): meal_types.append('lunch')
    if prop.get('isDinner'): meal_types.append('dinner')
    if prop.get('isSnack'): meal_types.append('snack')
    if prop.get('isDessert'): meal_types.append('dessert')
    if meal_types:
        parts.append(f"Meal types: {', '.join(meal_types)}")

    # Dietary properties (Tags định tính - giữ lại vì giúp phân loại món ăn)
    dietary = []
    if prop.get('isHighProtein'): dietary.append('high protein')
    if prop.get('isLowCarb'): dietary.append('low carb')
    if prop.get('isLowFat'): dietary.append('low fat')
    if prop.get('isHighFiber'): dietary.append('high fiber')
    if prop.get('isLowSodium'): dietary.append('low sodium')
    if dietary:
        parts.append(f"Dietary: {', '.join(dietary)}")

    # Cooking methods
    cooking = []
    if prop.get('needsMicrowave'): cooking.append('microwave')
    if prop.get('needsOven'): cooking.append('oven')
    if prop.get('needsStove'): cooking.append('stove')
    if prop.get('needsGrill'): cooking.append('grill')
    if prop.get('needsBlender'): cooking.append('blender')
    if prop.get('needsSlowCooker'): cooking.append('slow cooker')
    if cooking:
        parts.append(f"Cooking methods: {', '.join(cooking)}")

    # Time info
    total_time = prop.get('totalTime')
    if total_time and total_time > 0:
        parts.append(f"Total time: {total_time} minutes")

    complexity = prop.get('complexity')
    if complexity:
        if complexity < 3:
            parts.append("Difficulty: very easy")
        elif complexity < 5:
            parts.append("Difficulty: easy")
        elif complexity < 7:
            parts.append("Difficulty: medium")
        else:
            parts.append("Difficulty: hard")

    # Dish type
    dish_types = []
    if prop.get('mainDish'): dish_types.append('main dish')
    if prop.get('sideDish'): dish_types.append('side dish')
    if dish_types:
        parts.append(f"Dish type: {', '.join(dish_types)}")

    # Major ingredients - FIX: Xử lý an toàn kiểu dữ liệu
    major_ing = prop.get('majorIngredients', '')
    if major_ing:
        # Kiểm tra kiểu dữ liệu trước khi gọi replace()
        if isinstance(major_ing, str):
            # Clean up: "microwaved-sweet-potato" → "microwaved sweet potato"
            major_ing_clean = major_ing.replace('-', ' ')
            parts.append(f"Main ingredients: {major_ing_clean}")
        elif major_ing and str(major_ing) not in ['nan', 'NaN']:
            # Nếu không phải string nhưng có giá trị hợp lệ, convert sang string
            parts.append(f"Main ingredients: {str(major_ing)}")

    # Combine all parts
    text = '. '.join(parts)
    return text

# ============================================
# MAIN EXECUTION
# ============================================
def main():
    print("🚀 Starting Food Embeddings Re-computation...")

    # 1. Initialize MongoDB
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    total_docs = collection.count_documents({})
    print(f"📊 Connected to MongoDB. Total documents: {total_docs}")

    # 2. Initialize Model
    print(f"🧠 Loading embedding model: {MODEL_NAME}...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 3. Process Documents
    print("\n🔄 Processing documents...")
    updated_count = 0
    error_count = 0

    cursor = collection.find({})

    for doc in cursor:
        try:
            # Generate new text content
            text_content = create_embedding_text(doc)

            # Generate embedding
            embedding = embeddings_model.embed_query(text_content)

            # Update Document
            collection.update_one(
                {"_id": doc["_id"]},
                {
                    "$set": {
                        "text_content": text_content,
                        "embedding": embedding,
                        "embedding_updated_at": datetime.utcnow()
                    }
                }
            )

            updated_count += 1
            if updated_count % 10 == 0:
                print(f"   Processed {updated_count}/{total_docs} docs...")

        except Exception as e:
            error_count += 1
            print(f"   ❌ Error processing document {doc.get('_id')}: {e}")

    print("\n" + "="*50)
    print("✅ RECOMPUTATION COMPLETE")
    print("="*50)
    print(f"Total processed: {updated_count}")
    print(f"Errors: {error_count}")

    client.close()

if __name__ == "__main__":
    main()
