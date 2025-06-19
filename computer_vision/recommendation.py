import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import shutil
import numpy as np
from PIL import Image
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse

# Skip TensorFlow completely for deployment
TENSORFLOW_AVAILABLE = False
print("⚠️ TensorFlow disabled - using simplified mode for deployment")

# API URL configuration
apiUrl = os.getenv('RENDER_EXTERNAL_URL', 'http://localhost:8000')

user_img_path_recommend = 'images/upload/recommendation/user_image.jpg'
recommended_img_path = 'images/results/recommendation/'

# Skip model loading for deployment
feature_extractor = None
print("⚠️ Model loading disabled for deployment compatibility")

classes = {'0': 'Trousers', '1': 'Dress', '2': 'Sweater', '3': 'T-shirt', '4': 'Top', '5': 'Blouse'}

# Create demo data for testing
print("🔄 Creating demo data for deployment...")
demo_embeddings = [np.random.rand(512).tolist() for _ in range(30)]
demo_labels = ['0', '1', '2', '3', '4', '5'] * 5
demo_paths = [f'images/demo/item_{i}.jpg' for i in range(30)]
demo_ids = [f'demo_{i:03d}' for i in range(30)]

data = {
    'id': demo_ids,
    'embedding': demo_embeddings,
    'label': demo_labels,
    'path': demo_paths
}
print("✅ Demo data created successfully")

# Valid combinations for recommendations
valid_combinations = {
    '0': ['2', '3', '4', '5'],
    '1': ['2'],
    '2': ['0', '1', '5'],
    '3': ['0'],
    '4': ['0'],
    '5': ['0', '2'],
}

# ===| Helper functions |===
def preprocess_image(image_path, image_size=(224, 224)):
    """Image preprocessing"""
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize(image_size)
        image_array = np.array(image) / 255.0  # Normalize to [0, 1]
        return np.expand_dims(image_array, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"⚠️ Error preprocessing image: {e}")
        return np.random.rand(1, 224, 224, 3)

def extract_features(image_path, model):
    """Extract features from image"""
    print("🔄 Using dummy features (no model available)")
    return np.random.rand(1, 6), np.random.randint(0, 6)

def get_valid_items(input_class, data):
    """Get valid items based on input class"""
    valid_classes = valid_combinations.get(input_class, [])
    
    valid_items_embeddings = []
    valid_items_labels = []
    valid_items_paths = []
    
    # Filter items based on valid classes
    for i, label in enumerate(data['label']):
        if str(label) in valid_classes:
            valid_items_embeddings.append(data['embedding'][i])
            valid_items_labels.append(data['label'][i])
            valid_items_paths.append(data['path'][i])
    
    # If no valid items found, use first 10 items as fallback
    if not valid_items_embeddings:
        print("⚠️ No valid combinations found, using fallback items")
        valid_items_embeddings = data['embedding'][:10]
        valid_items_labels = data['label'][:10]
        valid_items_paths = data['path'][:10]
    
    return np.array(valid_items_embeddings), valid_items_labels, valid_items_paths

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    try:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    except:
        return 0.5  # Return neutral similarity if calculation fails

def simple_knn_recommend(query_embedding, valid_items_embeddings, valid_items_labels, valid_items_paths, k=5):
    """Simple KNN recommendation without scikit-learn"""
    if len(valid_items_embeddings) == 0:
        return []
    
    k = min(k, len(valid_items_embeddings))
    
    try:
        # Calculate similarities manually
        similarities = []
        query_flat = query_embedding.flatten()
        
        for i, item_embedding in enumerate(valid_items_embeddings):
            item_flat = np.array(item_embedding).flatten()
            # Ensure same dimensions
            min_len = min(len(query_flat), len(item_flat))
            similarity = cosine_similarity(query_flat[:min_len], item_flat[:min_len])
            similarities.append((similarity, i))
        
        # Sort by similarity (highest first)
        similarities.sort(reverse=True)
        top_k_indices = [idx for _, idx in similarities[:k]]
        top_k_similarities = [sim for sim, _ in similarities[:k]]
        
        recommended_paths = [valid_items_paths[idx] for idx in top_k_indices]
        
        paths = []
        # Create destination directory if it doesn't exist
        os.makedirs(recommended_img_path, exist_ok=True)
        
        # Create placeholder images
        for i, file_path in enumerate(recommended_paths):
            dest_path = f'{recommended_img_path}recommended_{i}.jpg'
            
            try:
                # Create a simple placeholder image
                placeholder = Image.new('RGB', (224, 224), color=(200, 200, 200))
                placeholder.save(dest_path)
                print(f"✅ Created placeholder image: {dest_path}")
            except Exception as e:
                print(f"⚠️ Could not create placeholder: {e}")
            
            paths.append(dest_path)

        recommended_labels_name = []
        for idx in top_k_indices:
            num = valid_items_labels[idx]
            recommended_labels_name.append(classes.get(str(num), 'Unknown'))

        print(f"🎯 Similarities: {top_k_similarities}")
        
        # Create recommendations
        recommendations = [
            {
                'similarity': str(1 - top_k_similarities[i]),  # Convert to distance-like metric
                'path': f'{apiUrl}/{paths[i]}',
                'label': recommended_labels_name[i]
            }
            for i in range(len(paths))
        ]

        return recommendations
        
    except Exception as e:
        print(f"⚠️ Error in KNN recommendation: {e}")
        return []

def recommend_outfit(file: UploadFile = File(...)):
    """Main recommendation function"""
    try:
        # Extract features
        query_image_embeddings, query_image_class = extract_features(user_img_path_recommend, feature_extractor)

        query_image_class = str(query_image_class)
        print(f'🔍 Query image class: {query_image_class}')

        # Get valid items
        valid_items_embeddings, valid_items_labels, valid_items_paths = get_valid_items(query_image_class, data)
        if len(valid_items_embeddings) == 0:
            return JSONResponse(content={
                "message": 'No valid recommendations found for the query class'
            })

        # Find recommendations using simple KNN
        recommendations = simple_knn_recommend(query_image_embeddings, valid_items_embeddings, valid_items_labels, valid_items_paths)

        print(f'📂 User image path: {user_img_path_recommend}')
        print(f'🏷️ Query image class: {query_image_class}')
        print(f'📋 Recommendations count: {len(recommendations)}')
        
        # Return recommendations in JSON format
        return JSONResponse(content={
            "user_image_class": f'{classes.get(query_image_class, "Unknown")}',
            "recommendations": recommendations,
            "system_status": "working_without_ml_libraries",
            "total_recommendations": len(recommendations)
        })
        
    except Exception as e:
        print(f"❌ Error in recommend_outfit: {str(e)}")
        return JSONResponse(content={
            "error": f"Error in recommendation: {str(e)}",
            "system_status": "error"
        }, status_code=500)
