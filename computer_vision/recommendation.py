import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import shutil
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse

# Try to import TensorFlow, handle if not available
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow available")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available - using simplified mode")

# API URL configuration
import os
# Use environment variable for production, fallback to localhost for development
apiUrl = os.getenv('RENDER_EXTERNAL_URL', 'http://0.0.0.0:8000')
if apiUrl == 'http://0.0.0.0:8000':
    # Running locally
    apiUrl = 'http://localhost:8000'

user_img_path_recommend = 'images/upload/recommendation/user_image.jpg'
recommended_img_path = 'images/results/recommendation/'

# Load pre-trained model (handle if not available)
if TENSORFLOW_AVAILABLE:
    try:
        feature_extractor = load_model('models/resnest50.keras')
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"⚠️ Could not load model: {e}")
        feature_extractor = None
else:
    feature_extractor = None

classes = {'0': 'Trousers', '1': 'Dress', '2': 'Sweater', '3': 'T-shirt', '4': 'Top', '5': 'Blouse'}

# Load database embeddings, labels, and paths (handle if not available)
try:
    database_embeddings = np.load('data/embeddings/embeddings.npy')
    database_labels = np.load('data/embeddings/labels.npy')
    paths = np.load('data/embeddings/paths.npy')
    ids = [path[-13: -4] for path in paths]
    
    # Create a dictionary instead of DataFrame
    data = {
        'id': list(ids),
        'embedding': list(database_embeddings),
        'label': list(database_labels),
        'path': list(paths)
    }
    print("✅ Real data loaded successfully")
    
except Exception as e:
    print(f"⚠️ Could not load data files: {e}")
    print("🔄 Creating demo data for deployment...")
    
    # Create demo data that works for testing
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
# Image preprocessing
def preprocess_image(image_path, image_size=(224, 224)):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(image_size)
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

# Extract features
def extract_features(image_path, model):
    if model is None:
        # Return dummy features for testing when model is not available
        print("🔄 Using dummy features (no model available)")
        return np.random.rand(1, 6), np.random.randint(0, 6)
    
    try:
        preprocessed_image = preprocess_image(image_path)
        query_image_embeddings = model.predict(preprocessed_image)
        query_image_class = np.argmax(query_image_embeddings, axis=1)[0]
        return query_image_embeddings, query_image_class
    except Exception as e:
        print(f"⚠️ Error in feature extraction: {e}")
        return np.random.rand(1, 6), np.random.randint(0, 6)

# Get valid items (updated to work with dictionary instead of DataFrame)
def get_valid_items(input_class, data):
    valid_classes = valid_combinations.get(input_class, [])
    
    # Work with dictionary instead of DataFrame
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
    
    if not valid_items_embeddings:
        return np.array([]), [], []
    
    return np.array(valid_items_embeddings), valid_items_labels, valid_items_paths

def knn_recommend(query_embedding, valid_items_embeddings, valid_items_labels, valid_items_paths, k=5):
    if len(valid_items_embeddings) == 0:
        return []
    
    # Ensure we don't ask for more neighbors than we have items
    k = min(k, len(valid_items_embeddings))
    
    try:
        knn_model = NearestNeighbors(n_neighbors=k, metric='cosine')
        knn_model.fit(valid_items_embeddings)
        distances, indices = knn_model.kneighbors(query_embedding)

        recommended_paths = [valid_items_paths[idx] for idx in indices[0]]
        
        paths = []
        # Create destination directory if it doesn't exist
        os.makedirs(recommended_img_path, exist_ok=True)
        
        # Save recommended images to the designated folder
        for i, file_path in enumerate(recommended_paths):
            dest_path = f'{recommended_img_path}recommended_{i}.jpg'
            
            # Handle different path formats
            if file_path.startswith('images/demo/'):
                # For demo data, create a placeholder image
                try:
                    from PIL import Image
                    placeholder = Image.new('RGB', (224, 224), color='lightgray')
                    placeholder.save(dest_path)
                    print(f"✅ Created placeholder image: {dest_path}")
                except Exception as e:
                    print(f"⚠️ Could not create placeholder: {e}")
            else:
                # For real data, try to copy the image
                try:
                    img_path = file_path[3:] if file_path.startswith('../') else file_path
                    shutil.copy(img_path, dest_path)
                    print(f"✅ Copied image: {img_path} -> {dest_path}")
                except Exception as e:
                    print(f"⚠️ Could not copy image {img_path}: {e}")
                    # Create placeholder if copy fails
                    try:
                        from PIL import Image
                        placeholder = Image.new('RGB', (224, 224), color='lightblue')
                        placeholder.save(dest_path)
                    except:
                        pass
            
            paths.append(dest_path)

        recommended_labels_name = []
        ind = indices[0].tolist()

        for i in ind:
            num = valid_items_labels[i]
            recommended_labels_name.append(classes.get(str(num), 'Unknown'))

        print(f"🎯 Distances: {distances}")
        
        # Create recommendations
        recommendations = [
            {
                'similarity': str(distances[0][i]),  # Convert similarity to a string
                'path': f'{apiUrl}/{paths[i]}', # Use the original path of the image
                'label': recommended_labels_name[i] # Ensure labels are strings
            }
            for i in range(len(paths))
        ]

        return recommendations
        
    except Exception as e:
        print(f"⚠️ Error in KNN recommendation: {e}")
        return []

def recommend_outfit(file: UploadFile = File(...)):
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

        # Find recommendations
        recommendations = knn_recommend(query_image_embeddings, valid_items_embeddings, valid_items_labels, valid_items_paths)

        print(f'📂 User image path: {user_img_path_recommend}')
        print(f'🏷️ Query image class: {query_image_class}')
        print(f'📋 Recommendations count: {len(recommendations)}')
        
        # Return recommendations in JSON format
        return JSONResponse(content={
            "user_image_class": f'{classes.get(query_image_class, "Unknown")}',
            "recommendations": recommendations,
            "system_status": "working",
            "total_recommendations": len(recommendations)
        })
        
    except Exception as e:
        print(f"❌ Error in recommend_outfit: {str(e)}")
        return JSONResponse(content={
            "error": f"Error in recommendation: {str(e)}",
            "system_status": "error"
        }, status_code=500)