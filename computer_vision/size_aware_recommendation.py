import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import shutil
import numpy as np
from PIL import Image
import cv2
from sklearn.neighbors import NearestNeighbors
from fastapi.responses import JSONResponse

# Try to import TensorFlow
# Skip TensorFlow completely for deployment
TENSORFLOW_AVAILABLE = False
print("⚠️ TensorFlow disabled for deployment compatibility")
 
# Import from your existing recommendation file
try:
    from .recommendation import (
        feature_extractor, data, classes, apiUrl, 
        user_img_path_recommend, recommended_img_path,
        preprocess_image, extract_features, get_valid_items, simple_knn_recommend
    )
except ImportError:
    from computer_vision.recommendation import (
        feature_extractor, data, classes, apiUrl, 
        user_img_path_recommend, recommended_img_path,
        preprocess_image, extract_features, get_valid_items, simple_knn_recommend
    )
except ImportError:
    from computer_vision.recommendation import (
        feature_extractor, data, classes, apiUrl, 
        user_img_path_recommend, recommended_img_path,
        preprocess_image, extract_features, get_valid_items, knn_recommend
    )

# Size compatibility rules - this is our "AI brain" for sizes
SIZE_COMPATIBILITY = {
    'S': {
        'avoid_styles': ['oversized', 'baggy', 'loose_fit'],
        'recommend_styles': ['fitted', 'slim', 'tailored'],
        'size_score_boost': 0.2  # Boost score for fitted items
    },
    'M': {
        'avoid_styles': ['very_tight'],
        'recommend_styles': ['fitted', 'regular', 'relaxed'],
        'size_score_boost': 0.1
    },
    'L': {
        'avoid_styles': ['very_tight', 'slim_fit'],
        'recommend_styles': ['regular', 'relaxed', 'comfortable'],
        'size_score_boost': 0.1
    },
    'XL': {
        'avoid_styles': ['tight', 'slim_fit', 'fitted'],
        'recommend_styles': ['relaxed', 'comfortable', 'loose'],
        'size_score_boost': 0.2
    },
    'XXL': {
        'avoid_styles': ['tight', 'slim_fit', 'fitted'],
        'recommend_styles': ['relaxed', 'comfortable', 'loose', 'plus_size'],
        'size_score_boost': 0.3
    }
}

BODY_TYPE_RULES = {
    'slim': {
        'avoid': ['baggy', 'oversized'],
        'recommend': ['fitted', 'tailored', 'structured'],
        'style_boost': 0.15
    },
    'regular': {
        'avoid': ['too_tight', 'too_loose'],
        'recommend': ['regular_fit', 'classic', 'standard'],
        'style_boost': 0.1
    },
    'curvy': {
        'avoid': ['shapeless', 'boxy'],
        'recommend': ['wrap', 'fitted_waist', 'a_line'],
        'style_boost': 0.2
    },
    'plus': {
        'avoid': ['tight', 'small_sizes'],
        'recommend': ['comfortable', 'flattering', 'plus_friendly'],
        'style_boost': 0.25
    }
}

def load_user_preferences():
    """Load user size preferences"""
    try:
        with open('user_preferences.json', 'r') as f:
            return json.load(f)
    except:
        return {'size': 'M', 'body_type': 'regular'}

def analyze_clothing_fit_style(image_path):
    """
    Analyze if clothing appears fitted, regular, or loose
    This is a simple version - we can make it smarter later
    """
    try:
        # Remove the API URL part to get actual file path
        if image_path.startswith('http'):
            image_path = image_path.replace(f'{apiUrl}/', '')
        
        image = cv2.imread(image_path)
        if image is None:
            return 'regular'
        
        # Simple analysis based on image dimensions
        height, width = image.shape[:2]
        aspect_ratio = width / height
        
        # Basic heuristic (we'll improve this later)
        if aspect_ratio > 0.8:
            return 'loose'
        elif aspect_ratio < 0.6:
            return 'fitted'
        else:
            return 'regular'
    except:
        return 'regular'

def calculate_size_compatibility_score(item_style, user_size, user_body_type):
    """
    Calculate how well an item matches user's size needs
    Returns score between 0-1 (1 = perfect match)
    """
    size_rules = SIZE_COMPATIBILITY.get(user_size, SIZE_COMPATIBILITY['M'])
    body_rules = BODY_TYPE_RULES.get(user_body_type, BODY_TYPE_RULES['regular'])
    
    base_score = 0.5  # Default neutral score
    
    # Penalty for avoid styles
    if item_style in size_rules['avoid_styles']:
        base_score -= 0.3
    
    if item_style in body_rules['avoid']:
        base_score -= 0.2
    
    # Bonus for recommended styles
    if item_style in size_rules['recommend_styles']:
        base_score += size_rules['size_score_boost']
    
    if item_style in body_rules['recommend']:
        base_score += body_rules['style_boost']
    
    # Keep score between 0 and 1
    return max(0, min(1, base_score))

def enhance_recommendations_with_size(recommendations, user_size, user_body_type):
    """
    Add size information and adjust scores for each recommendation
    """
    enhanced_recommendations = []
    
    for item in recommendations:
        # Analyze the clothing style
        item_style = analyze_clothing_fit_style(item['path'])
        
        # Calculate size compatibility
        size_score = calculate_size_compatibility_score(item_style, user_size, user_body_type)
        
        # Adjust the similarity score based on size compatibility
        original_similarity = float(item['similarity'])
        adjusted_similarity = original_similarity * (0.7 + 0.3 * size_score)  # Weight: 70% original, 30% size
        
        # Add size information to the item
        enhanced_item = item.copy()
        enhanced_item.update({
            'fit_style': item_style,
            'size_compatibility_score': round(size_score, 2),
            'original_similarity': round(original_similarity, 3),
            'adjusted_similarity': round(adjusted_similarity, 3),
            'size_appropriate': size_score > 0.6,
            'size_explanation': generate_size_explanation(item_style, user_size, user_body_type, size_score)
        })
        
        enhanced_recommendations.append(enhanced_item)
    
    # Sort by adjusted similarity (best matches first)
    enhanced_recommendations.sort(key=lambda x: x['adjusted_similarity'], reverse=True)
    
    return enhanced_recommendations

def generate_size_explanation(item_style, user_size, user_body_type, score):
    """Generate human-readable explanation for size matching"""
    if score > 0.8:
        return f"Perfect fit! This {item_style} style is ideal for {user_size} size and {user_body_type} body type."
    elif score > 0.6:
        return f"Good match. This {item_style} style works well for {user_size} size."
    elif score > 0.4:
        return f"Okay fit. This {item_style} style might work for {user_size} size."
    else:
        return f"May not be ideal. This {item_style} style might not be the best for {user_size} size."

def recommend_outfit_with_size():
    """
    Main function: Get recommendations and enhance them with size awareness
    """
    # Load user preferences
    preferences = load_user_preferences()
    user_size = preferences.get('size', 'M')
    user_body_type = preferences.get('body_type', 'regular')
    
    print(f"Recommending for size: {user_size}, body type: {user_body_type}")
    
    # Get regular recommendations using existing system
    try:
        query_image_embeddings, query_image_class = extract_features(user_img_path_recommend, feature_extractor)
        query_image_class = str(query_image_class)
        
        # Get valid items
        valid_items_embeddings, valid_items_labels, valid_items_paths = get_valid_items(query_image_class, data)
        
        if valid_items_embeddings.size == 0:
            return JSONResponse(content={
                "message": 'No valid recommendations found for the query class'
            })
        
        # Get initial recommendations (more items to filter from)
        initial_recommendations = knn_recommend(
            query_image_embeddings, 
            valid_items_embeddings, 
            valid_items_labels, 
            valid_items_paths, 
            k=10  # Get 10 items first
        )
        
        # Enhance with size information
        size_enhanced_recommendations = enhance_recommendations_with_size(
            initial_recommendations, 
            user_size, 
            user_body_type
        )
        
        # Take top 5 after size enhancement
        final_recommendations = size_enhanced_recommendations[:5]
        
        # Calculate statistics
        size_appropriate_count = len([r for r in size_enhanced_recommendations if r['size_appropriate']])
        
        return JSONResponse(content={
            "user_image_class": f'{classes[query_image_class]}',
            "user_size": user_size,
            "user_body_type": user_body_type,
            "total_analyzed": len(initial_recommendations),
            "size_appropriate_found": size_appropriate_count,
            "recommendations": final_recommendations,
            "size_system_active": True
        })
        
    except Exception as e:
        print(f"Error in size-aware recommendation: {str(e)}")
        return JSONResponse(content={
            "error": f"Error in recommendation: {str(e)}"
        }, status_code=500)