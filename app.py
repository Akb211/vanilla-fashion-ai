import shutil
import os
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from computer_vision.recommendation import recommend_outfit
# from computer_vision.segmentation import segment_image

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

# Mount images Files
os.makedirs("images", exist_ok=True)
os.makedirs("images/upload/recommendation", exist_ok=True)
os.makedirs("images/results/recommendation", exist_ok=True)

app.mount("/images", StaticFiles(directory="images"), name="images")

import os
# Use environment variable for production, fallback to localhost for development
apiUrl = os.getenv('RENDER_EXTERNAL_URL', 'http://0.0.0.0:8000')
if apiUrl == 'http://0.0.0.0:8000':
    # Running locally
    apiUrl = 'http://localhost:8000'
user_img_path_recommend = 'images/upload/recommendation/user_image.jpg'
user_img_path_segment = 'images/upload/segmentation/user_image.jpg'

# ===| FastAPI Routes |===

# Home route
@app.get("/", response_class=HTMLResponse)
async def home():
    return JSONResponse(content={"message": "Welcome to the Fashion Recommendation API!"}, status_code=200)

# ===| Original Recommendation route |===
@app.post("/recommend")
async def recommend(file: UploadFile = File(...)):
    return recommend_outfit()

# ===| Size-aware recommendation routes |===
@app.post("/upload_image_with_size")
async def upload_image_with_size(
    file: UploadFile = File(...),
    user_size: str = Form("M"),
    body_type: str = Form("regular")
):
    # Save uploaded file
    with open(user_img_path_recommend, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Save user size preferences to a simple JSON file
    user_preferences = {
        'size': user_size,
        'body_type': body_type,
        'image_url': f'{apiUrl}/{user_img_path_recommend}'
    }
    
    with open('user_preferences.json', 'w') as f:
        json.dump(user_preferences, f)

    return JSONResponse(content=user_preferences, status_code=200)

@app.post("/recommend_with_size")
async def recommend_with_size():
    from computer_vision.size_aware_recommendation import recommend_outfit_with_size
    return recommend_outfit_with_size()

# ===| Upload Image Recommendation route |===
@app.post("/upload_image_recommend")
async def upload_image_recommend(file: UploadFile = File(...)):
    # Save uploaded file
    with open(user_img_path_recommend, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return JSONResponse(content={'imageUrl': f'{apiUrl}/{user_img_path_recommend}'}, status_code=200)

# ===| Upload Image Segmentation route |===
@app.post("/upload_image_segment")
async def upload_image_segment(file: UploadFile = File(...)):
    # Save uploaded file
    with open(user_img_path_segment, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return JSONResponse(content={'imageUrl': f'{apiUrl}/{user_img_path_segment}'}, status_code=200)

# ===| Segmentation route |===
# @app.post("/segment")
# async def segmentation():
#     return segment_image()

# ===| Start the FastAPI app |===
# uvicorn.run('app', host='0.0.0.0', port=8000, workers=2) 
