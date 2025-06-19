# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# import numpy as np
# import cv2
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# from fastapi.responses import JSONResponse



# model = torch.load('models/segmentation_model.pth')

# apiUrl = 'http://0.0.0.0:8000'
# user_img_path_segment = 'images/upload/segmentation/user_image.jpg'
# segmented_img_path = 'images/results/segmentation/segmented_image.jpg'

# # Define your transformations
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     # Note: Resize is handled in the dataset class now
# ])

# resize_dim = (256, 256)  # Define your desired size for images and masks
# device = torch.device("cpu")

# # ===| Helper functions |===
# # Image preprocessing
# def getitem():
#     image = cv2.imread(user_img_path_segment)
#     if image is None:
#         raise FileNotFoundError(f"Image not found: {user_img_path_segment}")

#     # Resize the image and mask to fixed dimensions
#     image = cv2.resize(image, resize_dim, interpolation = cv2.INTER_LINEAR)

#     # Apply transformations if specified
#     if transform:
#         image = transform(image)

#     return image

# def visualize_predictions(model, image):
#     model = model.to(device)
#     model.eval()
#     image = image.to(device)
#     image = image.unsqueeze(0)  # Add batch dimension

#     # Make predictions
#     with torch.no_grad():
#         output = model(image)['out']
#         pred = torch.argmax(output, dim=1)

#     # Move images, masks, and predictions back to CPU for visualization
#     image = image.cpu().numpy()
#     pred = pred.cpu().numpy()

#     image = image.squeeze(0)
#     pred = pred.squeeze(0)

#     # Save the prediction mask as a .jpg image
#     pred_normalized = (pred - pred.min()) / (pred.max() - pred.min()) * 255  # Normalize to [0, 255]
#     pred_normalized = pred_normalized.astype(np.uint8)
#     pred_image = Image.fromarray(pred_normalized)
#     pred_image.save(segmented_img_path)  # Save the prediction as a .jpg file


#     # Set up the plot
#     # _, axes = plt.subplots(1, 2, figsize=(15, 5))

#     # Original Image
#     # axes[0].imshow(image.transpose(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
#     # axes[0].set_title('Original Image')
#     # axes[0].axis('off')

#     # Predicted Mask
#     # axes[1].imshow(pred, cmap='jet', alpha=0.5)
#     # axes[1].set_title('Predicted Mask')
#     # axes[1].axis('off')

#     # plt.tight_layout()
#     # plt.show()


# def segment_image():
#     img = getitem()

#     # Visualization of predictions
#     visualize_predictions(model, img)

#     return JSONResponse(content={"original": f'{apiUrl}/{user_img_path_segment}', "segmented": f'{apiUrl}/{segmented_img_path}'}, status_code=200)
