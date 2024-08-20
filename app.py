import os
import random
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import seaborn as sns
import pandas as pd
import uuid
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Query, Form, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse

from fastapi.staticfiles import StaticFiles
from pathlib import Path
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

class CacheControlMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

app = FastAPI()
app.add_middleware(CacheControlMiddleware)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Azure Blob Service Client
connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

if not connection_string:
    raise EnvironmentError("AZURE_STORAGE_CONNECTION_STRING environment variable not set.")
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

container_name = os.getenv('CONTAINER_NAME')
container_client = blob_service_client.get_container_client(container_name)

# Function to list blobs and generate image URLs from the final_imgs folder
def get_image_blobs() -> Dict[str, List[str]]:
    image_folders = {
        "real_calc": "real_calc",
        "real_normal": "real_normal",
        "synth_gan_calc": "synth_gan_calc",
        "synth_gan_normal": "synth_gan_normal",
        "synth_diff_normal": "synth_diff_normal",
        "synth_diff_calc": "synth_diff_calc",
    }

    all_images = []
    for category, folder in image_folders.items():
        # print(f"Fetching images from folder: {folder}")
        blobs = container_client.list_blobs(name_starts_with=f"final_imgs/{folder}/")
        for blob in blobs:
            if blob.name.endswith(('.png', '.jpg', '.jpeg')):
                # Generate SAS URL for blob access
                sas_token = generate_blob_sas(
                    account_name=blob_service_client.account_name,
                    container_name=container_name,
                    blob_name=blob.name,
                    account_key=blob_service_client.credential.account_key,
                    permission=BlobSasPermissions(read=True),
                    expiry=datetime.utcnow() + timedelta(hours=1)  # 1 hour validity
                )
                image_url = f"{blob_service_client.get_blob_client(container_name, blob.name).url}?{sas_token}"
                all_images.append((image_url, category))
                # print(f"Found image: {image_url} in category: {category}")
    
    return all_images

SHUFFLE_SEED = 42

# Function to shuffle images deterministically
def deterministic_shuffle(images: List[Tuple[str, str]], seed: int) -> List[Tuple[str, str]]:
    # Initialize the random number generator with the seed
    rng = random.Random(seed)
    # Create a copy of the image list to shuffle
    images_copy = images[:]
    rng.shuffle(images_copy)
    return images_copy

# TOTAL IMAGES TO DISPLAY
TOTAL_IMAGES = int(os.getenv('TOTAL_IMAGES'))

# Gather and balance images from the blobs
all_images = get_image_blobs()

# Balance the categories
def balance_images(images, total_images_per_category):
    categories = {}
    for image, category in images:
        if category not in categories:
            categories[category] = []
        categories[category].append((image, category))

    balanced_images = []
    for category_images in categories.values():
        balanced_images.extend(category_images[:total_images_per_category])

    return balanced_images

# Calculate total number of images per category
num_categories = len(set(category for _, category in all_images))
images_per_category = TOTAL_IMAGES // num_categories
# Ensure balanced image distribution
all_images = balance_images(all_images, images_per_category)
# Shuffle images for randomness
all_images = deterministic_shuffle(all_images, SHUFFLE_SEED)

# Store timestamps
start_time = datetime.now()
image_start_time = None

# Function to get the next image
def get_next_image(index):
    global image_start_time, start_time
    if index < len(all_images):
        image_info = {
            "image_url": all_images[index][0],
            "category": all_images[index][1],
        }
        image_start_time = datetime.now()
        print(f"Loading image {index}: {image_info['image_url']}")
        print(f"Image {index} start time: {image_start_time}")
        print(f"Original start time: {start_time}")  # Debugging statement
        return image_info
    else:
        return {"error": "No more images, finalize the evaluation please"}

class UserInfo(BaseModel):
    first_name: str
    last_name: str

user_info_storage = {}

def load_user_data(first_name: str, last_name: str) -> Optional[Dict]:
    try:
        blobs = container_client.list_blobs(name_starts_with=f"evaluations_")
        for blob in blobs:
            print(blob.name)
            if f"_{first_name}_{last_name}.jsonl" in blob.name:
                user_id = blob.name.split('_')[1]
                print(f'detected already started test with user id: {user_id}')
                blob_client = container_client.get_blob_client(blob.name)
                
                download_stream = blob_client.download_blob()
                existing_content = download_stream.readall().decode('utf-8')
                evaluations = [json.loads(line) for line in existing_content.splitlines()]

                if evaluations:
                    last_evaluation = evaluations[-1]
                    last_index = last_evaluation['index']  # Extract the index directly from the evaluation data
                    print('detected last index', last_index, last_evaluation)
                    
                    return {
                        "user_id": user_id,
                        "last_index": last_index
                    }
    except Exception as e:
        # Blob does not exist or error occurred
        print(f"Error retrieving user data: {e}")
        return None

    return None

@app.post("/start_test")
async def start_test(user_info: UserInfo):
    # Check if user already exists
    user_data = load_user_data(user_info.first_name, user_info.last_name)
    
    if user_data:
        # User already exists, continue from where they left off
        user_id = user_data["user_id"]
        last_index = user_data["last_index"]

        # Store user information
        user_info_storage[user_id] = {
            "first_name": user_info.first_name,
            "last_name": user_info.last_name
        }
        return JSONResponse(content={"success": True, "user_id": user_id, "last_index": last_index})
    
    
    else:
        # New user, create a new entry
        user_id = str(uuid.uuid4())
        user_info_storage[user_id] = {
            "first_name": user_info.first_name,
            "last_name": user_info.last_name
        }
        return JSONResponse(content={"success": True, "user_id": user_id, "last_index": 0})


# @app.post("/start_test")
# async def start_test(user_info: UserInfo):
#     # Generate a unique identifier for the user
#     user_id = str(uuid.uuid4())

#     # Store user information
#     user_info_storage[user_id] = {
#         "first_name": user_info.first_name,
#         "last_name": user_info.last_name
#     }
    
#     # Return the unique identifier to the client
#     return JSONResponse(content={"success": True, "user_id": user_id})

# Example endpoint to get an image
@app.get("/next_image/{index}")
async def get_image(index: int):
    image_info = get_next_image(index)
    if "error" in image_info:
        return JSONResponse(content={"error": image_info["error"]}, status_code=404)
    return image_info

# Example endpoint to get an image
# @app.get("/image/{index}")
# async def get_image(index: int):
#     image_info = get_next_image(index)
#     if "error" in image_info:
#         return {"error": image_info["error"]}
#     return image_info

@app.get("/")
def read_root():
    return FileResponse("static/index0.html")  # Serve the initial page

i = 0
@app.get("/test")
def test_page():
    global start_time
    if not start_time:
        start_time = datetime.now()
        print(f"Test started at: {start_time}")  # Debugging statement
    else:
        print(f"Test already started at: {start_time}")
    return FileResponse("static/index.html")  # Serve the Turing test page

# @app.get("/next_image/{index}")
# def next_image(index: int):
#     image_info = get_next_image(index)
#     if "error" in image_info:
#         if all_evaluated():
#             return JSONResponse(
#                 content={"message": "No more images. You can finalize the evaluation."},
#                 status_code=200,
#             )
#         else:
#             return JSONResponse(content=image_info, status_code=404)
#     return JSONResponse(content=image_info)

@app.post("/evaluate/")
async def evaluate_image(
    user_id: str = Form(...),
    image_path: str = Form(...),
    category: str = Form(...),
    is_real: bool = Form(...),
    realism_score: int = Form(...),
    calcification_seen: str = Form(...),
    index: int = Form(...)
):
    print(f"Received data: user_id={user_id}, image_path={image_path}, category={category}, is_real={is_real}, realism_score={realism_score}, calcification_seen={calcification_seen}")

    global image_start_time
    end_time = datetime.now()
    duration = (end_time - image_start_time).total_seconds()

    # Convert calcification_seen to boolean if needed
    calcification_seen = calcification_seen.lower() in ['true', '1', 'yes']

    # Process and store the evaluation
    evaluation = {
        "user_id": user_id,
        "image_path": image_path,
        "category": category,
        "is_real": is_real,
        "realism_score": realism_score,
        "calcification_seen": calcification_seen,
        "image_duration": duration,
        "index": index,
        "timestamp": end_time.isoformat(),
    }

    # Construct the blob name using the user ID
    blob_name = f"evaluations_{user_id}_{user_info_storage[user_id]['first_name']}_{user_info_storage[user_id]['last_name']}.jsonl"

    # Get the blob client
    blob_client = container_client.get_blob_client(blob_name)

    # Download the existing content
    existing_content = ""
    try:
        download_stream = blob_client.download_blob()
        existing_content = download_stream.readall().decode('utf-8')
    except Exception as e:
        # If the blob does not exist, an exception will be thrown
        print(f"No existing blob found or error downloading blob: {e}")

    # Append new data
    updated_content = existing_content + json.dumps(evaluation) + "\n"
    
    # Upload the updated content
    blob_client.upload_blob(updated_content, blob_type="BlockBlob", overwrite=True)

    # Print the updated content for debugging
    print(f"Current content of the blob '{blob_name}':")
    print(updated_content)

    return JSONResponse(content=evaluation)

@app.get("/metrics/")
def compute_metrics(user_id: str = Query(..., description="User ID for metrics computation")):
    # Read evaluations from Azure Blob Storage for a specific user
    blob_name = f"evaluations_{user_id}_{user_info_storage[user_id]['first_name']}_{user_info_storage[user_id]['last_name']}.jsonl"
    blob_client = container_client.get_blob_client(blob_name)

    try:
        download_stream = blob_client.download_blob()
        evaluations = [json.loads(line) for line in download_stream.content_as_text().splitlines()]
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Evaluations not found for user_id {user_id}")

    if not evaluations:
        return JSONResponse(
            content={"error": "No evaluation data found for the given user_id"}, status_code=404
        )

    # Initialize counters and lists
    true_positives_gan = false_positives_gan = true_negatives_gan = (
        false_negatives_gan
    ) = 0
    true_positives_diff = false_positives_diff = true_negatives_diff = (
        false_negatives_diff
    ) = 0
    realism_scores = []
    model_types = []
    categories = []
    durations = []

    # Ground truth categories
    ground_truth = {
        "real_calc": "real",
        "real_normal": "real",
        "synth_gan_calc": "fake",
        "synth_gan_normal": "fake",
        "synth_diff_calc": "fake",
        "synth_diff_normal": "fake",
    }

    # Initialize counters and lists
    true_positives_gan = false_negatives_gan = 0
    true_positives_diff = false_negatives_diff = 0
    true_negatives = false_positives = 0
    realism_scores = []
    model_types = []
    categories = []
    durations = []

    # Process each evaluation
    for eval in evaluations:
        true_category = ground_truth.get(eval["category"], None)
        if true_category is None:
            continue

        predicted_real = eval["is_real"]
        actual_real = true_category == "real"

        model_type = "gan" if "synth_gan" in eval['category'] else "diff"
        category = "Real" if actual_real else "Fake"

        realism_scores.append(eval["realism_score"])
        model_types.append(model_type)
        categories.append(category)

        # Debugging information
        print(f"Processing Evaluation: {eval['category']}")
        print(f"True Category: {true_category}, Predicted Real: {predicted_real}, Actual Real: {actual_real}, Model Type: {model_type}")

        if not actual_real:  # True category is fake
            if model_type == "gan":
                if predicted_real:
                    true_positives_gan += 1
                else:
                    false_negatives_gan += 1
            else:
                if predicted_real:
                    true_positives_diff += 1
                else:
                    false_negatives_diff += 1
        else:  # True category is real
            if predicted_real:
                false_positives += 1
            else:
                true_negatives += 1

        # Append the duration for each image evaluation
        durations.append(eval.get("image_duration", 0))

    # Calculate sensitivity and specificity for GAN and Diff models
    sensitivity_gan = (
        true_positives_gan / (true_positives_gan + false_negatives_gan)
        if (true_positives_gan + false_negatives_gan) > 0
        else 0
    )
    sensitivity_diff = (
        true_positives_diff / (true_positives_diff + false_negatives_diff)
        if (true_positives_diff + false_negatives_diff) > 0
        else 0
    )

    # Since TN and FP are shared, they are not calculated separately for each model
    # Specificity for GAN and Diff models can be calculated using the same TN and FP values
    specificity_gan = (
        true_negatives / (true_negatives + false_positives)
        if (true_negatives + false_positives) > 0
        else 0
    )
    specificity_diff = (
        true_negatives / (true_negatives + false_positives)
        if (true_negatives + false_positives) > 0
        else 0
    )

    # Calculate mean, std, and total duration for image evaluations
    mean_duration = np.mean(durations) if durations else 0
    std_duration = np.std(durations) if durations else 0
    total_duration = sum(durations)  # Total duration by summing up individual durations

    #After creating the initial DataFrame
    
    df = pd.DataFrame(
        {
            "Realism Score": realism_scores,
            "Model Type": model_types,
            "Category": categories,
        }
    )

    # Define new bins for the realism scores
    realism_bins = [0, 20, 40, 60, 80, 100]
    realism_labels = ['Very Low 0-20', 'Low 20-40', 'Medium 40-60', 'High 60-80', 'Very High 80-100']

    # Bin the realism scores into the new levels
    df["Realism Level"] = pd.cut(
        df["Realism Score"], bins=realism_bins, labels=realism_labels, include_lowest=True
    )

    # Create a new column that combines Category and Model Type for fake images
    df['Category_Model'] = df.apply(lambda row: f"{row['Category']}_{row['Model Type']}" if row['Category'] == 'Fake' else row['Category'], axis=1)


    # Define the figure and axes for subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 1. Stacked Bar Plot (using bins)
    # Count occurrences in each bin
    count_df = df.groupby(["Realism Level", "Category_Model"]).size().unstack(fill_value=0)

    # Define color mapping for categories
    color_map = {'Real': 'green', 'Fake_gan': 'red', 'Fake_diff': 'orange'}

    # Plot the count DataFrame with custom colors and stacked bars
    count_df.plot(kind="bar", stacked=True, ax=ax1, color=[color_map.get(cat, 'gray') for cat in count_df.columns])

    # Configure the first subplot (Stacked Bar Plot)
    ax1.set_xlabel("Realism Level")
    ax1.set_ylabel("Count")
    ax1.set_title("Count of Real and Fake Images (GAN/Diffusion) by Realism Level")
    ax1.legend(title="Category and Model Type")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_xticklabels(count_df.index, rotation=45, ha='right')

    # 2. Continuous Distribution Plot (No bins)
    # Plot the distribution of realism scores as a continuous line plot using Seaborn's KDE plot
    sns.kdeplot(df[df['Category_Model'] == 'Real']['Realism Score'], ax=ax2, label='Real', color='green', fill=True)
    sns.kdeplot(df[df['Category_Model'] == 'Fake_gan']['Realism Score'], ax=ax2, label='Fake GAN', color='red', fill=True)
    sns.kdeplot(df[df['Category_Model'] == 'Fake_diff']['Realism Score'], ax=ax2, label='Fake Diffusion', color='orange', fill=True)

    # Configure the second subplot (Distribution Plot)
    ax2.set_xlabel("Realism Score")
    ax2.set_ylabel("Density")
    ax2.set_title("Distribution of Real and Fake Images (GAN/Diffusion) by Realism Score")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(title="Category and Model Type")

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()

    metrics = {
        "sensitivity_gan": round(sensitivity_gan, 2),
        "specificity_gan": round(specificity_gan, 2),
        "sensitivity_diff": round(sensitivity_diff, 2),
        "specificity_diff": round(specificity_diff, 2),
        "total_evaluations": len(evaluations),
        "mean_duration": round(mean_duration, 2),
        "std_duration": round(std_duration, 2),
        "total_duration": round(total_duration, 2),  # Add total duration to metrics
        "realism_plot": plot_data,
    }

    return metrics

# Remove the global variable results_store if it’s not needed elsewhere
results_store = {}

@app.post("/finalize_evaluation/")
async def finalize_evaluation(request: Request):
    try:
        body = await request.json()
        user_id = body.get("user_id")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID is required")
        
        # Compute metrics, which includes the total duration
        metrics = compute_metrics(user_id)
        if metrics is None:
            raise HTTPException(status_code=404, detail="Metrics not found for user_id")
        
        # Store the metrics in results_store
        results_store[user_id] = metrics
        
        # print(f"Metrics computed: {metrics}")  # Log the metrics
        return JSONResponse(content=metrics)
    except Exception as e:
        print(f"Error in finalize_evaluation: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/results/")
async def results_page(user_id: str):
    metrics = results_store.get(user_id)
    if not metrics:
        raise HTTPException(status_code=404, detail=f"No results found for user_id {user_id}.")
    
    # Serve HTML page with results
    return HTMLResponse(content=open("static/results.html").read())

def all_evaluated() -> bool:
    try:
        with open("evaluations.json", "r") as f:
            total_evaluations = sum(1 for _ in f)
    except FileNotFoundError:
        return False

    return total_evaluations >= TOTAL_IMAGES

