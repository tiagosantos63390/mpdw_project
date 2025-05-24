# Phase 3 - Video Dialog Project

import os
import json
import base64
from io import BytesIO
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from opensearchpy import OpenSearch
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np
import streamlit as st

device = "cuda" if torch.cuda.is_available() else "cpu"

host = 'api.novasearch.org'
port = 443
user = 'user04'
password = 'no.LIMITS2100'
index_name = user

client = OpenSearch(
    hosts=[{'host': host, 'port': port}],
    http_compress=True,
    http_auth=(user, password),
    use_ssl=True,
    url_prefix='opensearch_v2',
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False
)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def compute_clip_embeddings(frame_folder):
    embeddings = {}
    for filename in sorted(os.listdir(frame_folder)):
        if not filename.endswith(".jpg"):
            continue
        image_path = os.path.join(frame_folder, filename)
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = clip_model.get_image_features(**inputs)
            embedding = outputs[0] / outputs[0].norm(p=2)
            embeddings[filename] = embedding.tolist()
    print(f"Computed CLIP embeddings for {len(embeddings)} frames in {frame_folder}")
    return embeddings

def encode_frame_to_vector(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
        embedding = outputs[0] / outputs[0].norm(p=2)
        return embedding.tolist()

def search_similar_frames(query_vector, k=5):
    query_knn = {
        "size": k,
        "_source": ["video_id", "image_path"],
        "query": {
            "knn": {
                "image_vec": {
                    "vector": query_vector,
                    "k": k
                }
            }
        }
    }
    response = client.search(index=index_name, body=query_knn)
    return response["hits"]["hits"]

def display_similar_frames(query_frame_path, k=5):
    query_vec = encode_frame_to_vector(query_frame_path)
    results = search_similar_frames(query_vec, k=k)
    result_paths = []
    for hit in results:
        img_path = hit["_source"]["image_path"].replace("\\", "/")
        result_paths.append(img_path)
    return result_paths



# ------ STREAMLIT ------

st.set_page_config(page_title="Visual Similarity Navigation", layout="wide")

st.title("Visual Similarity Navigation using CLIP and OpenSearch")

st.sidebar.header("Configuration")

# Video selection
video_names_and_ids = [
    {'name': "How to Cook Mashed Potatoes", 'id': "v_-rKS00dzFxQ"},
    {'name': "London 2012 Olympics", 'id': "v_-fjUWhSM6Hc"},
    {'name': "20 Exercises on Parallel Bars", 'id': "v_v7o9uSu9AVI"},
    {'name': "Vin Diesel Breakdancing Throwback", 'id': "v_RJpWgi0EaUE"},
    {'name': "Twickenham Festival 2015 Tug of War", 'id': "v_G7kqlq8WhRo"},
    {'name': "Washing my Face", 'id': "v_jTMdMnbW9OI"},
    {'name': "Girl in Balance Beam (gymnastics)", 'id': "v_9wtMJoqGTg0"},
    {'name': "Epic Rollerblading Montage 80s", 'id': "v_Ffi7vDa3C2I"},
    {'name': "'What U think about Rollerblading?'", 'id': "v_JRr3BruqS2Y"},
    {'name': "Preparing Angel Hair Pasta", 'id': "v_Mkljhl3D9-Q"}
]

video_names = [v["name"] for v in video_names_and_ids]
selected_name = st.sidebar.selectbox("Select a video to explore", video_names)
selected_video_id = next(v["id"] for v in video_names_and_ids if v["name"] == selected_name)
frame_folder = f"data/frames/{selected_video_id}"

# Number of results
k = st.sidebar.slider("Number of similar frames to display", 1, 10, 5)

# Load frames
frame_files = []
if os.path.exists(frame_folder):
    frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith(".jpg")])

if frame_files:
    st.subheader("Frame Selection")

    # Slider for selecting frame index
    frame_index = st.slider("Select position in video (frame index)", 0, len(frame_files)-1, 0)
    selected_frame = frame_files[frame_index]
    query_path = os.path.join(frame_folder, selected_frame)
    
    time_seconds = frame_index * 2
    minutes = time_seconds // 60
    seconds = time_seconds % 60
    st.markdown(f"**Time in video:** {int(minutes)} min {int(seconds)} sec")

    st.image(query_path, caption=f"Selected keyframe: {selected_frame}", use_container_width=True)

    if st.button("Find Similar Frames"):
        similar = display_similar_frames(query_path, k=k)
        st.subheader("Similar Frames")
        cols = st.columns(min(k, 5))
        for i, path in enumerate(similar):
            with cols[i % len(cols)]:
                st.image(path, caption=(os.path.basename(path)), use_container_width=True)
else:
    st.warning("No frames found for this video.")

# ------------------------------------------------------------
