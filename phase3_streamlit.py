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
import base64
from ollama import Client

device = "cuda" if torch.cuda.is_available() else "cpu"
client_ollama = Client(host='https://twiz.novasearch.org/ollama')

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

def retrieve_frame_by_text(question, video_id):
    inputs = clip_processor(text=question, return_tensors="pt", padding=True)
    with torch.no_grad():
        query_vec = clip_model.get_text_features(**inputs)
        query_vec = (query_vec / query_vec.norm(p=2)).squeeze().tolist()

    query_knn = {
        "size": 1,
        "_source": ["video_id", "image_path"],
        "query": {
            "bool": {
                "must": [
                    {"term": {"video_id": video_id}},
                    {"knn": {"image_vec": {"vector": query_vec, "k": 1}}}
                ]
            }
        }
    }
    response = client.search(index=index_name, body=query_knn)
    hits = response["hits"]["hits"]
    if not hits:
        return None
    return hits[0]["_source"]["image_path"]

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


# ------ STREAMLIT UI LAYOUT ------

st.set_page_config(page_title="Conversational agent", layout="wide")
st.title("Dialog manager with controlled dialog state-tracking")

col1, col2 = st.columns([1, 1])

# Initialize variables
query_path = None
frame_index = 0
selected_frame = None

# RIGHT: Configuration + Image + Frame Navigation
with col2:
    # Configuration Section (moved from sidebar)
    st.subheader("Select the frame manually")
    video_names = [v["name"] for v in video_names_and_ids]
    selected_name = st.selectbox("Select a video to explore", video_names)
    selected_video_id = next(v["id"] for v in video_names_and_ids if v["name"] == selected_name)
    frame_folder = f"data/frames/{selected_video_id}"

    # Load frames
    frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith(".jpg")]) if os.path.exists(frame_folder) else []

    if frame_files:
        # Frame Index Slider (now above image)
        frame_index = st.slider("Frame index", 0, len(frame_files) - 1, 0)
        selected_frame = frame_files[frame_index]
        query_path = os.path.join(frame_folder, selected_frame)
        
        time_seconds = frame_index * 2
        minutes = time_seconds // 60
        seconds = time_seconds % 60
        st.markdown(f"**Time in video:** {int(minutes)} min {int(seconds)} sec")

        # Frame Display Section
        st.subheader("Selected Frame")
        st.image(query_path, caption=f"Selected frame: {selected_frame}", use_container_width=True)
        
    else:
        st.warning("No frames found in this folder.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    if "last_image_path" not in st.session_state:
        st.session_state.last_image_path = None

# LEFT: Chatbox
with col1:
    st.subheader("Ask about the current frame")

    for msg in st.session_state.chat_history:
        role = "user" if msg["role"] == "user" else "bot"
        st.markdown(f"**{role}**: {msg['content']}")

    user_question = st.text_input("Enter your question:", key="llava_chat_question")

    if st.button("Submit") and user_question:
        search_keywords = ["show", "give me", "find", "search", "frame of", "what video", "who is", "where is"]
        lower_q = user_question.lower()

        if any(keyword in lower_q for keyword in search_keywords):
            best_frame_path = retrieve_frame_by_text(user_question, selected_video_id)
            if best_frame_path:
                st.session_state.last_image_path = best_frame_path
                query_path = best_frame_path
                selected_frame = os.path.basename(best_frame_path)
                frame_index = frame_files.index(selected_frame) if selected_frame in frame_files else 0
            else:
                st.warning("No frame matched your query. Try rephrasing!")
                query_path = st.session_state.last_image_path
        else:
            query_path = st.session_state.last_image_path
            if query_path is None:
                st.error("No image selected! Ask something like 'show me a man running' first.")
            else:
                with open(query_path, "rb") as img_file:
                    image_data = base64.b64encode(img_file.read()).decode('utf-8')

                st.session_state.chat_history.append({
                    "role": "user", "content": user_question, "images": [image_data]
                })

                try:
                    response = client_ollama.chat(
                        model="llava-phi3:latest",
                        messages=st.session_state.chat_history
                    )
                    llava_reply = response["message"]["content"]
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": llava_reply
                    })
                    st.success(f"LLaVA says: {llava_reply}")
                except Exception as e:
                    st.error(f"Error communicating with LLaVA: {str(e)}")