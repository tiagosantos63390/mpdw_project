# Phase 3 - Video Dialog Project

import os
import json
import base64
from io import BytesIO
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, AutoModelForSequenceClassification, AutoTokenizer, pipeline
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

# Initialize intent detection model
classifier = pipeline("zero-shot-classification",
                     model="facebook/bart-large-mnli")

TEXT_TO_IMAGE = 'search for a frame using text description'
IMAGE_TO_IMAGE = 'find similar frames to current image'
IMAGE_AND_TEXT_TO_IMAGE = 'search for frames similar to current image with text modifications'
IMAGE_TO_TEXT = 'describe what is happening in the current frame'

# Define intent labels
intent_labels = [
    TEXT_TO_IMAGE,
    IMAGE_TO_IMAGE,
    IMAGE_AND_TEXT_TO_IMAGE,
    IMAGE_TO_TEXT
]

# Add example queries for each intent to improve detection
intent_examples = {
    TEXT_TO_IMAGE: [
        'show me a frame with',
        'find a frame showing',
        'search for a frame containing',
        'look for a frame with',
        'find a frame where',
        'show me a frame of',
        'find a frame of',
        'search for a frame of',
        'look for a frame of',
        'show me a frame showing'
    ],
    IMAGE_TO_IMAGE: [
        'show me similar frames',
        'find frames like this one',
        'show more frames like the current one',
        'find similar images',
        'show me frames that look like this',
        'find frames similar to this',
        'show me more like this',
        'find more like this'
    ],
    IMAGE_AND_TEXT_TO_IMAGE: [
        'find similar frames but with',
        'show me frames like this but with',
        'find similar frames but in',
        'show me frames like this but in',
        'find similar frames but showing',
        'show me frames like this but showing',
        'find similar frames but',
        'show me frames like this but',
        'find frames like this but with',
        'show me frames like this but with',
        'find similar frames with',
        'show me similar frames with',
        'find frames like this with',
        'show me frames like this with',
        'find similar frames that have',
        'show me similar frames that have',
        'find frames like this that have',
        'show me frames like this that have'
    ],
    IMAGE_TO_TEXT: [
        'what is happening in this frame',
        'describe this frame',
        'what do you see in this image',
        'tell me what is in this frame',
        'what is shown in this picture',
        'what can you see in this frame',
        'describe what you see',
        'what is in this image'
    ]
}

def detect_intent(text):
    text = text.lower()
    
    # First check for explicit keywords
    for intent, examples in intent_examples.items():
        for example in examples:
            if example.lower() in text:
                return intent, 0.9  # High confidence for keyword match
    
    # If no keyword match, try the classifier
    result = classifier(text, intent_labels)
    
    # Only accept classifier results with high confidence
    if result['scores'][0] > 0.7:
        return result['labels'][0], result['scores'][0]
    
    # If confidence is low, try to infer from the text structure
    if any(word in text for word in ['show', 'find', 'search', 'look for']):
        # Check for image+text pattern first
        if ('but' in text or 'with' in text or 'that have' in text) and ('similar' in text or 'like this' in text):
            return IMAGE_AND_TEXT_TO_IMAGE, 0.8
        # Then check for image-to-image pattern
        elif 'similar' in text or 'like this' in text:
            return IMAGE_TO_IMAGE, 0.8
        # Finally, default to text-to-image
        else:
            return TEXT_TO_IMAGE, 0.8
    
    # Default to description if nothing else matches
    return IMAGE_TO_TEXT, 0.5

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
    """Convert an image to CLIP vector"""
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
        embedding = outputs[0] / outputs[0].norm(p=2)
        return embedding.tolist()

def retrieve_frame_by_text(question, video_id):
    """Find frames based on text description (text to image)"""
    try:
        # Process the text query
        inputs = clip_processor(text=question, return_tensors="pt", padding=True)
        with torch.no_grad():
            query_vec = clip_model.get_text_features(**inputs)
            query_vec = (query_vec / query_vec.norm(p=2)).squeeze().tolist()

        # Search for frames with a higher k value to get more potential matches
        query_knn = {
            "size": 5,  # Increased from 1 to get more potential matches
            "_source": ["video_id", "image_path"],
            "query": {
                "bool": {
                    "must": [
                        {"term": {"video_id": video_id}},
                        {"knn": {"image_vec": {"vector": query_vec, "k": 5}}}
                    ]
                }
            }
        }
        
        response = client.search(index=index_name, body=query_knn)
        hits = response["hits"]["hits"]
        
        if not hits:
            print(f"No matches found for query: {question}")
            return None
            
        # Return the best match
        return hits[0]["_source"]["image_path"]
        
    except Exception as e:
        print(f"Error in retrieve_frame_by_text: {str(e)}")
        return None

def search_similar_frames(query_vector, k=5, current_frame_path=None):
    """Search for similar frames using CLIP vector, skipping the current frame if provided"""
    query_knn = {
        "size": k + 1,  # Request one extra result to account for potential skipping
        "_source": ["video_id", "image_path"],
        "query": {
            "knn": {
                "image_vec": {
                    "vector": query_vector,
                    "k": k + 1
                }
            }
        }
    }
    response = client.search(index=index_name, body=query_knn)
    hits = response["hits"]["hits"]
    
    # If current_frame_path is provided, filter out the current frame
    if current_frame_path:
        hits = [hit for hit in hits if hit["_source"]["image_path"] != current_frame_path]
    
    # Return only the requested number of results
    return hits[:k]

def retrieve_frame_by_text_and_image(question, video_id, current_frame_path):
    """Find frames based on text and current image (image and text to image)"""
    # Get text vector
    inputs = clip_processor(text=question, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_vec = clip_model.get_text_features(**inputs)
        text_vec = (text_vec / text_vec.norm(p=2)).squeeze().tolist()
    
    # Get current image vector
    image_vec = encode_frame_to_vector(current_frame_path)
    
    # Combine vectors (simple average)
    combined_vec = [(t + i) / 2 for t, i in zip(text_vec, image_vec)]
    
    query_knn = {
        "size": 1,
        "_source": ["video_id", "image_path"],
        "query": {
            "bool": {
                "must": [
                    {"term": {"video_id": video_id}},
                    {"knn": {"image_vec": {"vector": combined_vec, "k": 1}}}
                ]
            }
        }
    }
    response = client.search(index=index_name, body=query_knn)
    hits = response["hits"]["hits"]
    if not hits:
        return None
    return hits[0]["_source"]["image_path"]

def get_frame_description(frame_path, question):
    """Get description of what's happening in a frame"""
    try:
        with open(frame_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        chat_history = [
            {
                "role": "user",
                "content": question,
                "images": [image_data]
            }
        ]
        
        response = client_ollama.chat(
            model="llava-phi3:latest",
            messages=chat_history
        )
        
        return response["message"]["content"]
    except Exception as e:
        return f"Error generating description: {str(e)}"

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

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    if "last_image_path" not in st.session_state:
        st.session_state.last_image_path = None
if "last_frame_id" not in st.session_state:
    st.session_state.last_frame_id = None
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "submit_triggered" not in st.session_state:
    st.session_state.submit_triggered = False
if "current_frame_index" not in st.session_state:
    st.session_state.current_frame_index = 0

# RIGHT: Selected Image and manual selection
with col2:
    # Select frame manually section   ------------
    st.subheader("Select the frame manually")
    video_names = [v["name"] for v in video_names_and_ids]
    selected_name = st.selectbox("Select a video to explore", video_names)
    selected_video_id = next(v["id"] for v in video_names_and_ids if v["name"] == selected_name)
    frame_folder = f"data/frames/{selected_video_id}"

    # Load frames
    frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith(".jpg")]) if os.path.exists(frame_folder) else []

    if frame_files:
        # Use session state to control the slider
        frame_index = st.slider("Frame index", 0, len(frame_files) - 1, st.session_state.current_frame_index)
        selected_frame = frame_files[frame_index]
        query_path = os.path.join(frame_folder, selected_frame)
        
        # Check if frame was changed manually
        if selected_frame != st.session_state.last_frame_id:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"The frame was changed manually to {selected_frame}"
            })
            st.session_state.last_frame_id = selected_frame

        time_seconds = frame_index * 2
        minutes = time_seconds // 60
        seconds = time_seconds % 60
        st.markdown(f"**Time in video:** {int(minutes)} min {int(seconds)} sec")

        # Frame Display Section
        st.subheader("Selected Frame")
        st.image(query_path, caption=f"Selected frame: {selected_frame}", use_container_width=True)

    else:
        st.warning("No frames found in this folder.")
        
# LEFT: Chatbox
with col1:
    st.subheader("Ask about the current frame")

    # Add custom CSS for chat styling
    st.markdown("""
        <style>
        .user-message {
            background-color:rgb(61, 64, 66);
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            border-left: 5px solid #0066cc;
            color: #f0f0f0;
        }
        .bot-message {
            background-color: rgb(85, 89, 92);
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            border-left: 5px solid #666666;
            color: #f0f0f0;
        }
        .message-container {
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        role = "User" if msg["role"] == "user" else "Bot"
        message_class = "user-message" if role == "User" else "bot-message"
        st.markdown(f"""
            <div class="message-container">
                <div class="{message_class}">
                    <strong>{role}:</strong> {msg['content']}
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Create a form to handle the input and submit button
    with st.form(key="chat_form"):
        user_question = st.text_input(
            "Enter your question:", 
            value=st.session_state.user_input, 
            key="llava_chat_question"
        )
        submit_button = st.form_submit_button("Submit")

    # Handle both button click and Enter key press
    if submit_button and user_question:
        # Clear the input after submission
        st.session_state.user_input = ""
        
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })
        
        # Check if frame has changed
        if selected_frame and selected_frame != st.session_state.last_frame_id:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"I changed the previous frame to the frame {selected_frame}"
            })
            st.session_state.last_frame_id = selected_frame

        # Detect intent
        detected_intent, intent_score = detect_intent(user_question)
        print(f"Detected Intent: {detected_intent} (confidence: {intent_score:.2f})")

        # Handle different intents
        with st.spinner('Processing your request...'):
            if detected_intent == "search for a frame using text description":
                best_frame_path = retrieve_frame_by_text(user_question, selected_video_id)
                if best_frame_path:
                    st.session_state.last_image_path = best_frame_path
                    query_path = best_frame_path
                    selected_frame = os.path.basename(best_frame_path)
                    # Update the frame index in session state
                    st.session_state.current_frame_index = frame_files.index(selected_frame) if selected_frame in frame_files else 0
                    # Add frame change to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"I found a frame matching your description of '{user_question}'."
                    })
                    st.session_state.last_frame_id = selected_frame
                else:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"I couldn't find any frames matching your description of '{user_question}'. Try using more specific or different terms!"
                    })
                    query_path = st.session_state.last_image_path

            elif detected_intent == "find similar frames to current image":
                if query_path:
                    # Get the vector for the current frame
                    current_vector = encode_frame_to_vector(query_path)
                    # Search for similar frames, skipping the current frame
                    similar_hits = search_similar_frames(current_vector, k=1, current_frame_path=query_path)
                    if similar_hits:
                        similar_frame_path = similar_hits[0]["_source"]["image_path"]
                        st.session_state.last_image_path = similar_frame_path
                        query_path = similar_frame_path
                        selected_frame = os.path.basename(similar_frame_path)
                        # Update the frame index in session state
                        st.session_state.current_frame_index = frame_files.index(selected_frame) if selected_frame in frame_files else 0
                        # Add frame change to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"I changed the previous frame to the frame {selected_frame}"
                        })
                        st.session_state.last_frame_id = selected_frame
                        # Add bot response
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": "I found a similar frame to the current one."
                        })

            elif detected_intent == "search for frames similar to current image with text modifications":
                if query_path:
                    # Combine current image and text query
                    best_frame_path = retrieve_frame_by_text_and_image(user_question, selected_video_id, query_path)
                    if best_frame_path:
                        st.session_state.last_image_path = best_frame_path
                        query_path = best_frame_path
                        selected_frame = os.path.basename(best_frame_path)
                        # Update the frame index in session state
                        st.session_state.current_frame_index = frame_files.index(selected_frame) if selected_frame in frame_files else 0
                        # Add frame change to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"I changed the previous frame to the frame {selected_frame}"
                        })
                        st.session_state.last_frame_id = selected_frame
                        # Add bot response
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": "I found a frame based on your text modifications."
                        })

            elif detected_intent == "describe what is happening in the current frame":
                if query_path:
                    with open(query_path, "rb") as img_file:
                        image_data = base64.b64encode(img_file.read()).decode('utf-8')

                    try:
                        # Create a temporary chat history for LLaVA
                        llava_chat = [
                            {
                                "role": "user",
                                "content": "Please describe what you see in this image in a natural, conversational way. Do not return any JSON or technical data.",
                                "images": [image_data]
                            }
                        ]
                        
                        response = client_ollama.chat(
                            model="llava-phi3:latest",
                            messages=llava_chat
                        )
                        llava_reply = response["message"]["content"]
                        
                        # Clean up the response if it contains JSON
                        if "{" in llava_reply and "}" in llava_reply:
                            try:
                                # Try to extract just the description if it's in JSON format
                                json_data = json.loads(llava_reply)
                                if isinstance(json_data, list) and len(json_data) > 0:
                                    if "description" in json_data[0]:
                                        llava_reply = json_data[0]["description"]
                            except:
                                # If JSON parsing fails, use a default message
                                llava_reply = "I can see the image but I'm having trouble describing it properly."
                        
                        # Add bot response to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": llava_reply
                        })
                    except Exception as e:
                        # Add error message to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"Error communicating with LLaVA: {str(e)}"
                        })
                else:
                    # Add error message if no query path is available
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "I'm sorry, but I don't have access to the current frame to describe it."
                    })
            
            # Force a rerun to update the chat display
            st.rerun()