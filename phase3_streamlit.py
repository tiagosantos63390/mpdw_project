# Phase 3 - Video Dialog Project

import os
import json
import base64
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForQuestionAnswering
from transformers import CLIPTextModel, CLIPVisionModel, CLIPFeatureExtractor, CLIPModel, CLIPProcessor
from opensearchpy import OpenSearch
import streamlit as st
from ollama import Client
from datetime import datetime
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Set page config first, before any other Streamlit commands
st.set_page_config(page_title="Conversational agent", layout="wide")

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

intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

# Define intent labels with more specific actions
TEXT_TO_IMAGE = 'find a new frame based on text description'
IMAGE_TO_IMAGE = 'find a frame similar to the current one'
IMAGE_AND_TEXT_TO_IMAGE = 'find a frame similar to current but with modifications'
IMAGE_TO_TEXT = 'describe what is in the current frame'
GENERAL_CONVERSATION = 'engage in general conversation'  # New intent

intent_labels = [TEXT_TO_IMAGE, IMAGE_TO_IMAGE, IMAGE_AND_TEXT_TO_IMAGE, IMAGE_TO_TEXT, GENERAL_CONVERSATION]

# Define examples for each intent with diverse language patterns
intent_examples = {
    TEXT_TO_IMAGE: [
        "show me a frame with",
        "find a frame showing",
        "search for a frame containing",
        "look for a frame with",
        "find a frame where",
        "show me a frame of",
        "find a frame of",
        "search for a frame of",
        "look for a frame of",
        "show me a frame showing",
        "get me a frame with",
        "i want to see a frame with",
        "can you find a frame with",
        "i need a frame showing",
        "find me a frame with"
    ],
    IMAGE_TO_IMAGE: [
        "show me similar frames",
        "find frames like this one",
        "show more frames like the current one",
        "find similar images",
        "show me frames that look like this",
        "find frames similar to this",
        "show me more like this",
        "find more like this",
        "get me another frame like this",
        "show me a different frame like this"
    ],
    IMAGE_AND_TEXT_TO_IMAGE: [
        "find similar frames but with",
        "show me frames like this but with",
        "find similar frames but in",
        "show me frames like this but in",
        "find similar frames but showing",
        "show me frames like this but showing",
        "find a frame like this that shows",
        "show me a similar frame with different",
        "find a frame like this in a different",
        "show me a similar frame from a different"
    ],
    IMAGE_TO_TEXT: [
        "what is happening in this frame",
        "describe this frame",
        "what do you see in this image",
        "tell me what is in this frame",
        "what is shown in this picture",
        "what can you see in this frame",
        "describe what you see",
        "what is in this image",
        "tell me about this frame",
        "explain what's in this frame"
    ],
    GENERAL_CONVERSATION: [
        "hello",
        "hi",
        "hey",
        "how are you",
        "what's up",
        "good morning",
        "good afternoon",
        "good evening",
        "how's it going",
        "nice to meet you",
        "thanks",
        "thank you",
        "bye",
        "goodbye",
        "see you later",
        "have a nice day",
        "what can you do",
        "help me",
        "what are your capabilities",
        "tell me about yourself"
    ]
}

# Initialize session state for logs
if "intent_logs" not in st.session_state:
    st.session_state.intent_logs = []

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text):
    """Clean and normalize text input"""
    # Convert to lowercase and remove extra whitespace
    text = text.lower().strip()
    
    # Remove special characters but keep important ones
    text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords but keep important ones
    stop_words = set(stopwords.words('english'))
    important_words = {'show', 'find', 'search', 'like', 'similar', 'frame', 'this', 'current', 'but', 'with', 'in', 'on', 'at'}
    stop_words = stop_words - important_words
    
    tokens = [t for t in tokens if t not in stop_words]
    
    # Join back into text
    return ' '.join(tokens)

# Define intent-specific templates
intent_templates = {
    TEXT_TO_IMAGE: [
        "The user wants to find a new frame that shows {}",
        "The user is looking for a frame containing {}",
        "The user wants to see a frame with {}",
        "The user is searching for a frame showing {}",
        "The user needs a frame that contains {}"
    ],
    IMAGE_TO_IMAGE: [
        "The user wants to find a frame similar to the current one",
        "The user is looking for frames like the current frame",
        "The user wants to see more frames like this",
        "The user needs similar frames to the current one",
        "The user is searching for frames that look like this"
    ],
    IMAGE_AND_TEXT_TO_IMAGE: [
        "The user wants to find a frame like this but with {}",
        "The user is looking for a similar frame with {}",
        "The user wants to see a frame like this but {}",
        "The user needs a similar frame that {}",
        "The user is searching for a frame like this with {}"
    ],
    IMAGE_TO_TEXT: [
        "The user wants to know what is in this frame",
        "The user is asking for a description of this frame",
        "The user wants to understand what's happening in this frame",
        "The user needs a description of the current frame",
        "The user is asking about the content of this frame"
    ],
    GENERAL_CONVERSATION: [
        "The user is engaging in general conversation",
        "The user is making small talk",
        "The user is greeting or saying goodbye",
        "The user is asking for help or information",
        "The user is making a general inquiry"
    ]
}

def log_intent_detection(text, intent, confidence, all_scores, slots=None):
    """Log intent detection results with slots if applicable"""
    log_entry = {
        "user_text": text,
        "detected_intent": intent,
        "confidence": confidence,
        "all_scores": dict(zip(all_scores['labels'], all_scores['scores']))
    }
    
    # Add slots if they exist
    if slots:
        log_entry["slots"] = slots
    
    st.session_state.intent_logs.append(log_entry)

def detect_intent(text):
    """Enhanced intent detection with preprocessing and better matching"""
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # First try pattern matching with improved patterns
    for intent, patterns in intent_examples.items():
        for pattern in patterns:
            # Use regex for more flexible matching
            pattern_regex = re.compile(r'\b' + re.escape(pattern.lower()) + r'\b')
            if pattern_regex.search(processed_text):
                # Additional context checks
                if intent == IMAGE_TO_IMAGE:
                    if not any(word in processed_text for word in ['similar', 'like', 'this', 'current']):
                        continue
                elif intent == TEXT_TO_IMAGE:
                    # Ensure there's content after the pattern
                    if not re.search(r'\b' + re.escape(pattern.lower()) + r'\s+\w+', processed_text):
                        continue
                return intent, 0.95
    
    # Try intent-specific templates
    best_result = None
    best_score = 0
    
    for intent, templates in intent_templates.items():
        for template in templates:
            # For templates that need slot filling, try to extract the slot
            if '{}' in template:
                # Extract potential slot content
                slot_match = re.search(r'\b(?:with|containing|showing|that|but)\s+([^.,!?]+)', processed_text)
                if slot_match:
                    filled_template = template.format(slot_match.group(1))
                else:
            continue
            else:
                filled_template = template
            
            result = intent_classifier(
                processed_text,
                [intent],  # Try one intent at a time
                hypothesis_template=filled_template,
                multi_label=False
            )
            
            if result['scores'][0] > best_score:
                best_result = result
                best_score = result['scores'][0]
    
    # Get the best matching intent and its score
    if best_result and best_score >= 0.6:
        return best_result['labels'][0], best_score
    
    # Fallback to keyword-based detection with lower confidence
    for intent, patterns in intent_examples.items():
        if any(pattern.lower() in processed_text for pattern in patterns):
            return intent, 0.7
    
    return None, 0.0

def extract_slots(text, intent):
    """Enhanced slot extraction with validation and type checking"""
    processed_text = preprocess_text(text)
    
    if intent == IMAGE_AND_TEXT_TO_IMAGE:
        # Extract modifications with better prompts
        prompts = [
            "What are the modifications requested?",
            "What changes does the user want to make?",
            "What is different about what the user wants?"
        ]
        answers = []
        for prompt in prompts:
            res = qa_pipeline(question=prompt, context=processed_text)
            if res['answer'].strip():
                answers.append(res['answer'].strip())
        
        # Use the most specific answer
        if answers:
            # Sort by length (longer answers tend to be more specific)
            best_answer = max(answers, key=len)
            return {"modifications": best_answer}
            
    elif intent == TEXT_TO_IMAGE:
        # Extract search term with better prompts
        prompts = [
            "What is being searched for?",
            "What does the user want to find?",
            "What content is the user looking for?"
        ]
        answers = []
        for prompt in prompts:
            res = qa_pipeline(question=prompt, context=processed_text)
            if res['answer'].strip():
                answers.append(res['answer'].strip())
        
        # Use the most specific answer
        if answers:
            best_answer = max(answers, key=len)
            return {"search_term": best_answer}
    
    return {}

def encode_frame_to_vector(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
        return (outputs[0] / outputs[0].norm(p=2)).tolist()

def retrieve_frame_by_text(question, video_id):
    inputs = clip_processor(text=question, return_tensors="pt", padding=True)
    with torch.no_grad():
        query_vec = clip_model.get_text_features(**inputs)
        query_vec = (query_vec / query_vec.norm(p=2)).squeeze().tolist()

    query_knn = {
        "size": 5,
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
    response = client.search(index='user04', body=query_knn)
    hits = response["hits"]["hits"]
    return hits[0]["_source"]["image_path"] if hits else None

def get_frame_description(frame_path, question):
    try:
        with open(frame_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        chat_history = [{"role": "user", "content": question, "images": [image_data]}]
        response = client_ollama.chat(model="llava-phi3:latest", messages=chat_history)
        return response["message"]["content"]
    except Exception as e:
        return f"Error generating description: {str(e)}"

def search_similar_frames(query_vector, k=2, current_frame_path=None):
    """Search for similar frames using KNN vector search, returning the second most similar frame"""
    query_knn = {
        "size": k,  # Get top 2 to ensure we can get the second most similar
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
    
    response = client.search(index='user04', body=query_knn)
    hits = response["hits"]["hits"]
    
    # Return the second hit (index 1) if available, otherwise None
    return hits[1]["_source"]["image_path"] if len(hits) > 1 else None

def retrieve_frame_by_text_and_image(text, video_id, current_frame_path):
    """Find frames similar to current image but with text modifications"""
    # Get current frame vector
    current_vector = encode_frame_to_vector(current_frame_path)
    
    # Get text vector
    inputs = clip_processor(text=text, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_vec = clip_model.get_text_features(**inputs)
        text_vec = (text_vec / text_vec.norm(p=2)).squeeze().tolist()
    
    # Combine search with both vectors
    query_knn = {
        "size": 5,
        "_source": ["video_id", "image_path"],
        "query": {
            "bool": {
                "must": [
                    {"term": {"video_id": video_id}},
                    {"knn": {"image_vec": {"vector": current_vector, "k": 5}}},
                    {"knn": {"image_vec": {"vector": text_vec, "k": 5}}}
                ],
                "must_not": [
                    {"term": {"image_path": current_frame_path}}
                ]
            }
        }
    }
    
    response = client.search(index='user04', body=query_knn)
    hits = response["hits"]["hits"]
    return hits[0]["_source"]["image_path"] if hits else None

def qa_pipeline(question, context):
    """Extract information from text using question answering"""
    inputs = qa_tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = qa_model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits)
        tokens = qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        answer = qa_tokenizer.convert_tokens_to_string(tokens[answer_start:answer_end + 1])
        return {"answer": answer}

def get_general_response(text):
    """Handle general conversation using LLaVA"""
    try:
        chat_history = [{"role": "user", "content": text}]
        response = client_ollama.chat(model="llava-phi3:latest", messages=chat_history)
        return response["message"]["content"]
    except Exception as e:
        return f"I'm having trouble responding right now: {str(e)}"

# Add collapsible sidebar for logs
with st.sidebar:
    st.title("Intent Detection Logs")
    
    # Add a button to clear logs
    if st.button("Clear Logs"):
        st.session_state.intent_logs = []
    
    # Show logs in reverse chronological order
    for log in reversed(st.session_state.intent_logs):
        with st.expander(f"Query: {log['user_text'][:50]}..."):
            st.write("**Chosen Intent:**", log['detected_intent'] if log['detected_intent'] else "None (confidence < 0.6)")
            st.write("**Confidence Scores:**")
            for intent, score in sorted(log['all_scores'].items()):
                st.write(f"- {intent}: {score:.2f}")
            if "slots" in log and log["slots"]:
                st.write("**Extracted Slots:**")
                for slot_name, slot_value in log["slots"].items():
                    st.write(f"- {slot_name}: {slot_value}")

# Main content
st.title("Dialog manager with controlled dialog state-tracking")

col1, col2 = st.columns([1, 1])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_frame_id" not in st.session_state:
    st.session_state.last_frame_id = None
if "current_frame_index" not in st.session_state:
    st.session_state.current_frame_index = 0

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

with col2:
    st.subheader("Select the frame manually")
    video_names = [v["name"] for v in video_names_and_ids]
    
    # Track video changes
    if "last_video_id" not in st.session_state:
        st.session_state.last_video_id = None
    
    selected_name = st.selectbox("Select a video to explore", video_names)
    selected_video_id = next(v["id"] for v in video_names_and_ids if v["name"] == selected_name)
    
    # Notify if video changed
    if selected_video_id != st.session_state.last_video_id:
        if st.session_state.last_video_id is not None:  # Don't notify on first load
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Switched to video {selected_name}"
            })
        st.session_state.last_video_id = selected_video_id
        # Reset frame index to first frame when video changes
        st.session_state.current_frame_index = 0
    
    frame_folder = f"data/frames/{selected_video_id}"
    frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith(".jpg")]) if os.path.exists(frame_folder) else []

    if frame_files:
        frame_index = st.slider("Frame index", 0, len(frame_files) - 1, st.session_state.current_frame_index)
        selected_frame = frame_files[frame_index]
        query_path = os.path.join(frame_folder, selected_frame)
        
        if selected_frame != st.session_state.last_frame_id:
            if st.session_state.last_frame_id is not None:  # no notify on first load
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Changed to frame {selected_frame}"
                })
            st.session_state.last_frame_id = selected_frame
            st.session_state.current_frame_index = frame_index
        
        st.subheader("Selected Frame")
        st.image(query_path, caption=f"Selected frame: {selected_frame}", use_container_width=True)
        
with col1:
    st.subheader("Ask about the current frame")

    for msg in st.session_state.chat_history:
        role = "User" if msg["role"] == "user" else "Bot"
        content = msg["content"]
        if role == "User" and "intent_info" in msg:
            content += msg["intent_info"]
        
        st.markdown(f"""
            <div style="background-color: {'rgb(61, 64, 66)' if role == 'User' else 'rgb(85, 89, 92)'};
                        padding: 10px; border-radius: 10px; margin: 5px 0;
                        border-left: 5px solid {'#0066cc' if role == 'User' else '#666666'};
                        color: #f0f0f0;">
                <strong>{role}:</strong> {content}
            </div>
        """, unsafe_allow_html=True)
    
    with st.form(key="chat_form"):
        user_question = st.text_input("Enter your question:", key="user_question")
        submit_button = st.form_submit_button("Submit")
    
    if submit_button and user_question:
        # Detect intent and get slots
        intent, confidence = detect_intent(user_question)
        slots = extract_slots(user_question, intent) if intent in [TEXT_TO_IMAGE, IMAGE_AND_TEXT_TO_IMAGE] else None
        
        # Log intent detection
        log_intent_detection(user_question, intent if confidence >= 0.6 else None, confidence, 
                           intent_classifier(user_question, intent_labels, 
                                          hypothesis_template="The user is explicitly requesting to {}",
                                          multi_label=False), 
                           slots=slots)
        
        # Add user message with intent information
        intent_info = f" (Intent: {intent}, Confidence: {confidence:.2f})" if intent else " (No intent detected)"
                st.session_state.chat_history.append({
            "role": "user",
            "content": user_question,
            "intent_info": intent_info
        })
        
        with st.spinner('Processing...'):
            if intent == GENERAL_CONVERSATION:
                response = get_general_response(user_question)
            elif intent is None:
                # Try general conversation as fallback
                if any(greeting in user_question.lower() for greeting in intent_examples[GENERAL_CONVERSATION]):
                    response = get_general_response(user_question)
                else:
                    response = "I'm not sure what you want to do. Could you please clarify? For example:\n" + \
                              "- To find a new frame, try: 'show me a frame with...'\n" + \
                              "- To find similar frames, try: 'find frames like this'\n" + \
                              "- To modify a search, try: 'find frames like this but...'\n" + \
                              "- To get a description, try: 'what's happening in this frame?'\n" + \
                              "- Or just say hello for a chat!"
            elif intent == TEXT_TO_IMAGE:
                search_term = slots.get("search_term", user_question)
                frame_path = retrieve_frame_by_text(search_term, selected_video_id)
                if frame_path:
                    selected_frame = os.path.basename(frame_path)
                    st.session_state.current_frame_index = frame_files.index(selected_frame)
                st.session_state.last_frame_id = selected_frame
                    response = f"I found a frame showing: {search_term}"
            else:
                    response = f"I couldn't find any frames showing: {search_term}"
            
            elif intent == IMAGE_TO_IMAGE and frame_files:
                current_vector = encode_frame_to_vector(query_path)
                similar_frame_path = search_similar_frames(current_vector)
                if similar_frame_path:
                    selected_frame = os.path.basename(similar_frame_path)
                    st.session_state.current_frame_index = frame_files.index(selected_frame)
                    st.session_state.last_frame_id = selected_frame
                    response = f"I found a similar frame: {selected_frame}"
                else:
                    response = "I couldn't find any similar frames."
            
            elif intent == IMAGE_AND_TEXT_TO_IMAGE and frame_files:
                modifications = slots.get("modifications", "")
                frame_path = retrieve_frame_by_text_and_image(modifications, selected_video_id, query_path)
                if frame_path:
                    selected_frame = os.path.basename(frame_path)
                    st.session_state.current_frame_index = frame_files.index(selected_frame)
                    st.session_state.last_frame_id = selected_frame
                    response = f"I found a frame with your requested modifications: {modifications}"
                else:
                    response = f"I couldn't find a frame with the requested modifications: {modifications}"
            
            elif intent == IMAGE_TO_TEXT and frame_files:
                response = get_frame_description(query_path, user_question)
            
            # Add bot response
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                "content": response
                    })
            st.rerun()