import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from difflib import SequenceMatcher

# Define function to initialize the Groq model
def initialize_groq_model():
    return ChatGroq(
        temperature=0,  # Low temperature to minimize creative generation
        model_name="llama-3.1-8b-instant",
        groq_api_key=st.secrets["general"]["GROQ_API_KEY"]
    )

# Simple text splitting function
def simple_text_split(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

@st.cache_data(show_spinner=True)
def scrape_website(urls):
    headers = {'User-Agent': 'GirOrganic/1.0 (+https://www.girorganic.com)'}
    url_contexts = {}
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                text = ' '.join(soup.stripped_strings)
                chunks = simple_text_split(text)
                url_contexts[url] = chunks
            else:
                st.error(f"Failed to retrieve {url}: HTTP {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error scraping {url}: {e}")
    return url_contexts

def find_relevant_chunks(query, contexts, token_limit=6000, prioritized_urls=None):
    prioritized_urls = prioritized_urls or []
    relevant_chunks = []
    total_tokens = 0
    query_token_count = len(query.split()) + 50
    available_tokens = token_limit - query_token_count
    prioritized_chunks_list = []
    other_chunks_list = []

    for url, chunks in contexts.items():
        for chunk in chunks:
            similarity = SequenceMatcher(None, query, chunk).ratio()
            token_count = len(chunk.split())
            # Categorize chunk as prioritized or not
            if url in prioritized_urls:
                prioritized_chunks_list.append((chunk, similarity, token_count, url))
            else:
                other_chunks_list.append((chunk, similarity, token_count, url))

    # Sort by descending similarity
    prioritized_chunks_list.sort(key=lambda x: x[1], reverse=True)
    other_chunks_list.sort(key=lambda x: x[1], reverse=True)

    # Add prioritized chunks first, then others, until token limit is reached
    for chunk, _, token_count, url in prioritized_chunks_list + other_chunks_list:
        if total_tokens + token_count <= available_tokens:
            relevant_chunks.append(chunk)
            total_tokens += token_count
        else:
            break

    return relevant_chunks

def truncate_context_to_token_limit(context, max_tokens):
    words = context.split()
    if len(words) > max_tokens:
        words = words[:max_tokens]
    return " ".join(words)

def main():
    st.markdown("""
    <style>
    .reportview-container {
        background-color: #f7fff7;
        color: #004d40;
    }
    .sidebar .sidebar-content {
        background-color: #f7fff7;
    }
    </style>
    """, unsafe_allow_html=True)

    st.image("transparent.png", width=200)  # Adjust path and size as necessary
    st.title(" GirOrganic Chat Assistant")
    st.write("Welcome! I am GirOrganic AI, how can I assist you today?")

    # Critical URLs to scrape
    urls = [
        "https://girorganic.com/"
        # Add more URLs as needed
    ]

    # Automatic scraping on app load
    if 'contexts' not in st.session_state:
        with st.spinner("Scraping data..."):
            st.session_state['contexts'] = scrape_website(urls)

    user_query = st.text_input("Enter your query here:")
    if user_query and st.button("Answer Query"):
        relevant_chunks = find_relevant_chunks(
            user_query, st.session_state['contexts'], token_limit=6000, prioritized_urls=urls
        )

        if not relevant_chunks:
            fallback_message = "I am sorry, I can't find an answer in our resources. Please contact us for more information."
            st.markdown(f"**Response:** {fallback_message}")
        else:
            context_to_send = "\n\n".join(relevant_chunks)
            context_to_send = truncate_context_to_token_limit(context_to_send, 6000)

            prompt = f"You are GirOrganic AI, Strictly use the infomrmation found on the website. Use the following context to answer the user's query:\n\n{context_to_send}\n\nQuery: {user_query}"
            groq_model = initialize_groq_model()
            response = groq_model.invoke(prompt, timeout=30)
            st.markdown(f"**Response:** {response.content.strip()}")

if __name__ == "__main__":
    main()
