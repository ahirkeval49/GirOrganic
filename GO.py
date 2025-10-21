# tartans.py
# CMU College of Engineering Program Navigator
# Author: Gemini (Google AI)
# Date: October 21, 2025 (Refactored for Keyword RAG - Version 5.0)

import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin
import google.generativeai as genai
from typing import Optional, Dict, Any, List
from difflib import SequenceMatcher # New import for keyword matching

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CMU Engineering Program Navigator",
    page_icon="🎓",
    layout="wide"
)

# --- UTILITIES (FROM GIRORGANIC APP) ---

def simple_text_split(text, chunk_size=750):
    """Splits text into chunks of a given size."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def find_relevant_chunks(query, contexts, token_limit=3800):
    """
    Finds relevant chunks using SequenceMatcher (keyword matching).
    'contexts' is a dict where key=program_name, value=list of text chunks.
    """
    relevant_chunks = []
    total_tokens = 0
    query_token_count = len(query.split()) + 50 # Reserve tokens for query/prompt
    available_tokens = token_limit - query_token_count
    chunks_list = []

    for program_name, chunks in contexts.items():
        for chunk in chunks:
            similarity = SequenceMatcher(None, query, chunk).ratio()
            # Only consider chunks with some similarity
            if similarity > 0.1: 
                token_count = len(chunk.split())
                chunks_list.append((chunk, similarity, token_count, program_name))

    # Sort by descending similarity
    chunks_list.sort(key=lambda x: x[1], reverse=True)

    # Add chunks until token limit is reached
    for chunk, _, token_count, _ in chunks_list:
        if total_tokens + token_count <= available_tokens:
            relevant_chunks.append(chunk)
            total_tokens += token_count
        else:
            break

    return relevant_chunks

def truncate_context_to_token_limit(context, max_tokens=3800):
    """Ensures the final context string fits within the token limit."""
    words = context.split()
    if len(words) > max_tokens:
        words = words[:max_tokens]
    return " ".join(words)


# --- DATA SCRAPING & PROCESSING ---

def robust_request(url: str, headers: Dict[str, str], timeout: int = 15) -> Optional[str]:
    """Handles HTTP requests with basic error checking."""
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        return None

@st.cache_data(ttl=86400) # Cache data for 24 hours
def get_cmu_program_data():
    """Scrapes the CMU program data and returns a DataFrame."""
    department_urls = [
        "https://engineering.cmu.edu/education/graduate-studies/programs/bme.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/cheme.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/cee.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/ece.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/epp.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/ini.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/iii.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/mse.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/meche.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/cmu-africa.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/sv.html"
    ]
    
    programs_list = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    st.subheader("Data Fetch Status (Cached)")
    status_expander = st.expander("Show Data Scraping Log", expanded=False)

    with status_expander:
        progress_bar = st.progress(0, text="Starting direct program content fetch...")
        for i, page_url in enumerate(department_urls):
            page_name = page_url.split('/')[-1]
            progress_bar.progress((i + 1) / len(department_urls), text=f"Processing: {page_name}")
            
            html_content = robust_request(page_url, headers)
            if not html_content:
                st.warning(f"Skipping page: {page_name} (Failed to retrieve).")
                continue
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            program_name_base = "CMU Program"
            if soup.find('h1'):
                program_name_base = soup.find('h1').get_text(strip=True).replace("Graduate", "").strip()
            elif soup.find('title'):
                program_name_base = soup.find('title').get_text(strip=True).split('|')[0].strip()
                
            description_text = 'No sufficiently detailed description found.'
            description_paragraphs = soup.body.find_all('p', limit=10) if soup.body else []
            text_parts = [p.get_text(strip=True) for p in description_paragraphs if len(p.get_text(strip=True)) > 50]
            
            if text_parts:
                description_text = ' '.join(text_parts)
                description_text = description_text[:700].rstrip() + '...' if len(description_text) > 700 else description_text
            
            programs_list.append({
                'name': f"Master of Science ({program_name_base})", 
                'url': page_url, 'description': description_text, 'degree_type': 'M.S.'
            })
            programs_list.append({
                'name': f"Doctor of Philosophy ({program_name_base})", 
                'url': page_url, 'description': description_text, 'degree_type': 'Ph.D.'
            })
            
            st.info(f"Indexed M.S. and Ph.D. from: {program_name_base}")
            time.sleep(0.1)

        progress_bar.empty()
        
    if not programs_list:
        st.error("Scraping finished, but no program data was collected.")
        return pd.DataFrame()
        
    st.success(f"Scraped and indexed {len(programs_list)} program variants.")
    return pd.DataFrame(programs_list)

@st.cache_data
def build_program_contexts(df_programs: pd.DataFrame) -> Dict[str, List[str]]:
    """Converts the program DataFrame into a dictionary of text chunks for RAG."""
    contexts = {}
    for _, row in df_programs.iterrows():
        # Create a single text block for each program
        full_text = f"Program Name: {row['name']}\nDegree: {row['degree_type']}\nURL: {row['url']}\nDescription: {row['description']}"
        
        # Split that text block into chunks
        contexts[row['name']] = simple_text_split(full_text, chunk_size=750)
    return contexts

# --- AI PROVIDER (CHAT-ONLY) FUNCTIONS ---

def _call_analysis_api(api_key: str, endpoint: str, payload: Dict[str, Any], model: str) -> str:
    """Generic function to call a chat completions API."""
    for attempt in range(3):
        try:
            response = requests.post(endpoint, headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=45)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            return f"AI analysis unavailable: {model} API call failed: {e}"
        except Exception as e:
            return f"AI analysis unavailable: {model} API processing failed: {e}"
    return "AI analysis unavailable: Unknown error."

@st.cache_data
def get_deepseek_rag_analysis(api_key: str, prompt: str) -> str:
    """Gets a RAG analysis from DeepSeek using a provided prompt."""
    if not api_key: return "AI analysis unavailable: DeepSeek API key missing."
    
    payload = {
        "model": "deepseek-chat", 
        "messages": [{"role": "user", "content": prompt}], 
        "temperature": 0.3, # Lower temp for factual answers
        "max_tokens": 1024
    }
    endpoint = "https://api.deepseek.com/v1/chat/completions"
    return _call_analysis_api(api_key, endpoint, payload, "DeepSeek")

@st.cache_data
def get_gemini_rag_analysis(api_key: str, prompt: str) -> str:
    """Gets a RAG analysis from Gemini using a provided prompt."""
    if not api_key: return "AI analysis unavailable: Gemini API key missing."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: 
        return f"AI analysis failed to generate (Gemini): {e}"

# --- MAIN APPLICATION LOGIC ---
def main():
    st.title("🎓 CMU Engineering Program Navigator")
    st.markdown("Answer a few questions to discover the graduate program that best fits your academic and career goals.")

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.image("https://www.cmu.edu/brand/brand-guidelines/assets/images/wordmarks-and-initials/cmu-wordmark-stacked-r-c.png", width='stretch')
        st.header("AI Configuration")
        
        available_providers = []
        if st.secrets.get("DEEPSEEK_API_KEY"): available_providers.append("DeepSeek")
        if st.secrets.get("GEMINI_API_KEY"): available_providers.append("Google Gemini")
        
        if not available_providers: 
            st.error("No AI provider API key found in app secrets.")
            st.stop()
            
        ai_provider = st.selectbox("Choose AI Provider", available_providers) if len(available_providers) > 1 else available_providers[0]
        st.info(f"Using **{ai_provider}** API for analysis.")
        st.markdown("---")
        st.info("This tool is a proof-of-concept and not an official admissions tool.")

    # Select the correct chat functions based on user choice
    api_key = st.secrets.get("DEEPSEEK_API_KEY") if ai_provider == "DeepSeek" else st.secrets.get("GEMINI_API_KEY")
    analysis_function = get_deepseek_rag_analysis if ai_provider == "DeepSeek" else get_gemini_rag_analysis

    # Load and process data (runs once due to caching)
    df_programs = get_cmu_program_data()
    if df_programs.empty:
        st.warning("Program data could not be loaded. Cannot proceed."); return

    # Build the searchable keyword contexts
    program_contexts = build_program_contexts(df_programs)

    # --- Interactive Questionnaire ---
    st.subheader("Tell us about yourself:")
    
    common_majors = ["Computer Science", "Electrical Engineering", "Mechanical Engineering", "Civil Engineering", 
                     "Chemical Engineering", "Biomedical Engineering", "Materials Science", "Physics", 
                     "Mathematics", "Industrial Design/Art", "Other Engineering"]
    
    with st.form("student_profile_form"):
        col1, col2 = st.columns(2)
        with col1:
            degree_level = st.radio("What degree level are you pursuing?", ("M.S.", "Ph.D."), horizontal=True)
            background = st.multiselect("What is your academic background?", options=common_majors, default=["Mechanical Engineering"])
        with col2:
            career_goal = st.selectbox("What is your primary career ambition?", 
                                     ("Industry Leadership (e.g., Tech, Manufacturing, Energy)", 
                                      "Research & Academia (e.g., Professor, National Lab Scientist)",
                                      "Startup & Entrepreneurship",
                                      "Government & Public Policy"))
        
        learning_style = st.slider(
            "What's your preferred learning style?", 0, 100, 50,
            help="0 = Purely Theoretical/Research, 100 = Highly Applied/Project-Based"
        )
        
        keywords = st.text_area(
            "List specific keywords, topics, or technologies that interest you.",
            placeholder="e.g., machine learning, robotics, sustainable energy, quantum computing, battery technology..."
        )
        
        submitted = st.form_submit_button("🎓 Find My Program", width='stretch')

    if submitted:
        # --- Synthesize User Query ---
        style_desc = ""
        if learning_style < 20: style_desc = "a heavily theoretical and research-focused program"
        elif learning_style < 40: style_desc = "a program with a strong theoretical basis"
        elif learning_style < 60: style_desc = "a balanced program with both theory and hands-on projects"
        elif learning_style < 80: style_desc = "a project-driven program"
        else: style_desc = "a highly applied, hands-on, and project-based program"
        
        synthesized_query = (
            f"I am a student with a background in {', '.join(background)} looking for a {degree_level} program. "
            f"My primary career goal is {career_goal.split('(')[0].strip()}. "
            f"I am most interested in topics like {keywords}. "
            f"I thrive in {style_desc}."
        )

        with st.expander("Your Generated Profile for AI Matching"):
            st.code(synthesized_query, language='text')
        
        # --- Keyword Matching & RAG (Replaces Embedding) ---
        
        with st.spinner(f"🔍 Finding relevant programs using keyword matching..."):
            # Filter the context dictionary to only include the degree level selected
            filtered_contexts = {
                name: chunks for name, chunks in program_contexts.items() 
                if f"Degree: {degree_level}" in chunks[0]
            }
            
            relevant_chunks = find_relevant_chunks(
                synthesized_query, 
                filtered_contexts, 
                token_limit=3800 # Use a safe token limit
            )

        if not relevant_chunks:
            st.warning("No strong keyword matches found. Try adjusting your criteria or using different keywords.")
            st.stop()
            
        # --- AI Analysis (RAG) ---
        
        # Build the final prompt for the chat model
        context_to_send = "\n\n---\n\n".join(relevant_chunks)
        context_to_send = truncate_context_to_token_limit(context_to_send, 3800) # Final safety check
        
        rag_prompt = f"""You are an expert CMU academic advisor. A student has provided their profile and query. 
        Your task is to analyze the following program descriptions and provide a recommendation.

        **Student Query:**
        "{synthesized_query}"

        ---
        **Provided Program Information (Context):**
        {context_to_send}
        ---

        **Your Task:**
        Based *only* on the provided program information, identify the top 1-3 programs that best match the student's query. 
        For each match, provide a 3-point analysis in markdown:
        - **Program Fit:** How well does this program align with the student's stated background and goals?
        - **Key Application Skills:** Based on their profile, what specific skills should this student highlight?
        - **Potential Career Trajectory:** How does this degree help them achieve their specific career ambitions?

        If no programs in the context are a good fit, please state that clearly.
        """
        
        with st.spinner(f"🤖 Generating personalized AI Advisor analysis using {ai_provider}..."):
            analysis_result = analysis_function(api_key, rag_prompt)
            
            st.subheader("AI-Powered Advisor Analysis")
            st.info(analysis_result)

if __name__ == "__main__":
    main()
