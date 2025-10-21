# tartans.py
# CMU College of Engineering Program Navigator
# Author: Gemini (Google AI)
# Date: October 21, 2025 (Corrected Keyword RAG Version - v5.1)

import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import google.generativeai as genai
from typing import Optional, Dict, Any, List
from difflib import SequenceMatcher # For keyword matching

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CMU Engineering Program Navigator",
    page_icon="🎓",
    layout="wide"
)

# --- UTILITIES ---

def robust_request(url: str, headers: Dict[str, str], timeout: int = 15) -> Optional[str]:
    """Handles HTTP requests with basic error checking."""
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        st.warning(f"Request failed for {url}: {e}")
        return None

def simple_text_split(text, chunk_size=750):
    """Splits text into chunks of a given size."""
    # Ensure text is a string
    text = str(text) if text is not None else ""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def find_relevant_chunks(query, contexts, token_limit=3800):
    """
    Finds relevant chunks using SequenceMatcher (keyword matching).
    'contexts' is a dict where key=program_name, value=list of text chunks.
    """
    relevant_chunks = []
    total_tokens = 0
    # Estimate tokens needed for the rest of the prompt structure
    prompt_overhead = 200
    query_token_count = len(query.split()) + prompt_overhead
    available_tokens = token_limit - query_token_count
    chunks_list = []

    for program_name, chunks in contexts.items():
        for chunk in chunks:
            # Ensure chunk is a string before matching
            chunk_str = str(chunk) if chunk is not None else ""
            similarity = SequenceMatcher(None, query, chunk_str).ratio()
            # Only consider chunks with some similarity and content
            if similarity > 0.05 and chunk_str:
                token_count = len(chunk_str.split())
                chunks_list.append((chunk_str, similarity, token_count, program_name))

    # Sort by descending similarity
    chunks_list.sort(key=lambda x: x[1], reverse=True)

    # Add chunks until token limit is reached
    seen_chunks = set() # Avoid adding duplicate chunks if descriptions are similar
    for chunk, _, token_count, _ in chunks_list:
        if chunk not in seen_chunks and total_tokens + token_count <= available_tokens:
            relevant_chunks.append(chunk)
            seen_chunks.add(chunk)
            total_tokens += token_count
        elif total_tokens + token_count > available_tokens:
            break # Stop adding chunks if we exceed the limit

    return relevant_chunks

def truncate_context_to_token_limit(context, max_tokens=3800):
    """Ensures the final context string fits within the token limit."""
    words = context.split()
    if len(words) > max_tokens:
        words = words[:max_tokens]
    return " ".join(words)

# --- DATA SCRAPING & PROCESSING ---

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
            h1 = soup.find('h1')
            title = soup.find('title')
            if h1:
                program_name_base = h1.get_text(strip=True).replace("Graduate", "").strip()
            elif title:
                program_name_base = title.get_text(strip=True).split('|')[0].strip()

            description_text = 'No sufficiently detailed description found.'
            # Try finding a main content area first for better quality text
            main_content = soup.find('main') or soup.find('article') or soup.body
            if main_content:
                 description_paragraphs = main_content.find_all('p', limit=15) # Increase limit slightly
                 text_parts = [p.get_text(" ", strip=True) for p in description_paragraphs if len(p.get_text(strip=True)) > 70] # Be stricter on length
                 if text_parts:
                     description_text = ' '.join(text_parts)
                     # Limit description length more carefully
                     max_desc_len = 1500
                     if len(description_text) > max_desc_len:
                         description_text = description_text[:max_desc_len].rsplit(' ', 1)[0] + '...'
            else:
                 st.warning(f"Could not find main content area for {page_name}")


            programs_list.append({
                'name': f"Master of Science ({program_name_base})",
                'url': page_url, 'description': description_text, 'degree_type': 'M.S.'
            })
            programs_list.append({
                'name': f"Doctor of Philosophy ({program_name_base})",
                'url': page_url, 'description': description_text, 'degree_type': 'Ph.D.'
            })

            st.info(f"Indexed M.S. and Ph.D. from: {program_name_base}")
            time.sleep(0.05) # Slightly faster sleep

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
    if df_programs.empty:
        return contexts
    for _, row in df_programs.iterrows():
        # Create a single text block for each program
        full_text = f"Program Name: {row['name']}\nDegree Type: {row['degree_type']}\nSource URL: {row['url']}\n\nProgram Description:\n{row['description']}"

        # Split that text block into chunks
        contexts[row['name']] = simple_text_split(full_text, chunk_size=750)
    st.success(f"Created searchable text chunks for {len(contexts)} programs.")
    return contexts

# --- AI PROVIDER (CHAT-ONLY) FUNCTIONS ---

def _call_deepseek_chat_api(api_key: str, payload: Dict[str, Any]) -> str:
    """Specific function to call DeepSeek chat completions API."""
    endpoint = "https://api.deepseek.com/v1/chat/completions"
    for attempt in range(3):
        try:
            response = requests.post(endpoint, headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=60) # Increased timeout
            response.raise_for_status()
            # Add basic check for expected response structure
            json_response = response.json()
            if 'choices' in json_response and len(json_response['choices']) > 0 and 'message' in json_response['choices'][0] and 'content' in json_response['choices'][0]['message']:
                 return json_response['choices'][0]['message']['content']
            else:
                 st.error(f"DeepSeek API returned unexpected response format: {json_response}")
                 return "AI analysis unavailable: Unexpected API response format."
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            st.error(f"DeepSeek Chat API call failed after multiple retries: {e}")
            return f"AI analysis unavailable: DeepSeek API call failed: {e}"
        except Exception as e:
            st.error(f"DeepSeek Chat API processing failed: {e}")
            return f"AI analysis unavailable: DeepSeek API processing failed: {e}"
    return "AI analysis unavailable: Failed after multiple retries."

@st.cache_data(ttl=3600) # Cache analysis results for 1 hour
def get_deepseek_rag_analysis(api_key: str, prompt: str) -> str:
    """Gets a RAG analysis from DeepSeek using a provided prompt."""
    if not api_key: return "AI analysis unavailable: DeepSeek API key missing."

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3, # Lower temp for more factual, context-based answers
        "max_tokens": 1500, # Allow longer analysis
        "stream": False
    }
    return _call_deepseek_chat_api(api_key, payload)

@st.cache_data(ttl=3600) # Cache analysis results for 1 hour
def get_gemini_rag_analysis(api_key: str, prompt: str) -> str:
    """Gets a RAG analysis from Gemini using a provided prompt."""
    if not api_key: return "AI analysis unavailable: Gemini API key missing."
    try:
        genai.configure(api_key=api_key)
        # Configure for slightly more deterministic output if needed, although Gemini Pro is generally good
        generation_config = genai.types.GenerationConfig(
            temperature=0.3,
            max_output_tokens=1500
        )
        model = genai.GenerativeModel('gemini-1.5-pro-latest', generation_config=generation_config)
        response = model.generate_content(prompt)
        # Add safety check for Gemini response
        if response.parts:
            return response.text
        else:
             st.warning(f"Gemini analysis generation blocked or empty. Reason: {response.prompt_feedback}")
             return f"AI analysis generation blocked. Reason: {response.prompt_feedback}"

    except Exception as e:
        st.error(f"Gemini analysis failed to generate: {e}")
        return f"AI analysis failed to generate (Gemini): {e}"

# --- MAIN APPLICATION LOGIC ---
def main():
    st.title("🎓 CMU Engineering Program Navigator (Keyword Search Version)")
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

        # Default to Gemini if available, as DeepSeek embeddings are problematic
        default_provider_index = 0
        if "Google Gemini" in available_providers:
            default_provider_index = available_providers.index("Google Gemini")

        ai_provider = st.selectbox(
            "Choose AI Provider for Analysis",
            available_providers,
            index=default_provider_index
            ) if len(available_providers) > 1 else available_providers[0]

        st.info(f"Using **{ai_provider} Chat API** for analysis.\n\n*(Embeddings are bypassed in this version)*")
        st.markdown("---")
        st.info("This tool is a proof-of-concept and not an official admissions tool.")

    # Select the correct chat functions based on user choice
    api_key = st.secrets.get("DEEPSEEK_API_KEY") if ai_provider == "DeepSeek" else st.secrets.get("GEMINI_API_KEY")
    if not api_key:
         st.error(f"{ai_provider} API key not found in secrets. Please add it.")
         st.stop()

    analysis_function = get_deepseek_rag_analysis if ai_provider == "DeepSeek" else get_gemini_rag_analysis

    # Load and process data (runs once due to caching)
    df_programs = get_cmu_program_data()
    if df_programs.empty:
        st.error("Program data could not be loaded. Cannot proceed.")
        st.stop() # Stop execution if data loading fails

    # Build the searchable keyword contexts (runs once due to caching)
    with st.spinner("Building searchable text index... (Runs once)"):
        program_contexts = build_program_contexts(df_programs)
    if not program_contexts:
        st.error("Failed to build program context index. Cannot proceed.")
        st.stop()


    # --- Interactive Questionnaire ---
    st.subheader("Tell us about yourself:")

    common_majors = ["Computer Science", "Electrical Engineering", "Mechanical Engineering", "Civil Engineering",
                     "Chemical Engineering", "Biomedical Engineering", "Materials Science", "Physics",
                     "Mathematics", "Industrial Design/Art", "Other Engineering"]

    with st.form("student_profile_form"):
        col1, col2 = st.columns(2)
        with col1:
            degree_level = st.radio("What degree level are you pursuing?", ("M.S.", "Ph.D."), horizontal=True, key="degree_level")
            background = st.multiselect("What is your academic background?", options=common_majors, default=["Mechanical Engineering"], key="background")
        with col2:
            career_goal = st.selectbox("What is your primary career ambition?",
                                     ("Industry Leadership (e.g., Tech, Manufacturing, Energy)",
                                      "Research & Academia (e.g., Professor, National Lab Scientist)",
                                      "Startup & Entrepreneurship",
                                      "Government & Public Policy"), key="career_goal")

        learning_style = st.slider(
            "What's your preferred learning style?", 0, 100, 50,
            help="0 = Purely Theoretical/Research, 100 = Highly Applied/Project-Based", key="learning_style"
        )

        keywords = st.text_area(
            "List specific keywords, topics, or technologies that interest you. (e.g., machine learning, robotics, sustainable energy)",
            placeholder="Be specific! This helps find better matches.", key="keywords"
        )

        submitted = st.form_submit_button("🎓 Find My Program", use_container_width=True) # Changed back from width='stretch' based on your prev code

    if submitted:
        # --- Synthesize User Query ---
        style_desc = ""
        if learning_style < 20: style_desc = "a heavily theoretical and research-focused program"
        elif learning_style < 40: style_desc = "a program with a strong theoretical basis"
        elif learning_style < 60: style_desc = "a balanced program with both theory and hands-on projects"
        elif learning_style < 80: style_desc = "a project-driven program"
        else: style_desc = "a highly applied, hands-on, and project-based program"

        # Ensure background list is not empty for joining
        background_str = ', '.join(background) if background else "an unspecified background"

        synthesized_query = (
            f"I am a student with a background in {background_str} looking for a {degree_level} program. "
            f"My primary career goal is {career_goal.split('(')[0].strip()}. "
            f"My preferred learning style involves {style_desc}. "
            f"I am particularly interested in topics like: {keywords if keywords else 'general engineering topics'}."
        )

        with st.expander("Your Generated Profile for AI Matching"):
            st.write(synthesized_query) # Use write for better wrapping

        # --- Keyword Matching & RAG ---
        max_context_tokens = 3800 # Define max tokens for context

        with st.spinner(f"🔍 Finding relevant programs for a {degree_level} applicant using keyword matching..."):
            # Filter the context dictionary to only include the degree level selected
            filtered_contexts = {}
            for name, chunks in program_contexts.items():
                 # Check if the first chunk (which contains metadata) mentions the degree type
                 if chunks and f"Degree Type: {degree_level}" in chunks[0]:
                      filtered_contexts[name] = chunks

            if not filtered_contexts:
                 st.warning(f"No programs found for the degree level '{degree_level}'. Check scraping data.")
                 st.stop()

            relevant_chunks = find_relevant_chunks(
                synthesized_query,
                filtered_contexts,
                token_limit=max_context_tokens
            )

        if not relevant_chunks:
            st.warning("Could not find relevant program information based on your keywords. Try broadening your interests or checking the scraped data.")
            # Optionally, show some generic info or stop
            st.stop()

        # --- AI Analysis (RAG) ---
        context_to_send = "\n\n---\n\n".join(relevant_chunks)
        # Final safety check on token limit before sending to API
        context_to_send = truncate_context_to_token_limit(context_to_send, max_context_tokens)

        rag_prompt = f"""You are an expert CMU academic advisor. A prospective student has provided their profile and interests.
        Your task is to analyze the provided snippets of CMU Engineering program descriptions and recommend the best fit(s).

        **Student Profile & Query:**
        "{synthesized_query}"

        ---
        **Provided Program Information Snippets (Use ONLY this information):**
        {context_to_send}
        ---

        **Instructions:**
        1. Carefully review the Student Profile & Query.
        2. Analyze *each* program snippet provided in the Context section.
        3. Identify the top 1-3 programs from the snippets that appear to be the **best match** for the student.
        4. For *each* recommended program, provide a concise 3-point analysis in markdown format:
           - **Program Fit:** Explain *why* this program aligns with the student's background, goals, and interests, referencing specifics from their profile and the program description snippet.
           - **Potential Considerations:** Mention any potential mismatches or aspects the student should investigate further (e.g., if the description focuses heavily on theory but the student prefers applied work).
           - **Next Steps:** Suggest specific keywords or questions the student could use to search the official program website (use the Source URL if provided in the snippet) for more details.
        5. If **no programs** in the provided snippets seem like a good fit, state that clearly and explain why. Do not invent information or recommend programs not present in the snippets.
        6. Keep your entire response focused and based *strictly* on the text provided in the "Program Information Snippets" section. Do not add external knowledge.
        """

        with st.spinner(f"🤖 Generating personalized AI Advisor analysis using {ai_provider}..."):
            # Pass the API key explicitly to the analysis function
            analysis_result = analysis_function(api_key, rag_prompt)

            st.subheader(f"AI Advisor Analysis (using {ai_provider})")
            # Display results using st.markdown for better formatting
            st.markdown(analysis_result)

            # Optional: Show the context that was sent to the AI
            with st.expander("Show Context Sent to AI"):
                st.text(context_to_send)

if __name__ == "__main__":
    main()
