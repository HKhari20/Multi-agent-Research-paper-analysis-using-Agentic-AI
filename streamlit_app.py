import streamlit as st
import PyPDF2
import os
from dotenv import load_dotenv
from agents import MultiAgentSystem
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

# Configure the Streamlit page for an innovative layout
st.set_page_config(
    page_title="Agentic Paper Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern, dark-themed UI with styled Agent Cards
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.main-header {
    font-size: 2.5rem;
    font-weight: 600;
    background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px;
}
.sub-header {
    color: #8892B0;
    font-size: 1.1rem;
    margin-bottom: 30px;
}
.agent-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #1E2127;
    border-left: 5px solid;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    color: #E2E8F0;
}
.agent-summarizer { border-left-color: #FF6B6B; }
.agent-methodology { border-left-color: #4ECDC4; }
.agent-critic { border-left-color: #FFE66D; }
.agent-synth { border-left-color: #a855f7; }

.agent-title {
    font-weight: 600;
    font-size: 1.2rem;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Multi-Agent Research Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-driven knowledge extraction and peer-review for academic papers.</p>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    
    st.markdown("<small>[🔑 Get your Gemini API Key from Google AI Studio](https://aistudio.google.com/app/apikey)</small>", unsafe_allow_html=True)
    
    api_connected = False
    if not api_key:
        st.warning("Please enter your Gemini API Key to enable the agents.")
    else:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            st.success("✅ API Connected Successfully!")
            api_connected = True
        except Exception as e:
            st.error("❌ Connection Failed: Invalid API Key")
            
    if api_connected:
        # Use a fixed free model instead of the dropdown
        selected_model = "gemini-3.5-flash"
        st.info(f"Using fixed free model: {selected_model}")
        
    st.divider()
    
    st.header("📄 Upload Paper")
    uploaded_file = st.file_uploader("Upload a Research Paper (PDF)", type="pdf")
    
    st.divider()
    
    st.markdown("### 🤖 The Agent Team")
    st.markdown("- **Agent 1:** Summarizer 📝\n- **Agent 2:** Methodologist 🔬\n- **Agent 3:** Critic ⚖️\n- **Agent 4:** Synthesizer 🧠")

def extract_text_from_pdf(pdf_file):
    """Extracts raw text from the uploaded PDF document."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

if uploaded_file and api_connected:
    if st.button("🚀 Deploy Agent Team", use_container_width=True, type="primary"):
        
        # 1. Parsing Phase
        with st.spinner("Processing PDF Document..."):
            paper_text = extract_text_from_pdf(uploaded_file)
        
        st.success(f"PDF Successfully Parsed! Initiating Multi-Agent Pipeline...")
        
        # Initialize our Multi-Agent System with the dynamically selected model
        mas = MultiAgentSystem(api_key, selected_model)
        
        # Limit text to roughly 100k characters for API safety/speed
        context = paper_text[:100000]
        
        # 2. Parallel Agent Analysis Phase (Visualized via columns)
        st.markdown("### 🔄 Agent Workspaces (Live Execution)")
        progress_bar = st.progress(0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.spinner("📝 Summarizer Agent is reading..."):
                summary = mas.run_agent(
                    "Summarizer", 
                    "Expert Academic Summarizer", 
                    "Provide a comprehensive, high-level summary of the paper. Include the main problem, proposed solution, and conclusion.", 
                    context
                )
            st.markdown(f'<div class="agent-box agent-summarizer"><div class="agent-title">📝 Summarizer Output</div>{summary}</div>', unsafe_allow_html=True)
            progress_bar.progress(33)
            
            with st.spinner("🔬 Methodologist Agent is extracting techniques..."):
                methodology = mas.run_agent(
                    "Methodologist", 
                    "Strict Data Scientist and Methodologist", 
                    "Extract and detail the algorithms, mathematical models, datasets, and specific methodologies used in this paper.", 
                    context
                )
            st.markdown(f'<div class="agent-box agent-methodology"><div class="agent-title">🔬 Methodologist Output</div>{methodology}</div>', unsafe_allow_html=True)
            progress_bar.progress(66)

        with col2:
            with st.spinner("⚖️ Critic Agent is reviewing..."):
                critique = mas.run_agent(
                    "Critic", 
                    "Harsh but fair Peer Reviewer", 
                    "Identify limitations, weaknesses, and potential biases in the paper's methodology or conclusions.", 
                    context
                )
            st.markdown(f'<div class="agent-box agent-critic"><div class="agent-title">⚖️ Critic Output</div>{critique}</div>', unsafe_allow_html=True)
            progress_bar.progress(85)
            
            with st.spinner("🧠 Synthesizer Agent is compiling..."):
                synth_context = f"Summary:\n{summary}\n\nMethodology:\n{methodology}\n\nCritique:\n{critique}"
                final_report = mas.run_agent(
                    "Synthesizer", 
                    "Lead Editor of an AI Research Journal", 
                    "Based on the outputs of the Summarizer, Methodologist, and Critic, write a cohesive, professional executive briefing.", 
                    synth_context
                )
            st.markdown(f'<div class="agent-box agent-synth"><div class="agent-title">🧠 Final Synthesis</div>{final_report}</div>', unsafe_allow_html=True)
            progress_bar.progress(100)
        
        st.balloons()
        st.divider()
        
        # 3. Final Report Presentation
        st.markdown("## 📊 Executive Briefing")
        st.markdown(final_report)
        
else:
    # Landing Page State - Show architecture visualization
    st.info("Upload a PDF research paper and ensure your API key is provided and connected to begin.")
    
    st.markdown("### 🧬 Agentic Pipeline Architecture")
    
    # Innovative UI element: Interactive Sankey Diagram for Data Flow
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 20,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = ["Raw PDF Data", "Summarizer Agent", "Methodologist Agent", "Critic Agent", "Synthesizer Agent", "Final Executive Briefing"],
          color = ["#E2E8F0", "#FF6B6B", "#4ECDC4", "#FFE66D", "#a855f7", "#1E2127"]
        ),
        link = dict(
          source = [0, 0, 0, 1, 2, 3, 4], 
          target = [1, 2, 3, 4, 4, 4, 5],
          value = [2, 2, 2, 2, 2, 2, 6]
      ))])
    
    fig.update_layout(
        title_text="Data Flow: From Raw Paper to Knowledge Extraction", 
        font_size=12, 
        height=400, 
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)
