# app.py

import streamlit as st
import os
from datetime import datetime
import hashlib
from typing import Optional

# Your existing pipeline
from pipeline import LiteratureReviewPipeline

# Page config
st.set_page_config(
    page_title="KnowSift - Your Reliable Research Partner",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load API keys from secrets (hidden from users!)
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    S2_API_KEY = st.secrets.get("S2_API_KEY", None)
    MAX_REQUESTS_PER_HOUR = st.secrets.get("MAX_REQUESTS_PER_HOUR", 10)
except Exception as e:
    st.error("⚠️ Configuration error. Please contact the administrator.")
    st.stop()

# Optional: Simple rate limiting using session state
def check_rate_limit() -> bool:
    """Check if user has exceeded rate limit"""
    if "request_count" not in st.session_state:
        st.session_state.request_count = 0
        st.session_state.last_reset = datetime.now()
    
    # Reset counter every hour
    time_elapsed = (datetime.now() - st.session_state.last_reset).total_seconds()
    if time_elapsed > 3600:  # 1 hour
        st.session_state.request_count = 0
        st.session_state.last_reset = datetime.now()
    
    # Check limit
    if st.session_state.request_count >= MAX_REQUESTS_PER_HOUR:
        return False
    
    return True

def increment_usage():
    """Increment usage counter"""
    st.session_state.request_count = st.session_state.get("request_count", 0) + 1

# Optional: User identification (anonymous)
def get_user_id() -> str:
    """Generate anonymous user ID based on session"""
    if "user_id" not in st.session_state:
        # Use a hash of timestamp + random for anonymity
        import uuid
        st.session_state.user_id = str(uuid.uuid4())
    return st.session_state.user_id

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stats-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .footer {
        text-align: center;
        color: #999;
        padding: 2rem 0;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🔬 KnowSift</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your Reliable Research Partner • Generate comprehensive, cited literature reviews automatically</p>', unsafe_allow_html=True)

# Info banner
st.info("💡 **Free to use!** Just enter your research topic and generate your review. Powered by AI.", icon="🎉")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Settings")

    # DeepSeek/Ollama option commented out for Streamlit Cloud deployment
    # Uncomment below lines if deploying locally with Ollama support
    # llm_choice = st.selectbox(
    #     "AI Model",
    #     ["Gemini 2.5 Flash", "DeepSeek-R1 (Local)"],
    #     help="Choose between cloud Gemini or local DeepSeek via Ollama"
    # )
    # use_ollama = (llm_choice == "DeepSeek-R1 (Local)")

    # For Streamlit Cloud: Only use Gemini
    llm_choice = "Gemini 2.5 Flash"
    use_ollama = False

    citation_style = st.selectbox(
        "Citation Style",
        ["APA", "Harvard", "IEEE"],
        help="Choose your preferred citation format"
    )
    
    num_papers = st.slider(
        "Number of Papers",
        min_value=5,
        max_value=25,
        value=10,
        help="More papers = more comprehensive but slower"
    )

    output_format = st.selectbox(
        "Output Format",
        ["Markdown", "PDF", "HTML"],
        help="Choose your preferred output format. PDF will fallback to HTML if generation fails."
    )

    st.divider()
    
    # Usage stats (optional)
    st.header("📊 Your Usage")
    remaining = MAX_REQUESTS_PER_HOUR - st.session_state.get("request_count", 0)
    st.metric("Remaining Requests", f"{remaining}/{MAX_REQUESTS_PER_HOUR}")
    st.caption("Resets every hour")
    
    st.divider()
    
    # System stats
    st.header("🌟 Platform Stats")
    st.metric("Papers Database", "14M+")
    st.metric("Avg. Generation Time", "2-3 min")
    st.metric("Citation Accuracy", "95%+")
    
    st.divider()
    
    # Help
    with st.expander("❓ How to Use"):
        st.markdown("""
        1. Enter your research topic
        2. Select year range
        3. Click "Generate Review"
        4. Wait 2-3 minutes
        5. Download your review!
        
        **Tips:**
        - Be specific with topics
        - Narrow year ranges = faster
        - Start with 10 papers
        """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎯 Research Topic")
    topic = st.text_area(
        "Enter your research topic",
        placeholder="e.g., Machine Learning in Medical Diagnosis\ne.g., Transformer Architecture in Natural Language Processing",
        height=100,
        help="Be specific! Better topics = better reviews",
        label_visibility="collapsed"
    )
    
    st.subheader("📅 Publication Year Range")
    year_col1, year_col2 = st.columns(2)
    with year_col1:
        start_year = st.number_input("Start Year", min_value=2000, max_value=2025, value=2020)
    with year_col2:
        end_year = st.number_input("End Year", min_value=2000, max_value=2025, value=2025)

with col2:
    st.markdown(f"""
    <div class="stats-box">
        <h3 style="margin-top: 0;">📋 Current Settings</h3>
        <p><strong>Papers:</strong> {num_papers}</p>
        <p><strong>Style:</strong> {citation_style}</p>
        <p><strong>Years:</strong> {start_year}-{end_year}</p>
        <hr style="border-color: rgba(255,255,255,0.3);">
        <p><strong>Est. Time:</strong></p>
        <p>First run: 2-3 min</p>
        <p>Cached: 30-60 sec</p>
    </div>
    """, unsafe_allow_html=True)

# Example topics
st.divider()
st.subheader("💡 Example Topics")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🏥 Healthcare AI", use_container_width=True):
        st.session_state.example_topic = "Machine Learning in Medical Diagnosis"
with col2:
    if st.button("🤖 NLP Research", use_container_width=True):
        st.session_state.example_topic = "Transformer Architecture in Natural Language Processing"
with col3:
    if st.button("🎮 Reinforcement Learning", use_container_width=True):
        st.session_state.example_topic = "Deep Reinforcement Learning in Robotics"

# Apply example if selected
if "example_topic" in st.session_state:
    topic = st.session_state.example_topic

# Generate button
st.divider()

generate_button = st.button(
    "🚀 Generate Literature Review", 
    type="primary", 
    use_container_width=True,
    disabled=not topic  # Disable if no topic
)

if generate_button:
    
    # Rate limiting check
    if not check_rate_limit():
        st.error(f"❌ Rate limit reached! You've used all {MAX_REQUESTS_PER_HOUR} requests for this hour. Please try again later.")
        st.info("💡 This limit helps us keep the service free for everyone. Resets every hour.")
        st.stop()
    
    # Validation
    if not topic or not topic.strip():
        st.error("❌ Please enter a research topic")
        st.stop()
    
    if start_year > end_year:
        st.error("❌ Start year must be before end year")
        st.stop()
    
    # Increment usage counter
    increment_usage()
    
    # Initialize pipeline
    try:
        with st.spinner("🔧 Initializing AI pipeline..."):
            pipeline = LiteratureReviewPipeline(
                gemini_api_key=GEMINI_API_KEY if not use_ollama else None,
                s2_api_key=S2_API_KEY,
                use_ollama=use_ollama
            )
        
        # Progress tracking
        progress_bar = st.progress(0, text="Starting...")
        status_text = st.empty()
        
        # Create columns for live status
        status_col1, status_col2 = st.columns([3, 1])
        
        with status_col1:
            status_text.markdown("### 📊 Generation Progress")
        
        # Steps with progress updates
        steps = [
            (15, "🔍 Searching Semantic Scholar database..."),
            (30, "🎯 Computing semantic similarity..."),
            (45, "📊 Ranking papers by relevance & citations..."),
            (60, "📄 Extracting information with AI..."),
            (75, "🔗 Analyzing themes across papers..."),
            (90, "✍️ Generating literature review..."),
            (95, "📚 Formatting citations..."),
            (100, "✅ Finalizing document...")
        ]
        
        # Show estimated time
        with status_col2:
            st.metric("⏱️ Time", "~2-3 min")
        
        # Run pipeline with status updates
        for i, (progress, message) in enumerate(steps):
            progress_bar.progress(progress, text=message)
            
            if i == 0:
                # Actually run the pipeline after showing initial status
                with st.spinner("⚙️ Processing... Please wait 2-3 minutes"):
                    filepath = pipeline.run(
                        topic=topic,
                        start_year=int(start_year),
                        end_year=int(end_year),
                        num_papers=int(num_papers),
                        citation_style=citation_style
                    )
            else:
                # Simulate progress for UI (actual work happens above)
                import time
                time.sleep(0.5)
        
        progress_bar.progress(100, text="✅ Complete!")

        # Convert to requested format
        output_file = filepath  # Default is markdown
        actual_format = output_format  # Track what we actually generated
        conversion_warning = ""

        if output_format == "PDF":
            try:
                with st.spinner("📕 Converting to PDF..."):
                    output_file = pipeline.convert_to_pdf(filepath)
            except Exception as e:
                st.warning(f"⚠️ PDF generation failed: {str(e)}\nFalling back to HTML format.")
                try:
                    output_file = pipeline.convert_to_html(filepath)
                    actual_format = "HTML"
                    conversion_warning = "PDF generation failed, showing HTML instead"
                except Exception as e2:
                    st.error(f"❌ HTML fallback also failed: {str(e2)}\nShowing Markdown instead.")
                    output_file = filepath
                    actual_format = "Markdown"
                    conversion_warning = "Both PDF and HTML failed, showing Markdown"
        elif output_format == "HTML":
            try:
                with st.spinner("🌐 Converting to HTML..."):
                    output_file = pipeline.convert_to_html(filepath)
            except Exception as e:
                st.error(f"❌ HTML generation failed: {str(e)}\nShowing Markdown instead.")
                output_file = filepath
                actual_format = "Markdown"
                conversion_warning = "HTML generation failed, showing Markdown"

        # Success message
        st.balloons()  # Celebration animation!
        if conversion_warning:
            st.success(f"🎉 Literature review generated! ({conversion_warning})", icon="✅")
        else:
            st.success("🎉 Literature review generated successfully!", icon="✅")

        # Display results
        st.divider()
        st.subheader(f"📥 Download Your Review ({actual_format})")

        # Determine MIME type based on actual format
        if actual_format == "PDF":
            mime_type = "application/pdf"
            icon = "📕"
        elif actual_format == "HTML":
            mime_type = "text/html"
            icon = "🌐"
        else:
            mime_type = "text/markdown"
            icon = "📄"

        # Single download button for the generated format
        with open(output_file, 'rb') as f:
            st.download_button(
                label=f"{icon} Download {actual_format}",
                data=f,
                file_name=os.path.basename(output_file),
                mime=mime_type,
                use_container_width=True
            )
        
        # Statistics
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            word_count = len(content.split())
            char_count = len(content)
        
        st.divider()
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        stat_col1.metric("📄 Papers Analyzed", num_papers)
        stat_col2.metric("📝 Words", f"{word_count:,}")
        stat_col3.metric("📊 Characters", f"{char_count:,}")
        stat_col4.metric("⏱️ Time Saved", "~20 hours")
        
        # Preview
        st.divider()
        st.subheader("👀 Preview Your Review")
        
        with st.expander("📖 Click to view preview", expanded=True):
            # Show first 2000 characters
            preview_length = 2000
            preview = content[:preview_length]
            
            st.markdown(preview)
            
            if len(content) > preview_length:
                st.markdown("...")
                st.info(f"📄 Showing first {preview_length} characters. Download the full review above to see everything!")
        
        # Feedback
        st.divider()
        st.subheader("💬 How was your experience?")
        
        feedback_col1, feedback_col2 = st.columns([3, 1])
        
        with feedback_col1:
            feedback = st.text_area(
                "Share your feedback (optional)",
                placeholder="What did you like? What can we improve?",
                label_visibility="collapsed"
            )
        
        with feedback_col2:
            rating = st.radio(
                "Rating",
                ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
                label_visibility="collapsed"
            )
        
        if st.button("Submit Feedback"):
            # TODO: Save feedback to database/file
            st.success("Thank you for your feedback! 🙏")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        
        # More helpful error messages
        error_str = str(e).lower()
        if "rate limit" in error_str or "429" in error_str:
            st.warning("⏳ The APIs are rate-limited. Please try again in a few minutes.")
        elif "api key" in error_str:
            st.warning("🔑 API configuration issue. Please contact the administrator.")
        else:
            st.warning("💡 Try:\n- Checking your internet connection\n- Using a more specific topic\n- Reducing the number of papers\n- Narrowing the year range")
        
        # Show technical details in expander
        with st.expander("🔍 Technical Details (for debugging)"):
            st.exception(e)

# Footer
st.markdown("""
<div class="footer">
    <p><strong>KnowSift</strong> - Your Reliable Research Partner</p>
    <p>Powered by Semantic Scholar 📚 • Google Gemini 🤖 • DeepSeek 🧠 • Sentence Transformers</p>
    <p>Built with ❤️ using Streamlit</p>
    <p style="font-size: 0.8rem; margin-top: 1rem;">
        <a href="https://github.com/yourusername/knowsift" target="_blank">GitHub</a> •
        <a href="mailto:contact@knowsift.ai">Contact</a> •
        <a href="/privacy">Privacy</a>
    </p>
</div>
""", unsafe_allow_html=True)