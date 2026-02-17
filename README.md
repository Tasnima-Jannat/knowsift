# KnowSift 🔬
## Your Reliable Research Partner

Generate comprehensive, cited literature reviews automatically using AI. Perfect for researchers, students, and academics who need quick, quality literature reviews.

## ✨ Features

- 🔍 **Semantic Scholar Integration** - Search 14M+ academic papers
- 🤖 **AI-Powered Analysis** - Uses Google Gemini 2.5 Flash for extraction and synthesis
- 🎯 **Semantic Similarity** - Finds the most relevant papers using sentence transformers
- 📊 **Citation-Based Ranking** - Ranks papers by impact (normalized by age)
- 📚 **Multiple Citation Styles** - APA, Harvard, IEEE supported
- 📄 **Multiple Output Formats** - Markdown, HTML, and PDF with automatic fallback
- 💾 **Smart Caching** - Faster subsequent runs for the same topic
- ⚡ **Exponential Backoff** - Automatic retry logic for API rate limits
- 🎨 **Beautiful Streamlit UI** - Easy to use, no coding required
- 🔒 **Secure** - API keys stored in backend, never exposed to users



## 📖 Usage Guide

### Step-by-Step

1. **Enter Your Research Topic**
   - Be specific! Better: "Transformer Architecture in Natural Language Processing"
   - Avoid: "Machine Learning" (too broad)

2. **Select Year Range**
   - Narrow ranges = faster generation
   - Recommended: 3-5 year window for current research

3. **Configure Settings** (in sidebar)
   - **Number of Papers:** 5-25 (default: 10)
   - **Citation Style:** APA, Harvard, or IEEE
   - **Output Format:** Markdown, PDF, or HTML

4. **Generate Review**
   - Click "🚀 Generate Literature Review"
   - Wait 2-3 minutes for generation
   - Progress bar shows current step

5. **Download Your Review**
   - Click download button for your chosen format
   - PDF automatically falls back to HTML if generation fails

### Tips for Best Results

- **Specific topics** yield better results than broad topics
- **Recent years** (2020-2024) often have more papers available
- **Start with 10 papers** for quick testing, increase later
- **First run** takes longer due to model downloads and no caching
- **Subsequent runs** of same topic are much faster (30-60 seconds)

## 🏗️ Architecture

```
KnowSift/
├── app.py                      # Streamlit UI with rate limiting
├── pipeline.py                 # Core literature review generation logic
│   ├── Config                  # Configuration management
│   ├── PaperDiscovery          # Semantic Scholar search
│   ├── SemanticSearcher        # Similarity computation
│   ├── PaperRanker             # Citation-based ranking
│   ├── InformationExtractor    # AI extraction from papers
│   ├── CrossPaperAnalyzer      # Theme and gap analysis
│   ├── LiteratureReviewGenerator # Review synthesis
│   ├── CitationFormatter       # Reference formatting
│   ├── OutputGenerator         # Document creation
│   └── FormatConverter         # PDF/HTML conversion
├── requirements.txt            # Python dependencies
├── .streamlit/secrets.toml     # API keys (not in git)
├── .gitignore                  # Git exclusions
├── cache/                      # Cached API responses (auto-created)
└── output/                     # Generated reviews (auto-created)
```

## ⚙️ Configuration Options

### Rate Limiting

Adjust in `.streamlit/secrets.toml`:
```toml
MAX_REQUESTS_PER_HOUR = 10  # Increase for production use
```

### Cache and Output Directories

Default locations (automatically created):
- `./cache/` - Cached API responses for faster subsequent runs
- `./output/` - Generated literature reviews in all formats

### Clearing Cache

```bash
# Windows
rmdir /s cache

# Linux/Mac
rm -rf cache/
```

## 🧪 Testing

### Quick Test (5 papers)

```bash
streamlit run app.py
```

Then in the UI:
1. Topic: "Machine Learning in Healthcare"
2. Years: 2022-2024
3. Papers: 5
4. Format: Markdown (fastest)
5. Generate

Expected: Completes in 2-3 minutes

### Full Test (All Formats)

Test each output format:
- **Markdown:** Should always work
- **HTML:** Should work if markdown2 is installed
- **PDF:** May fallback to HTML (requires weasyprint or pdfkit + wkhtmltopdf)

## 🚢 Deployment

### Deploy to Streamlit Cloud (Recommended)

1. **Push to GitHub** (excluding secrets.toml - already in .gitignore)

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/knowsift.git
git push -u origin main
```

2. **Go to [share.streamlit.io](https://share.streamlit.io)**

3. **Connect your GitHub repository**

4. **Add secrets in Streamlit Cloud dashboard:**
   - Go to App settings → Secrets
   - Paste your secrets.toml content

5. **Deploy!**

Your app will be live at: `https://your-app-name.streamlit.app`

### Deploy to Other Platforms

#### Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Name your app: `knowsift`

Build and run:
```bash
docker build -t lit-review-generator .
docker run -p 8501:8501 lit-review-generator
```

#### Railway/Render

Add `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

## 🔧 Troubleshooting

### "Rate limit exceeded"

**Cause:** Too many requests in 1 hour
**Solution:**
- Wait 1 hour for reset
- Increase `MAX_REQUESTS_PER_HOUR` in secrets.toml
- Add Semantic Scholar API key for higher limits

### "No papers found"

**Cause:** Topic too specific or year range too narrow
**Solution:**
- Try broader topic
- Expand year range
- Check internet connection
- Verify Semantic Scholar API is accessible

### "API key error"

**Cause:** Invalid or missing Gemini API key
**Solution:**
- Verify `GEMINI_API_KEY` in `.streamlit/secrets.toml`
- Check key is valid at [Google AI Studio](https://makersuite.google.com/app/apikey)
- Ensure no extra spaces or quotes in the key

### "PDF generation failed"

**Cause:** weasyprint or pdfkit not properly installed
**Expected:** App automatically falls back to HTML
**Solution (if you want PDF):**

**Option 1 - weasyprint (recommended):**
```bash
pip install weasyprint
```

**Option 2 - pdfkit:**
```bash
pip install pdfkit
# Also install wkhtmltopdf from https://wkhtmltopdf.org/downloads.html
```

### Slow Performance

**Cause:** Various factors
**Solutions:**
- **First run:** Takes 2-3 minutes (model download + no cache) - this is normal
- **Reduce papers:** Try 5 instead of 25
- **Narrow year range:** Fewer years to search
- **Check cache:** Subsequent runs should be faster (30-60 seconds)

### "Module not found" errors

**Cause:** Missing dependencies
**Solution:**
```bash
pip install -r requirements.txt
```

## 📊 Performance Benchmarks

| Papers | First Run | Cached Run | Output Size |
|--------|-----------|------------|-------------|
| 5      | 2-3 min   | 30-60 sec  | ~1500 words |
| 10     | 3-5 min   | 1-2 min    | ~2000 words |
| 25     | 8-12 min  | 3-5 min    | ~3000 words |

*Times vary based on internet speed, API response times, and system performance*

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add more output formats (Word, LaTeX, BibTeX)
- [ ] Implement PDF text extraction for full paper analysis
- [ ] Add custom AI prompt configuration
- [ ] Create REST API endpoint
- [ ] Add visualization (citation networks, trend charts)
- [ ] Support for multiple languages
- [ ] Database backend for storing reviews
- [ ] User authentication system

## 📄 License

[Your License Here - e.g., MIT License]

## 🙏 Credits

Built with:
- **Semantic Scholar** - Academic paper database (14M+ papers)
- **Google Gemini 2.5 Flash** - AI analysis and generation
- **Sentence Transformers** - Semantic similarity (`all-MiniLM-L6-v2`)
- **Streamlit** - Web interface framework
- **weasyprint/pdfkit** - PDF generation
- **markdown2** - HTML conversion

## 📧 Support

For issues and questions:
- GitHub Issues: https://github.com/yourusername/knowsift/issues
- Email: contact@knowsift.ai

## 🔄 Changelog

### Version 1.0.0 (2024-02-06)
- Initial release
- Support for APA, Harvard, IEEE citations
- Multiple output formats (MD, HTML, PDF)
- Exponential backoff retry logic
- Smart caching system
- Rate limiting for production use

---

**Made with ❤️ for the research community**
