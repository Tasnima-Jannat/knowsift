# KnowSift - Quick Start Guide

**Get your app live in 10 minutes!** ⚡

---

## For You (The Developer)

### 3 Simple Steps:

**1. Get API Key** (2 minutes)
- Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
- Click "Create API Key"
- Copy the key

**2. Push to GitHub** (3 minutes)
```bash
cd "d:\Codes\Test\Python\Lit Review Generator"
git init
git add .
git commit -m "Deploy KnowSift"
git remote add origin https://github.com/YOUR_USERNAME/knowsift.git
git push -u origin main
```

**3. Deploy to Streamlit** (5 minutes)
- Go to [share.streamlit.io](https://share.streamlit.io)
- Click "New app"
- Select your `knowsift` repository
- In "Advanced settings" → "Secrets", paste:
```toml
GEMINI_API_KEY = "your-key-here"
S2_API_KEY = ""
MAX_REQUESTS_PER_HOUR = 10
```
- Click "Deploy"

**Done!** 🎉

Your app URL: `https://your-app-name.streamlit.app`

---

## For Your Friend (The User)

### Using KnowSift:

1. **Visit the app URL** you shared
2. **Enter a research topic:**
   - Example: "Machine Learning in Healthcare"
   - Example: "Climate Change Impact on Agriculture"
3. **Choose settings** (sidebar):
   - Citation style: APA, Harvard, or IEEE
   - Number of papers: 5-25
   - Output format: Markdown, PDF, or HTML
4. **Click "Generate Review"**
5. **Wait 2-3 minutes** ☕
6. **Download your review!** 📥

### Tips for Best Results:

✅ **Good topics:**
- "Transformer Models in Natural Language Processing"
- "CRISPR Gene Editing in Medicine"
- "Quantum Computing Applications"

❌ **Too broad:**
- "Machine Learning" (be more specific)
- "Science" (too general)

✅ **Good year ranges:**
- 2020-2024 (recent research)
- 2018-2023 (5-year window)

### Features:

- **Free to use** (10 reviews per hour)
- **No registration** required
- **Professional citations** in your chosen style
- **Multiple formats** (Markdown, HTML, PDF)
- **14M+ papers** from Semantic Scholar
- **AI-powered** analysis and synthesis

---

## Handover Checklist

When giving this to your friend:

- [ ] App is deployed and working
- [ ] You've tested it yourself
- [ ] You've added her as GitHub collaborator
- [ ] You've shared Streamlit dashboard access
- [ ] She has the app URL
- [ ] She knows how to use it (walked through above)
- [ ] She knows how to check if it's down (visit URL)
- [ ] She has your contact info for support

---

## Support

### For Users (Your Friend's Users)

**Problem?** Try these:
1. Check internet connection
2. Try a different topic
3. Reduce number of papers to 5
4. Wait a few minutes and try again

Still broken? Contact: contact@knowsift.ai

### For Admin (Your Friend)

See `HANDOVER.md` for:
- How to check logs
- How to change settings
- How to update content
- Troubleshooting guide

---

## Quick Reference

| What | Where | Time |
|------|-------|------|
| Generate review | Main page | 2-3 min |
| Check logs | Streamlit dashboard | 1 min |
| Update content | GitHub | 5 min |
| Change rate limit | Streamlit secrets | 1 min |
| View usage stats | Streamlit dashboard | 1 min |

---

## That's It!

KnowSift is:
- ✅ Free forever (with rate limits)
- ✅ No maintenance required
- ✅ Updates automatically from GitHub
- ✅ Professional quality output
- ✅ Easy to use

**Just share the URL and let users generate reviews!** 🚀

---

Need more details?
- **Deployment:** See `DEPLOYMENT.md`
- **Management:** See `HANDOVER.md`
- **Technical:** See `README.md`
