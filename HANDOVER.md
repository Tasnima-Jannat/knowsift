# KnowSift - Handover Guide 🔬

**Your Reliable Research Partner**

This document will help you deploy, manage, and maintain KnowSift for your users.

---

## 🚀 Quick Deployment (5 Minutes)

### Step 1: Get Your Gemini API Key (Free)

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key (looks like: `AIzaSy...`)

**Rate Limits (Free Tier):**
- 10 requests per minute
- 250 requests per day
- Perfect for personal or small-scale use!

---

### Step 2: Deploy to Streamlit Cloud (Free)

1. **Create GitHub account** (if you don't have one): [github.com](https://github.com)

2. **Upload KnowSift:**
   - Create new repository called "knowsift"
   - Upload all files from the project folder
   - **Important:** DON'T upload `.streamlit/secrets.toml` (contains API keys!)

3. **Deploy:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select repository: `knowsift`
   - Main file path: `app.py`

4. **Add Your API Key:**
   - Before deploying, click "Advanced settings"
   - Go to "Secrets" section
   - Paste this (replace with YOUR actual key):
   ```toml
   GEMINI_API_KEY = "your-actual-key-here"
   S2_API_KEY = ""
   MAX_REQUESTS_PER_HOUR = 10
   ```
   - Click "Deploy"

5. **Done!** Your app will be live at: `https://your-app-name.streamlit.app`

---

## 📱 Sharing Your App

Once deployed, share this link with users:
```
https://your-app-name.streamlit.app
```

**What users can do:**
- Generate literature reviews on any topic
- Choose citation styles (APA, Harvard, IEEE)
- Download in multiple formats (Markdown, PDF, HTML)
- Analyze 5-25 papers per review
- Free to use (rate limited to 10 reviews/hour per user)

---

## 🔧 How to Make Changes

### Change the Rate Limit

If you want to allow more reviews per hour:

1. Go to your Streamlit Cloud dashboard
2. Click on your app
3. Go to "Settings" → "Secrets"
4. Change this line:
```toml
MAX_REQUESTS_PER_HOUR = 20  # Change from 10 to 20 (or any number)
```
5. Save (app will automatically restart)

### Update App Content

1. Edit files on GitHub (click the file → Edit button)
2. Make your changes
3. Commit the changes
4. Streamlit automatically redeploys in 2-3 minutes

**Common edits:**
- Change tagline: Edit `app.py` line 95
- Update footer email: Edit `app.py` line 394
- Modify example topics: Edit `app.py` lines 204-211

---

## 📊 Monitoring Your App

### View Usage Stats

Streamlit Cloud dashboard shows:
- Number of users
- Active sessions
- Error logs (if something breaks)

### View App Logs

1. Go to Streamlit Cloud dashboard
2. Click on your app
3. Click "Manage app" → "Logs"
4. See real-time activity

---

## 🆘 Troubleshooting

### Problem: "Rate limit exceeded"
**Cause:** Gemini API hit the 10/min or 250/day limit
**Fix:** Wait or upgrade to paid tier ($0.35 per 1M tokens)

### Problem: "No papers found"
**Cause:** Topic too specific or internet connection issue
**Fix:** Tell users to broaden topic or check connection

### Problem: App is slow
**Cause:** Normal - AI processing takes 2-3 minutes
**Fix:** Nothing needed, this is expected

### Problem: App shows error message
**Fix:**
1. Check logs in Streamlit dashboard
2. Verify API key is still valid
3. Check GitHub repo for issues
4. Restart app in dashboard

---

## 💰 Cost Breakdown

**Free Tier (Current Setup):**
- Streamlit Cloud: FREE
- Gemini API: FREE (with limits)
- GitHub: FREE
- **Total: $0/month** ✅

**If You Need More:**
- Gemini API Paid: ~$0.35 per 1M tokens (~10,000 reviews)
- Still uses free Streamlit Cloud
- **Total: ~$5-10/month** for heavy usage

---

## 🔐 Security Notes

**Your API key is safe because:**
- It's stored in Streamlit Cloud secrets (not in code)
- Users never see it
- It's not in GitHub repository
- Only you can access it in dashboard

**To regenerate API key:**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Delete old key
3. Create new key
4. Update in Streamlit Cloud secrets

---

## 📞 Getting Help

**If you get stuck:**
1. Check logs in Streamlit Cloud dashboard
2. Read error messages carefully
3. Google the error (usually has solutions)
4. Check Streamlit documentation: [docs.streamlit.io](https://docs.streamlit.io)

**For bugs in the app:**
- Create an issue on GitHub repository
- Or contact: contact@knowsift.ai

---

## 🎓 Quick Reference

| Task | Where | How |
|------|-------|-----|
| Change rate limit | Streamlit secrets | Edit `MAX_REQUESTS_PER_HOUR` |
| View logs | Streamlit dashboard | Manage app → Logs |
| Update API key | Streamlit secrets | Edit `GEMINI_API_KEY` |
| Change content | GitHub | Edit files directly |
| Check usage | Streamlit dashboard | App analytics |

---

## ✅ Success Checklist

After handover, verify:
- [ ] App is live and accessible
- [ ] You can generate a test review
- [ ] Downloads work (Markdown, HTML, PDF)
- [ ] Rate limiting shows correct numbers
- [ ] You have access to Streamlit dashboard
- [ ] You have access to GitHub repository
- [ ] You know how to view logs
- [ ] You have API key backed up somewhere safe

---

## 🎉 You're All Set!

KnowSift is now ready for your users. The app:
- Runs 24/7 automatically
- Updates when you push to GitHub
- Handles rate limiting automatically
- Saves citations in proper format
- Generates professional literature reviews

**Just share the URL and let your users start researching!** 📚

---

*Made with ❤️ for the research community*
