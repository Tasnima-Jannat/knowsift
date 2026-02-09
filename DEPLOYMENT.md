# KnowSift - Deployment Instructions

**Step-by-step guide for deploying to Streamlit Cloud**

---

## Prerequisites

- [ ] GitHub account
- [ ] Google account (for Gemini API)
- [ ] All KnowSift files ready

---

## Step 1: Get Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with Google
3. Click **"Get API Key"** or **"Create API Key"**
4. Copy the key (starts with `AIzaSy...`)
5. **Save it somewhere safe!**

**Rate Limits (Free):**
- 10 requests/minute
- 250 requests/day
- Free forever

---

## Step 2: Create GitHub Repository

### Option A: Using GitHub Desktop (Easier)

1. Download [GitHub Desktop](https://desktop.github.com)
2. Install and sign in
3. Click **"Add"** → **"Create New Repository"**
4. Name: `knowsift`
5. Local path: `d:\Codes\Test\Python\Lit Review Generator`
6. Click **"Create Repository"**
7. Click **"Publish repository"**
8. **Uncheck** "Keep this code private" (Streamlit needs public repos)
9. Click **"Publish"**

### Option B: Using Command Line

```bash
cd "d:\Codes\Test\Python\Lit Review Generator"
git init
git add .
git commit -m "Initial KnowSift deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/knowsift.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

---

## Step 3: Deploy to Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**

2. **Sign in with GitHub**

3. **Click "New app"**

4. **Fill in details:**
   - Repository: `YOUR_USERNAME/knowsift`
   - Branch: `main`
   - Main file path: `app.py`

5. **Click "Advanced settings"**

6. **Add Secrets:**
   - Click on "Secrets" section
   - Paste this (replace `YOUR_KEY_HERE` with your actual Gemini API key):

   ```toml
   GEMINI_API_KEY = "YOUR_KEY_HERE"
   S2_API_KEY = ""
   MAX_REQUESTS_PER_HOUR = 10
   ```

7. **Click "Deploy!"**

8. **Wait 3-5 minutes** for deployment to complete

---

## Step 4: Test Your App

Once deployed, Streamlit will show your app URL like:
```
https://knowsift-abc123.streamlit.app
```

**Test these:**
1. Visit the URL
2. Enter a test topic: "Machine Learning in Healthcare"
3. Set years: 2022-2024
4. Papers: 5 (quick test)
5. Click "Generate Review"
6. Wait 2-3 minutes
7. Download the review

**If it works:** ✅ You're done!

**If it fails:** See troubleshooting below

---

## Step 5: Share Your App

Your app is now live 24/7 at:
```
https://your-app-name.streamlit.app
```

Share this link with users! 🎉

---

## Troubleshooting

### "API key error"
❌ **Problem:** Invalid or missing API key
✅ **Fix:**
1. Go to Streamlit dashboard
2. Click your app → Settings → Secrets
3. Verify `GEMINI_API_KEY` is correct
4. Save and restart app

### "App failed to load"
❌ **Problem:** Code error or missing dependencies
✅ **Fix:**
1. Click "Manage app" → "Logs"
2. Read error message
3. Check if all files were uploaded to GitHub
4. Verify `requirements.txt` exists

### "No papers found"
❌ **Problem:** Topic too specific or API connection issue
✅ **Fix:** Try a broader topic like "Machine Learning"

### App is slow
✅ **This is normal!** First-time generation takes 2-3 minutes while:
- Loading models (~90MB download)
- Searching papers
- Processing with AI

Subsequent runs are faster (30-60 seconds) due to caching.

---

## Updating Your App

After initial deployment, to make changes:

1. Edit files on GitHub (or locally and push)
2. Commit changes
3. Streamlit auto-deploys in 2-3 minutes

**No need to redeploy manually!**

---

## Configuration Options

### Increase Rate Limit

In Streamlit Cloud secrets, change:
```toml
MAX_REQUESTS_PER_HOUR = 20  # Allow 20 reviews/hour instead of 10
```

### Change App Name

In Streamlit dashboard:
1. Click app → Settings
2. Change "App name"
3. Save (URL will update)

### Custom Domain (Optional)

1. Get a domain (e.g., from Namecheap, GoDaddy)
2. In Streamlit settings → Custom domain
3. Follow DNS setup instructions

---

## Maintenance

### Weekly
- [ ] Check app is running (visit URL)
- [ ] Review usage logs if issues reported

### Monthly
- [ ] Check API key is still valid
- [ ] Review user feedback
- [ ] Update content if needed

### As Needed
- [ ] Update if users report bugs
- [ ] Adjust rate limits based on usage
- [ ] Add new features

---

## Backup Plan

**If Streamlit Cloud has issues:**

1. **Quick fix:** Deploy to Railway/Render (similar process)
2. **Local hosting:** Run on your computer:
   ```bash
   streamlit run app.py
   ```
3. **Share via ngrok:** Temporary public URL:
   ```bash
   ngrok http 8501
   ```

---

## Important Files

| File | Purpose | Edit? |
|------|---------|-------|
| `app.py` | Main app code | Yes, carefully |
| `pipeline.py` | Backend logic | No, unless needed |
| `requirements.txt` | Dependencies | No |
| `.gitignore` | Git exclusions | No |
| `README.md` | Documentation | Yes |
| `.streamlit/secrets.toml` | **DO NOT UPLOAD TO GITHUB!** | Only in Streamlit dashboard |

---

## Success Criteria

✅ Your deployment is successful when:
- App loads at the URL
- Can generate a test review
- Downloads work
- No errors in logs
- Rate limiting shows correct counts

---

## Next Steps After Deployment

1. **Share URL** with your friend
2. **Give her access:**
   - Add her as collaborator on GitHub
   - Share Streamlit dashboard access
3. **Walk through HANDOVER.md** together
4. **Do a test generation** together
5. **Show her how to check logs**

---

## Support Resources

- **Streamlit Docs:** [docs.streamlit.io](https://docs.streamlit.io)
- **Gemini API Docs:** [ai.google.dev](https://ai.google.dev)
- **GitHub Help:** [docs.github.com](https://docs.github.com)
- **KnowSift Issues:** Create issue on your GitHub repo

---

**You're ready to deploy! 🚀**

Follow the steps above, and your app will be live in minutes. Good luck!
