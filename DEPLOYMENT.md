# Deployment Guide

## Vercel Deployment

### Prerequisites

1. **GitHub Repository**: Push your code to GitHub
2. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
3. **Environment Variables**: Prepare your TomTom API key

### Quick Deploy Steps

1. **Import Project**
   - Go to [vercel.com](https://vercel.com)
   - Click "Import Project"
   - Select your GitHub repository

2. **Configure Environment Variables**
   - In Vercel dashboard, go to Settings → Environment Variables
   - Add: `TOMTOM_API_KEY` = your actual API key

3. **Deploy**
   - Vercel will automatically detect `vercel.json` and deploy
   - Your app will be live at `https://your-project.vercel.app`

### Important Notes

⚠️ **Large Dependencies**: 
- PyTorch, OSMnx, and other large packages may cause build timeouts on Vercel's free tier
- Consider using Docker deployment or alternative hosting (Render, Hugging Face Spaces)

⚠️ **Data Files**:
- Large data files (`.graphml`, `.parquet`, `.pt`) are excluded via `.gitignore`
- For production, upload these to cloud storage (S3, Google Cloud Storage) and load them at runtime
- Or include minimal sample data for demo purposes

### Alternative Hosting Options

**Render** (Recommended for Streamlit):
- Better support for Python apps
- Free tier available
- Use `render.yaml` for configuration

**Hugging Face Spaces**:
- Great for ML demos
- Free GPU available
- Built-in Streamlit support

**Docker + Any Cloud**:
- Full control over environment
- Can pre-install heavy dependencies
- Use provided `Dockerfile`

### Environment Variables

Create `.env.example` (template):
```
TOMTOM_API_KEY=your_api_key_here
```

Never commit `.env` file - it's in `.gitignore`

