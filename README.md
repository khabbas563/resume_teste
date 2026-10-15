# AI Resume ATS Checker

A Streamlit app that scores a resume for Applicant Tracking Systems (ATS) and suggests concrete improvements, powered by Google Gemini Flash.

## Features

- Upload a resume as **PDF, DOCX or TXT**
- Optional **job description** for a targeted keyword match
- **ATS score (0-100)** with a breakdown: formatting, keywords, experience, skills, education, readability
- Strengths, weaknesses, **missing keywords**, formatting issues
- Prioritised **improvement list** and **before/after bullet rewrites**
- Download the report as JSON

## Project structure

```
.
├── app.py             # Streamlit app
├── requirements.txt   # Python dependencies
└── README.md
```

## Run locally

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).
2. Install and run:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Provide your key (pick one):

```bash
# Option A: environment variable
export GEMINI_API_KEY="your-key"          # Windows PowerShell: $env:GEMINI_API_KEY="your-key"

# Option B: Streamlit secrets file (never commit it)
mkdir -p .streamlit
echo 'GEMINI_API_KEY = "your-key"' > .streamlit/secrets.toml

# Option C: just paste the key into the sidebar when the app opens
```

4. Start the app:

```bash
streamlit run app.py
```

## Configuration

| Name | Where | Default | Purpose |
|------|-------|---------|---------|
| `GEMINI_API_KEY` | env var or Streamlit secret | none | Your Gemini API key |
| `GEMINI_MODEL` | env var or Streamlit secret | `gemini-3.8-flash` | Gemini model id. If it is retired (404), the app automatically switches to the newest stable Gemini Flash model your key can use and shows which one it used |

## Push to GitHub

1. Create a `.gitignore` so secrets are never uploaded:

```bash
printf ".venv/\n__pycache__/\n.streamlit/secrets.toml\n.env\n" > .gitignore
```

2. Create an empty repository on [github.com/new](https://github.com/new) (e.g. `ats-resume-checker`; do not add a README there).
3. Run in your project folder:

```bash
git init
git add .
git commit -m "Initial commit: AI resume ATS checker"
git branch -M main
git remote add origin https://github.com/<your-username>/ats-resume-checker.git
git push -u origin main
```

If GitHub asks for a password, use a [personal access token](https://github.com/settings/tokens) instead of your account password.

## Deploy on Streamlit Community Cloud (free)

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **Create app** and choose your repo, branch `main`, and main file `app.py`.
3. Open **Advanced settings → Secrets** and paste:

```toml
GEMINI_API_KEY = "your-key"
```

4. Click **Deploy**. You will get a public URL like `https://<your-app>.streamlit.app`.

Updating the app later is just `git push`; Streamlit redeploys automatically.

## Notes and limitations

- Scanned/image-only PDFs have no extractable text; export a text-based PDF or DOCX instead.
- The "ATS score" is an AI estimate, not the output of a specific real ATS. Use it as guidance.
- The resume text is sent to Google's Gemini API. Do not use this with data you are not allowed to share.
- Free-tier API keys have rate limits; if you see a quota error, wait a minute and retry.

## Troubleshooting

- **`404 NOT_FOUND ... model is no longer available`**: Google retired that model name. This version handles it automatically; if you set `GEMINI_MODEL` or the sidebar field to an old name, clear it or enter a current Flash model name.
- **API key invalid**: create a new key at https://aistudio.google.com/apikey and make sure there are no spaces or quotes around it in the secrets file.
- **Old library error**: run `pip install -U google-genai`.
