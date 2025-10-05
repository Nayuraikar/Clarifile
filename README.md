# Clarifile – Cognitive AI for File Management

Clarifile is an **AI-powered file assistant** that helps you **summarize, search, organize, and declutter** documents across formats such as PDF, DOCX, images, and audio.  
It integrates seamlessly with **Google Drive through a Chrome Extension** and offers a clean **React + TypeScript dashboard** for intelligent document management.  
Built with a **privacy-first architecture**, all processing can run **locally on your device**, ensuring data ownership and compliance.

---

## System Requirements

- **Python:** 3.12.6  
- **Node.js:** v18.20.5  
- **npm:** 9.x or later  
- **Operating System:** Windows / macOS 
- **Browser**: Chrome

---

## Key Features

- **AI Assistant:** Summarize and query multiple documents using natural language.  
- **Hybrid Search:** Combines semantic and keyword search across text, OCR-extracted data, and audio transcriptions.  
- **Smart Categorization:** Automatically organizes files into logical groups such as “patient reports”, “contracts”, or “lecture notes”, and improves with user feedback.  
- **Duplicate Detection:** Detects exact, near, and semantic duplicates with a side-by-side visual review and undo functionality.  
- **Google Drive Integration:** Syncs organization and categorization back to Drive with one click through the Chrome Extension.  
- **Privacy-First:** Uses OAuth2 for secure authentication and on-device processing to maintain data privacy.  
- **Stakeholder Impact:**  
  - **Medical professionals** can instantly retrieve and organize patient data and scans.  
  - **Businesses** can group client contracts, invoices, or identity files in seconds.  
  - **Students and researchers** can summarize papers, cluster notes, and clear duplicates effortlessly.  

---

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Nayuraikar/hackucchino_RVCE_4_NayanaJagadishRaikar
cd hackucchino_RVCE_4_NayanaJagadishRaikar
````

---

### 2. Create and Activate Virtual Environment

```bash
py -3.12 -m venv .venv
```

Activate the environment:

* **macOS/Linux:**

  ```bash
  source .venv/bin/activate
  ```
* **Windows (PowerShell):**

  ```powershell
  .\.venv\Scripts\Activate
  ```

---

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Install Frontend Dependencies

```bash
cd ui
npm install

```

---

## Google OAuth 2.0 Setup for Clarifile

Clarifile uses **Google OAuth 2.0** for secure Google Drive access. Follow the setup process below.

To streamline your testing process, we’ve provided a preconfigured test account that already has most integrations and credentials set up.

**Test Gmail (Recommended):** clarifiletester@gmail.com
Note: This account is pre-linked with Google Drive and Gemini API access for faster setup.

Using this test account is highly recommended for quick evaluation — it avoids additional OAuth configuration steps and helps conserve Gemini API tokens, which have limited usage under our current setup.

**Refer to the setup guide:**  [Setup Guide]( https://drive.google.com/file/d/13FWE3T-Qzh8-_KpBorFleiJqmPsdE39O/view?usp=sharing)

---

### 1. Create a New Project in Google Cloud Console

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click the project dropdown → **New Project**
3. Name it (e.g., “Clarifile”) → **Create**

---

### 2. Enable Required APIs

Navigate to **APIs & Services → Library**, then enable:

* **Google Drive API**

---

### 3. Configure OAuth Consent Screen

1. Go to **APIs & Services → OAuth consent screen**
2. Select **External** → **Create**
3. Enter:

   * App name: `Clarifile`
   * Support email: clarifiletester@gmail.com
4. Save and continue.

---

### 4. Create OAuth 2.0 Credentials

1. Go to **APIs & Services → Credentials**
2. Click **Create Credentials → OAuth client ID**
3. Choose **Web application**
4. Add authorized redirect URIs:

   * `https://<your-extension-id>.chromiumapp.org/`
5. Click **Create** to generate: (copy and save it for future use)

   * **Client ID**
   * **Client Secret** 

---

### 5. Update Gateway Configuration

In `gateway/index.js`, update: (line 24-26)

```js
const CLIENT_ID = 'xxxx-your-client-id-xxxx.apps.googleusercontent.com';
const CLIENT_SECRET = 'xxxx-your-client-secret-xxxx';
const REDIRECT_URI = 'https://xxxx-your-redirect-url-xxxx.chromiumapp.org/';
```

This securely connects the backend to Google Drive OAuth.

---

### 6. Update Chrome Extension Configuration

In your extension folder:

**manifest.json** (line-33)

```json
"oauth2": {
  "client_id": "xxxx-your-client-id-xxxx.apps.googleusercontent.com",
  "scopes": ["https://www.googleapis.com/auth/drive"]
}
```

**background.js** (line-1)

```js
const CLIENT_ID = "xxxx-your-client-id-xxxx.apps.googleusercontent.com";
```

### 7 Add test Gemini API Keys
1. Copy keys in the submission link
2. Paste the keys in services/parser/gemini_keys.txt
---

### 8. Testing OAuth

1. Start the backend and frontend servers.
2. Launch the Chrome extension and sign in with Google.
3. If you see “App not verified,” click **Advanced → Go to Clarifile**.

---

## Running the Application

### Backend (FastAPI + Node Gateway)

Open two terminals:

```bash
# Terminal 1 – FastAPI Parser Service
python -m uvicorn services.parser.app:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 – Node Gateway
cd gateway
npm install express cors body-parser axios
node index.js
```

---

### Frontend (React + TypeScript)

```bash
cd ui
npm run dev
```

Access Clarifile at:
**[http://127.0.0.1:4000]**

---

## Chrome Extension Setup

1. Open Chrome → navigate to `chrome://extensions`
2. Enable **Developer Mode**
3. Click **Load Unpacked**
4. Select the `Clarifile/extension/` folder
5. Confirm `manifest.json` and `background.js` contain your OAuth credentials

---

## Submissions

* **Demo Video:** [Watch Here](https://youtu.be/wrjyyFHm6ds)
* **Project Report:** [View Here](https://drive.google.com/file/d/1CMPkh3qeBoIWSGEILDMgXxtHojUZcwCZ/view?usp=sharing)
* **OAuth and Extension Setup Guide (PPT):** [View Here]( https://drive.google.com/file/d/13FWE3T-Qzh8-_KpBorFleiJqmPsdE39O/view?usp=sharing)
* **Troubleshooting Guide:** [View Here](https://drive.google.com/file/d/13cqUu5V8M6VSk1v-ZjaF6mQHOyvwwheW/view?usp=sharing)
* **Gemini API KEYS:** [Access Here](https://drive.google.com/file/d/11K0x7SGosW0UiJaMXVXxkbXeQrqlWKlL/view?usp=sharing)
* **Sample File Database:** [Access Here](https://drive.google.com/drive/folders/13CtI1eb4ihWnHQqogMe7TiHfvFKU2-ap?usp=sharing)
* **Compiled Submission** [Access Here](https://drive.google.com/drive/folders/1nyuIU0aZ8OaZZaTMnsFZFxdYAE6VrkZU?usp=sharing)
---

## Troubleshooting

If issues occur:

* Ensure redirect URIs match Google Cloud credentials
* Verify both backend services are running
* Reauthenticate through the Chrome Extension if sync fails
* Full documentation: [Troubleshooting Clarifile](https://your-link-to-troubleshooting-doc.com)
* Contact: nayanajagadishraikar.cs23@rvce.edu.in 
---

## Business and Impact

Clarifile directly transforms how individuals and organizations manage digital content. It reduces wasted time, enhances productivity, and ensures privacy through on-device intelligence.

---

### For Medical Professionals

* Locate and organize patient records, scans, and prescriptions across file formats.
* Categorize files automatically by patient name or diagnosis.
* Eliminate redundant copies while ensuring compliance.
* Keep all sensitive data processed locally for privacy and security.

**Impact:** Clarifile gives doctors and hospitals instant access to accurate patient histories, improving response times and data reliability.

---

### For Businesses and Enterprises

* Group all client contracts, invoices, or identity documents within seconds.
* Use advanced semantic search to locate information like “pending invoices” or “expiring contracts.”
* Move, delete, or organize files directly from the search panel.
* Maintain version control and document traceability effortlessly.

**Impact:** Clarifile streamlines administrative workflows, saves hours of manual labor, and ensures compliance with data handling standards.

---

### For Students and Researchers

* Summarize multiple research papers into key insights and references.
* Extract study flashcards, highlight core concepts, or build timelines from notes.
* Automatically categorize materials by subject or topic.
* Identify and delete redundant files, keeping your workspace light and efficient.

**Impact:** Clarifile transforms how students learn and how researchers organize data, allowing for smarter study and faster discovery.

---

### Scalability and Future Potential

* **Modular Architecture:** Built using FastAPI, Node.js, and React, easily deployable across personal or enterprise environments.
* **Extensible Ecosystem:** Designed for integration with other storage systems such as OneDrive or Dropbox.
* **Privacy-Preserving AI:** Uses local embeddings and OCR for compliance in healthcare, legal, and financial domains.

---

### Broader Vision

Clarifile’s mission is to make digital organization intelligent and effortless.
It helps users **focus on content, not file names**, and brings clarity to every digital workspace.
Whether you’re a student, a professional, or a business, Clarifile ensures your documents are always **organized, accessible, and secure**.

---

## Tech Stack Overview

| Layer       | Technology                           | Description                                                 |
| ----------- | ------------------------------------ | ----------------------------------------------------------- |
| Frontend    | React + TypeScript                   | Modern, fast, and modular dashboard                         |
| Backend     | Node.js Gateway + FastAPI Parser     | Handles routing, API integration, OCR, and NLP              |
| AI Models   | Gemini, Sentence Transformers, spaCy | Summarization, intent detection, semantic search            |
| Database    | FAISS + Vector Store                 | Enables efficient similarity search and document embeddings |
| Integration | Chrome Extension (Manifest V3)       | Google Drive sync and on-device automation                  |

---

### Clarifile — Bringing AI-Driven Order to the Chaos