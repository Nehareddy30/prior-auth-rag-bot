# 🏥 Prior Authorization RAG Bot
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://neha-payer-policy-bot.streamlit.app/)

🔴 **[Live Demo → neha-payer-policy-bot.streamlit.app](https://neha-payer-policy-bot.streamlit.app/)**

A **Retrieval-Augmented Generation (RAG)** pipeline that answers prior authorization eligibility questions by searching payer coverage policy PDFs — built without any paid API.

## 🔍 What This Project Does

Healthcare payers manage hundreds of coverage policy documents. When a provider submits a prior authorization request, staff must manually search these PDFs — a slow, error-prone process.

This bot automates that lookup:
1. **Ingests** coverage policy PDFs and extracts text
2. **Chunks** text into searchable 800-character segments
3. **Indexes** chunks using FAISS vector search + TF-IDF
4. **Retrieves** the most relevant policy sections for any question
5. **Logs** every query for audit trail (HIPAA compliance practice)

## 💬 Sample Questions & Results

| Question | Policy Found |
|---|---|
| Is lumbar spine MRI covered without conservative treatment? | MSK Policy Section 3.1 — CPT 72148 |
| What docs are needed for bariatric surgery? | BH Policy Section 4 — CPT 43644 |
| Are telehealth visits covered for behavioral health? | BH Policy Section 2 — CPT 90837 |

## 🗂️ Project Structure

## ⚙️ Setup

```bash
pip install pypdf faiss-cpu fpdf2 scikit-learn numpy
```

Then open `prior_auth_rag_bot.ipynb` in Jupyter and run cells top to bottom.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| PDF Parsing | pypdf |
| Vectorization | scikit-learn TF-IDF |
| Vector Search | FAISS |
| PDF Generation | fpdf2 |
| Audit Logging | Python JSON |

## 🏥 Healthcare Context

- References real CPT codes (72148, 73721, 43644)
- References real ICD-10 codes (M54.5, S83, E11)
- Audit logging mirrors HIPAA compliance requirements
- Synthetic policy PDFs mirror real payer document structure
- Designed for prior authorization decision support workflows

## 👩‍💻 Author

Neha Pannala — AI Engineer | Healthcare & Consulting Domain  
[LinkedIn](https://linkedin.com/in/neha-pannala) · [GitHub](https://github.com/Nehareddy30) · 
