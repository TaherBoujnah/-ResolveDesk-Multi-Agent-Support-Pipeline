# ⚡ ResolveDesk | Multi-Agent Support Pipeline

**ResolveDesk** is an enterprise-grade, autonomous customer support gatekeeper built on **Databricks Serverless**. It utilizes a multi-agent MLOps pipeline to intercept, route, and safely resolve customer support traffic. 

By combining sub-millisecond machine learning triage with policy-grounded Vector RAG and an active LLM compliance guardrail, the system significantly reduces GPU compute costs while guaranteeing zero financial leakage from AI hallucinations.

---

## 🧠 System Architecture: The 4-Stage Defense Pipeline

This application evaluates customer tickets through a strict, multi-stage verification process before any response reaches the user:

1. **ML Triage Router (Compute Optimization):** A sub-millisecond Random Forest classifier intercepts the incoming text. Non-urgent traffic (like 5-star positive reviews) is instantly acknowledged and short-circuited, bypassing the LLM entirely to save 100% on GPU compute costs.
2. **Vector RAG (Policy Grounding):** For actual issues, the ticket is vectorized using Databricks' **BGE-Large-En** model. The system queries an embedded corporate policy PDF to retrieve the exact legal or procedural clauses tied to the complaint.
3. **Generative Draft (Llama 3.3 70B):** An expert support agent prompt is sent to a serverless **Llama 3.3 70B Instruct** endpoint, strictly grounding its drafted resolution in the retrieved context.
4. **Zero-Temperature Guardrail (Factual Audit):** Before dispatch, a secondary, locked-down (`0.0 Temp`) LLM audits the draft against the original policy. If the AI attempts to invent unauthorized discounts or violate commercial rules, the output is instantly blocked and escalated to a human.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit (Custom Obsidian / Deep Space Radial UI)
* **Hosting:** Databricks Apps (Fully Managed Serverless Container)
* **Models / Endpoints (Databricks Foundation Model APIs):**
  * `databricks-meta-llama-3-3-70b-instruct` (Generative Drafts & Auditing)
  * `databricks-bge-large-en` (Vector Embeddings)
  * `support-ticket-api` (Custom Random Forest MLflow Endpoint)
* **Core Libraries:** `numpy`, `pypdf`, `scikit-learn`, `mlflow`

---

## 🚀 Deployment Instructions (Databricks Apps)

Because this application relies on Databricks serverless endpoints, it is designed to be hosted directly within a Databricks Workspace via **Databricks Apps**.

### 1. Repository Setup
Ensure your Git repository contains the following files:
* `app.py` (The main Streamlit application)
* `requirements.txt` (Dependencies)
* `app.yaml` (Databricks App configuration)
* `Primion_Return_Policy.pdf` (Default fallback policy manual)

**`requirements.txt`:**
```text
streamlit==1.32.0
numpy
pypdf
scikit-learn
mlflow
