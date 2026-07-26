import streamlit as st
import numpy as np
import pypdf
from mlflow.deployments import get_deploy_client
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page Configuration & Header
st.set_page_config(page_title="AI Support & RAG Guardrails", page_icon="🛡️", layout="wide")
st.title("🛡️ Enterprise AI Support Engine with Dynamic RAG & Guardrails")
st.markdown("Upload any corporate PDF document to dynamically index policies, route tickets via ML triage, and enforce strict LLM compliance guardrails.")
st.divider()

# 2. Connect to Databricks Serverless Endpoints
@st.cache_resource
def get_client():
    return get_deploy_client("databricks")

client = get_client()
ML_ROUTER_ENDPOINT = "support-ticket-api"
EMBED_ENDPOINT = "databricks-bge-large-en"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

# 3. Helper: PDF Parser & Semantic Chunker
def parse_and_chunk_pdf(uploaded_file, chunk_size=300, overlap=50):
    reader = pypdf.PdfReader(uploaded_file)
    full_text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            full_text += extracted + "\n"
            
    words = full_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_str = " ".join(words[i:i + chunk_size])
        if len(chunk_str.strip()) > 20:
            chunks.append(chunk_str)
    return chunks

# 4. Helper: Active Output Guardrail
def run_compliance_guardrail(policy_context, ai_draft):
    guardrail_prompt = f"""
    You are an automated corporate compliance guardrail. Your job is to audit an AI-generated customer service response.
    
    OFFICIAL GOVERNING POLICY (From Uploaded PDF):
    "{policy_context}"
    
    AI-GENERATED DRAFT:
    "{ai_draft}"
    
    TASK: Verify if the AI draft is 100% faithful to the official policy. If it invents unauthorized discounts, promises timelines not in the policy, or contradicts the rules, it fails.
    Reply ONLY with either:
    "PASSED" (if fully compliant)
    "BLOCKED: [Brief reason why it violated policy]" (if it hallucinates or breaks rules)
    """
    res = client.predict(
        endpoint=LLM_ENDPOINT,
        inputs={"messages": [{"role": "user", "content": guardrail_prompt}], "max_tokens": 30, "temperature": 0.0}
    )
    return res["choices"][0]["message"]["content"].strip()

# 5. UI Layout: Two Tabs for SaaS Experience
tab_kb, tab_agent = st.tabs(["📚 1. Knowledge Base (PDF Upload)", "🤖 2. Live Ticket & Guardrail Audit"])

with tab_kb:
    st.subheader("Manage Corporate Policy Knowledge Base")
    st.write("Upload a PDF document (e.g., Return Guidelines, Shipping SLA, Warranty Manual) to index it into the active vector space.")
    
    col_upload, col_status = st.columns([1, 1])
    with col_upload:
        uploaded_pdf = st.file_uploader("Drop Corporate Policy PDF here:", type=["pdf"])
        use_default = st.checkbox("Or use built-in sample E-Commerce manual for testing", value=(uploaded_pdf is None))
        
    with col_status:
        if uploaded_pdf is not None and not use_default:
            with st.spinner("Extracting text and generating vector embeddings..."):
                chunks = parse_and_chunk_pdf(uploaded_pdf)
                embed_res = client.predict(endpoint=EMBED_ENDPOINT, inputs={"input": chunks})
                embeddings = np.array([e["embedding"] for e in embed_res["data"]])
                
                st.session_state["kb_chunks"] = chunks
                st.session_state["kb_embeddings"] = embeddings
                st.session_state["kb_name"] = uploaded_pdf.name
            st.success(f"✅ Successfully indexed `{uploaded_pdf.name}` into {len(chunks)} semantic vector chunks!")
            
        elif use_default:
            default_chunks = [
                "SECTION 1: DAMAGED ITEMS. If any item arrives torn, ripped, or damaged on the first day, the customer is entitled to an immediate 100% full refund or replacement. Do NOT require the customer to ship back the damaged item.",
                "SECTION 2: SHIPPING DELAYS. Standard shipping takes 3-5 business days. If delivery is delayed by over 2 business days beyond the estimate, support agents are authorized to apologize and issue an exact $15 store credit voucher.",
                "SECTION 3: STANDARD RETURNS. Undamaged clothing can be returned within 30 days of receipt strictly if items are unworn, unwashed, and have all original price tags attached."
            ]
            embed_res = client.predict(endpoint=EMBED_ENDPOINT, inputs={"input": default_chunks})
            st.session_state["kb_chunks"] = default_chunks
            st.session_state["kb_embeddings"] = np.array([e["embedding"] for e in embed_res["data"]])
            st.session_state["kb_name"] = "Default_E-Commerce_Manual.pdf"
            st.info(f"ℹ️ Active Knowledge Base: `{st.session_state['kb_name']}` ({len(default_chunks)} indexed rules)")
        else:
            st.warning("Please upload a PDF or check the box to use the default testing manual.")

with tab_agent:
    st.subheader("Live Ticket Triage & Resolution with Active Guardrails")
    
    if "kb_chunks" not in st.session_state:
        st.error("⚠️ No Knowledge Base loaded! Please go to Tab 1 and upload a PDF or enable the sample manual.")
    else:
        col_ticket, col_output = st.columns([1, 1.2])
        
        with col_ticket:
            st.markdown(f"**Active Knowledge Base:** `{st.session_state['kb_name']}`")
            sample_query = "My shipment has been stuck in transit for 6 days! Where is my order?! I want compensation right now."
            ticket_text = st.text_area("Paste Incoming Customer Support Ticket:", value=sample_query, height=150)
            run_btn = st.button("🚀 Process Ticket Through AI Pipeline", type="primary", use_container_width=True)
            
        with col_output:
            if run_btn and ticket_text:
                with st.spinner("Step 1: Running ML Triage Classifier..."):
                    try:
                        ml_res = client.predict(endpoint=ML_ROUTER_ENDPOINT, inputs={"dataframe_split": {"columns": ["Review_Text"], "data": [[ticket_text]]}})
                        rating = ml_res["predictions"][0]
                    except Exception:
                        rating = 1
                
                st.metric(label="🎯 MLOps Triage Router Prediction", value=f"{rating} / 5 Stars")
                
                if rating >= 4:
                    st.success("🟢 Positive Review — Short-Circuit Activated! Bypassing RAG & LLM layers to conserve GPU compute.")
                    st.write("---")
                    st.markdown(f"**Automated Reply:** Thank you so much for the {rating}-star review! We are thrilled you love your purchase.")
                else:
                    with st.spinner("Step 2: Retrieving PDF Policy & Drafting Resolution..."):
                        # Vector Retrieval against uploaded PDF chunks
                        q_embed = client.predict(endpoint=EMBED_ENDPOINT, inputs={"input": [ticket_text]})
                        q_vec = np.array([q_embed["data"][0]["embedding"]])
                        sims = cosine_similarity(q_vec, st.session_state["kb_embeddings"])[0]
                        best_idx = np.argmax(sims)
                        
                        retrieved_chunk = st.session_state["kb_chunks"][best_idx]
                        confidence = sims[best_idx]
                        
                        st.info(f"📚 **Retrieved Policy from `{st.session_state['kb_name']}`** (Similarity: `{confidence:.2%}`):\n\n> *\"{retrieved_chunk}\"*")
                        
                        # Generate Draft
                        prompt = f"You are an expert customer support agent. A customer left a complaint ({rating} Stars). Using ONLY this policy: '{retrieved_chunk}', draft a professional, empathetic resolution for: '{ticket_text}'"
                        llm_res = client.predict(endpoint=LLM_ENDPOINT, inputs={"messages": [{"role": "user", "content": prompt}], "max_tokens": 150, "temperature": 0.3})
                        draft = llm_res["choices"][0]["message"]["content"].strip()
                    
                    with st.spinner("Step 3: Running Active Compliance Guardrail..."):
                        verdict = run_compliance_guardrail(retrieved_chunk, draft)
                        
                    st.write("---")
                    st.markdown("### 🤖 RAG Resolution & Guardrail Status:")
                    
                    if "PASSED" in verdict.upper():
                        st.success("🛡️ **GUARDRAIL PASSED:** Output verified 100% compliant with uploaded PDF manual.")
                        st.markdown(f"**Final Customer Reply:**\n\n{draft}")
                    else:
                        st.error(f"🛡️ **GUARDRAIL BLOCKED OUTPUT:** {verdict}")
                        st.warning("⚠️ **System Override Triggered:** The LLM attempted to generate an unauthorized response. Falling back to safe human-handoff protocol.")
                        st.markdown("**Fallback Customer Reply:**\n\nWe sincerely apologize for the frustration you've experienced. Your ticket has been escalated directly to a Senior Support Manager who is reviewing your account and our official policies to resolve this immediately.")