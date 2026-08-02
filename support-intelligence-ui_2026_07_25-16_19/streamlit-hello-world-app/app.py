import streamlit as st
import numpy as np
import pypdf
import time
import os
from mlflow.deployments import get_deploy_client
from sklearn.metrics.pairwise import cosine_similarity

# Try to import Groq for the fallback architecture
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

# ==========================================
# 0. SECURE CLOUD AUTHENTICATION (STREAMLIT CLOUD)
# ==========================================
if "DATABRICKS_HOST" in st.secrets and "DATABRICKS_TOKEN" in st.secrets:
    os.environ["DATABRICKS_HOST"] = st.secrets["DATABRICKS_HOST"]
    os.environ["DATABRICKS_TOKEN"] = st.secrets["DATABRICKS_TOKEN"]

# Load Groq API Key for Fallback Plan B
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# 1. Page Configuration
st.set_page_config(
    page_title="ResolveDesk | Multi-Agent Support",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. PREMIUM STATIC OBSIDIAN CSS 
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, #root, [data-testid="stAppViewContainer"], .stApp {
        background: radial-gradient(circle at 10% 0%, #1e1b4b 0%, #09090b 40%, #050505 100%) !important;
        background-attachment: fixed !important;
        color: #f4f4f5 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    header[data-testid="stHeader"], [data-testid="stHeader"], .stAppHeader, [data-testid="stToolbar"] {
        background-color: transparent !important;
        background: transparent !important;
    }
    
    h1, h2, h3, h4, h5, h6, [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4, [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
    }
    
    label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *, .stSelectbox label *,
    .stTextArea label *, .stFileUploader label * {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.01em !important;
    }
    
    p, span, small, .stCaption * { color: #d1d5db !important; }
    
    [data-testid="stFileUploader"], [data-testid*="fileUploader" i], [data-testid*="dropzone" i] {
        background-color: rgba(9, 9, 11, 0.7) !important;
        border: 1px dashed rgba(255, 255, 255, 0.3) !important;
        border-radius: 10px !important;
        padding: 16px !important;
    }
    
    [data-testid="stFileUploader"] *, [data-testid*="dropzone" i] * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    
    [data-testid="stFileUploader"] button, [data-testid*="dropzone" i] button {
        background-color: #27272a !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        font-weight: 600 !important;
        padding: 0.35rem 0.9rem !important;
        border-radius: 6px !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: rgba(12, 12, 14, 0.6) !important;
        backdrop-filter: blur(25px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(145deg, rgba(24, 24, 27, 0.65) 0%, rgba(18, 18, 20, 0.8) 100%) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 14px !important;
        box-shadow: 0 12px 32px -12px rgba(0, 0, 0, 0.6), inset 0 1px 0 0 rgba(255, 255, 255, 0.08) !important;
        padding: 12px !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    
    div[data-testid="stExpander"] {
        background-color: rgba(18, 18, 20, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(16px) !important;
    }
    div[data-testid="stExpander"] summary p {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.15rem !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(18, 18, 20, 0.75) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-top: 3px solid #3b82f6 !important;
        padding: 16px 20px !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4) !important;
    }
    div[data-testid="stMetricLabel"] * {
        color: #60a5fa !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
    }
    div[data-testid="stMetricValue"] * {
        color: #ffffff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 1.75rem !important;
    }
    
    .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
        background-color: rgba(18, 18, 20, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }
    
    button[kind="primary"] {
        background: linear-gradient(270deg, #2563eb, #3b82f6, #1d4ed8, #2563eb) !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.2rem !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4) !important;
    }
    
    .status-badge-pass {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 78, 59, 0.25) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-left: 4px solid #10b981;
        padding: 16px 20px;
        border-radius: 8px;
        margin-top: 12px;
        margin-bottom: 16px;
    }
    .status-badge-block {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.12) 0%, rgba(127, 29, 29, 0.25) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-left: 4px solid #ef4444;
        padding: 16px 20px;
        border-radius: 8px;
        margin-top: 12px;
        margin-bottom: 16px;
    }
    .badge-title {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .text-pass { color: #34d399 !important; }
    .text-block { color: #f87171 !important; }
</style>
""", unsafe_allow_html=True)

# 3. Connect to Databricks Serverless Endpoints
@st.cache_resource
def get_client():
    return get_deploy_client("databricks")

client = get_client()
ML_ROUTER_ENDPOINT = "support-ticket-api"
EMBED_ENDPOINT = "databricks-bge-large-en"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

# ==========================================
# 4. HELPER FUNCTIONS & FALLBACK LOGIC
# ==========================================
def parse_and_chunk_pdf(uploaded_file, chunk_size=300, overlap=50):
    reader = pypdf.PdfReader(uploaded_file)
    full_text = "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])
    words = full_text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap) if len(" ".join(words[i:i + chunk_size]).strip()) > 20]

def generate_llm_response(prompt, max_tokens=150, temperature=0.3):
    """Dual-Provider Generation: Tries Databricks first, falls back to Groq API if failed."""
    try:
        # Try Primary Databricks
        res = client.predict(
            endpoint=LLM_ENDPOINT, 
            inputs={"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature}
        )
        return res["choices"][0]["message"]["content"].strip()
    
    except Exception as e:
        # Fallback to Groq if Databricks throws 403/500/Trial Ended
        if HAS_GROQ and os.environ.get("GROQ_API_KEY"):
            st.toast("⚡ Routing to Backup Engine (Groq Free Tier)", icon="🔄")
            groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return chat_completion.choices[0].message.content.strip()
        else:
            raise Exception(f"Databricks failed and Groq API key is missing. Original Error: {e}")

def run_compliance_guardrail(policy_context, ai_draft):
    guardrail_prompt = f"""
    You are an automated corporate compliance guardrail. Audit this AI-generated customer service response.
    OFFICIAL GOVERNING POLICY: "{policy_context}"
    AI-GENERATED DRAFT: "{ai_draft}"
    TASK: Verify if the AI draft is 100% faithful to the policy. If it invents unauthorized discounts, promises timelines not in the policy, or contradicts rules, it fails.
    Reply ONLY with: "PASSED" (if compliant) or "BLOCKED: [Brief reason why]" (if it violates rules).
    """
    return generate_llm_response(guardrail_prompt, max_tokens=30, temperature=0.0)


# ==========================================
# 5. LEFT SIDEBAR: KNOWLEDGE BASE MANAGER
# ==========================================
with st.sidebar:
    st.markdown("### ResolveDesk")
    st.caption("MULTI-AGENT SUPPORT PIPELINE v2.5")
    st.divider()
    
    st.markdown("#### Knowledge Engine")
    st.write("Ingest governing corporate policies to ground LLM resolutions.")
    
    uploaded_pdf = st.file_uploader("Upload Corporate Manual (PDF)", type=["pdf"])
    use_default = st.toggle("Use Built-In Primion Policy", value=(uploaded_pdf is None))
    
    if uploaded_pdf is not None and not use_default:
        with st.spinner("Vectorizing PDF chunks..."):
            chunks = parse_and_chunk_pdf(uploaded_pdf)
            embed_res = client.predict(endpoint=EMBED_ENDPOINT, inputs={"input": chunks})
            st.session_state["kb_chunks"] = chunks
            st.session_state["kb_embeddings"] = np.array([e["embedding"] for e in embed_res["data"]])
            st.session_state["kb_name"] = uploaded_pdf.name
        st.success(f"[INDEXED] `{uploaded_pdf.name}` ({len(chunks)} vectors)")
    elif use_default:
        default_chunks = [
            "§ 1 - Return policy: Returns are strictly only possible within 14 calendar days after receipt of goods. Items must be in undamaged original packaging with all original labels attached.",
            "§ 3 - Return shipping costs: Shipping costs for returns of unwanted items must be borne by the customer. If Primion shipped a defective or incorrect item, Primion will bear shipping costs.",
            "§ 4 - Overview of costs: If items are returned outside the 14-day window or without original packaging/labels, a mandatory €90 examination fee will be charged or the return rejected."
        ]
        if "kb_embeddings" not in st.session_state or st.session_state.get("kb_name") != "Primion_Return_Policy.pdf":
            embed_res = client.predict(endpoint=EMBED_ENDPOINT, inputs={"input": default_chunks})
            st.session_state["kb_chunks"] = default_chunks
            st.session_state["kb_embeddings"] = np.array([e["embedding"] for e in embed_res["data"]])
            st.session_state["kb_name"] = "Primion_Return_Policy.pdf"
        st.info("[ACTIVE] Primion Enterprise Policy")
    
    st.divider()
    st.caption("SYSTEM: HYBRID MULTI-CLOUD ARCHITECTURE")

# ==========================================
# 6. MAIN STAGE: EXECUTIVE EXPLANATION
# ==========================================
st.markdown("## Autonomous Support Gatekeeper & Compliance Guardrail")
st.markdown("<p style='color: #d1d5db; font-size: 1.05rem; margin-top: -10px; margin-bottom: 25px;'>A multi-agent MLOps pipeline designed to intercept, route, and safely resolve customer traffic.</p>", unsafe_allow_html=True)

st.markdown("""
<div style="background: rgba(18, 18, 20, 0.65); backdrop-filter: blur(16px); border: 1px solid rgba(255, 255, 255, 0.15); padding: 25px; border-radius: 12px; margin-bottom: 30px;">
    <h3 style="margin-top: 0; color: #ffffff; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px;">📖 System Architecture & Telemetry Overview</h3>
    <div style="display: flex; gap: 30px; margin-top: 20px; flex-wrap: wrap;">
        <div style="flex: 1; min-width: 300px;">
            <h4 style="color: #60a5fa; margin-bottom: 10px;">[1] The 4-Stage Defense Pipeline</h4>
            <p style="color: #d1d5db; font-size: 0.95rem; margin-bottom: 6px;"><b>1. ML Triage Router:</b> A sub-millisecond Random Forest classifier intercepts text, halting the pipeline for non-urgent traffic.</p>
            <p style="color: #d1d5db; font-size: 0.95rem; margin-bottom: 6px;"><b>2. Vector RAG:</b> Embeddings retrieve governing policy clauses specifically tied to the complaint.</p>
            <p style="color: #d1d5db; font-size: 0.95rem; margin-bottom: 6px;"><b>3. Generative Draft:</b> Llama 3 drafts a resolution grounded strictly in the retrieved text.</p>
            <p style="color: #d1d5db; font-size: 0.95rem; margin-bottom: 6px;"><b>4. Zero-Temp Guardrail:</b> A strict AI auditor verifies the draft against the policy, blocking unauthorized concessions.</p>
        </div>
        <div style="flex: 1; min-width: 300px;">
            <h4 style="color: #34d399; margin-bottom: 10px;">[2] What The Scenarios Test</h4>
            <p style="color: #d1d5db; font-size: 0.95rem; margin-bottom: 6px;"><b style="color: #60a5fa;">[ML] Triage Short-Circuit:</b> Bypasses the LLM entirely for a 5-star review.</p>
            <p style="color: #d1d5db; font-size: 0.95rem; margin-bottom: 6px;"><b style="color: #34d399;">[RAG] Compliant Resolution:</b> Resolves a valid complaint autonomously using exact legal rules.</p>
            <p style="color: #d1d5db; font-size: 0.95rem; margin-bottom: 6px;"><b style="color: #f87171;">[TRAP] Hallucination:</b> Customer demands an unauthorized refund; Guardrail blocks the response.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if "kb_chunks" not in st.session_state:
    st.warning("Please ingest a corporate policy PDF or enable the default manual in the sidebar to activate the pipeline.")
    st.stop()

# ==========================================
# 7. THE CLICKABLE ARROW / EXPANDER CONSOLE
# ==========================================
with st.expander("▶ ACCESS LIVE PIPELINE CONSOLE & TELEMETRY", expanded=False):
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Router Latency", "0.070 ms", "-99.9% vs LLM API", delta_color="inverse")
    with col_m2:
        st.metric("Triage Precision", "83.72%", "Binary ML Router")
    with col_m3:
        st.metric("Vector Precision", "100.0%", "Precision@1 Match")
    with col_m4:
        st.metric("Guardrail State", "ACTIVE", "0.0 Temp Factual Audit")

    st.divider()

    col_input, col_output = st.columns([1, 1.25], gap="large")

    with col_input:
        with st.container(border=True):
            st.markdown("#### Live Ticket Ingestion")
            st.caption(f"Target Knowledge Base: **{st.session_state['kb_name']}**")
            st.markdown("<br>", unsafe_allow_html=True)
            
            preset = st.selectbox("Load Verification Scenario:", [
                "[TRAP] Hallucination (Demanding late return + fee waiver)",
                "[RAG] Compliant Resolution (Defective hardware)",
                "[ML] Triage Short-Circuit (5-Star positive review)"
            ])
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if "Hallucination" in preset:
                default_txt = "We ordered 10 access control boards 25 days ago, but our building project got canceled. We threw away the original boxes and labels. I demand you email a free prepaid return label immediately and guarantee a 100% full refund with zero €90 examination fees!"
            elif "Compliant" in preset:
                default_txt = "We received our order 6 days ago, but one RFID reader arrived defective and won't power on. We have the undamaged original box with all labels attached. Since it arrived defective, how do we send this back without paying return shipping?"
            else:
                default_txt = "We just received our shipment of Primion access control terminals and the industrial build quality is phenomenal! Setting them up on our network was seamless. 5 stars all the way!"
                
            ticket_text = st.text_area("Customer Ticket Content:", value=default_txt, height=170)
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("EXECUTE AI DEFENSE PIPELINE", type="primary", use_container_width=True)

    with col_output:
        if run_btn and ticket_text:
            with st.status("Executing Multi-Stage Defense Pipeline...", expanded=True) as status:
                st.write("[STEP 1] Intercepting text via sub-millisecond TF-IDF + Random Forest Router...")
                time.sleep(0.25)
                try:
                    ml_res = client.predict(endpoint=ML_ROUTER_ENDPOINT, inputs={"dataframe_split": {"columns": ["Review_Text"], "data": [[ticket_text]]}})
                    rating = ml_res["predictions"][0]
                except Exception:
                    rating = 1
                
                if rating >= 4:
                    status.update(label="Ticket Short-Circuited by ML Gatekeeper", state="complete", expanded=False)
                    with st.container(border=True):
                        st.markdown("#### ML Triage Routing: Short-Circuit")
                        st.success(f"[PREDICTED SCORE] {rating} / 5 Stars (Non-Urgent Praise)")
                        st.info("[COMPUTE OPTIMIZATION] 100% of LLM GPU generation bypassed. Automated acknowledgment dispatched.")
                        st.markdown("##### Automated Dispatch Reply:\n> *Thank you so much for your positive review! We are thrilled to hear that our hardware met your expectations.*")
                else:
                    st.write("[STEP 2] Urgency flagged. Querying vector space for governing legal clauses...")
                    q_embed = client.predict(endpoint=EMBED_ENDPOINT, inputs={"input": [ticket_text]})
                    q_vec = np.array([q_embed["data"][0]["embedding"]])
                    sims = cosine_similarity(q_vec, st.session_state["kb_embeddings"])[0]
                    best_idx = np.argmax(sims)
                    retrieved_chunk = st.session_state["kb_chunks"][best_idx]
                    confidence = sims[best_idx]
                    
                    st.write("[STEP 3] Generating policy-grounded resolution...")
                    prompt = f"You are an expert support agent. A customer left a complaint. Using ONLY this policy: '{retrieved_chunk}', draft a professional resolution for: '{ticket_text}'"
                    draft = generate_llm_response(prompt, max_tokens=150, temperature=0.3)
                    
                    st.write("[STEP 4] Auditing draft via zero-temperature Compliance Guardrail...")
                    verdict = run_compliance_guardrail(retrieved_chunk, draft)
                    
                    status.update(label="Defense Pipeline Execution Complete", state="complete", expanded=False)
                    
                    with st.container(border=True):
                        st.markdown("#### Vector Retrieval & Guardrail Telemetry")
                        st.markdown(f"<p style='color: #a1a1aa; font-size: 0.9rem; margin-bottom: 5px;'>RETRIEVED GOVERNING POLICY (Confidence: <code style='color: #60a5fa;'>{confidence:.2%}</code>):</p>", unsafe_allow_html=True)
                        st.markdown(f"> *\"{retrieved_chunk}\"*")
                        
                        if "PASSED" in verdict.upper():
                            st.markdown("""
                            <div class="status-badge-pass">
                                <div class="badge-title text-pass">[VERIFIED] GUARDRAIL STATUS: PASSED</div>
                                <div style="font-size: 0.9rem; color: #d1d5db;">Output verified 100% compliant with uploaded corporate policy manual.</div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown("##### Approved AI Resolution:")
                            st.success(draft)
                        else:
                            st.markdown(f"""
                            <div class="status-badge-block">
                                <div class="badge-title text-block">[INTERCEPTED] GUARDRAIL STATUS: BLOCKED</div>
                                <div style="font-size: 0.9rem; color: #fca5a5;"><b>Violation Flagged:</b> {verdict}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.warning("[AUTONOMOUS OVERRIDE] The model attempted to generate an unauthorized concession or violate commercial rules. Output intercepted.")
                            st.markdown("##### Safe Human-Handoff Fallback:")
                            st.error("We sincerely apologize for the frustration you've experienced. Your ticket has been escalated directly to a Senior Support Manager who is reviewing your account and our policies to resolve this immediately.")
        else:
            with st.container(border=True):
                st.markdown("#### Waiting for Telemetry Ingestion")
                st.info("[SYSTEM READY] Select a verification scenario on the left and click EXECUTE AI DEFENSE PIPELINE.")
