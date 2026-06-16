import streamlit as st
import pickle
import re
import os
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# NLTK SETUP
# ─────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

download_nltk()
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: #F5F5F0;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}
.hero-sub {
    font-size: 1rem;
    color: #888;
    margin-bottom: 2rem;
    font-family: 'DM Sans', sans-serif;
}
.verdict-fake {
    background: linear-gradient(135deg, #3D0B0B, #5C1010);
    border: 1px solid #A32D2D;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
    color: #FFB3B3;
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: 0.1em;
}
.verdict-real {
    background: linear-gradient(135deg, #0B1F0D, #0F2E12);
    border: 1px solid #3B6D11;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
    color: #B3FFBE;
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: 0.1em;
}
.metric-card {
    background: #1A1A1A;
    border: 1px solid #2A2A2A;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-label {
    font-size: 0.75rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #F5F5F0;
}
.keyword-pill {
    display: inline-block;
    background: #1E1E1E;
    border: 1px solid #333;
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.8rem;
    color: #AAA;
    margin: 3px;
}
.info-box {
    background: #141414;
    border-left: 3px solid #444;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.2rem;
    color: #888;
    font-size: 0.85rem;
    margin-top: 1rem;
}
.model-badge-bert {
    display: inline-block;
    background: #1A2744;
    border: 1px solid #2E4F8A;
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.8rem;
    color: #7BA7E8;
    margin-bottom: 0.5rem;
}
.model-badge-tfidf {
    display: inline-block;
    background: #1A2A1A;
    border: 1px solid #3B6D11;
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.8rem;
    color: #8AFF9A;
    margin-bottom: 0.5rem;
}
.stTextArea textarea {
    background-color: #141414 !important;
    color: #F5F5F0 !important;
    border: 1px solid #2A2A2A !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
.stButton > button {
    background: #F5F5F0 !important;
    color: #0D0D0D !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-size: 0.95rem !important;
}
div[data-testid="stSidebar"] {
    background-color: #0D0D0D !important;
    border-right: 1px solid #1A1A1A;
}
.sidebar-info {
    background: #141414;
    border: 1px solid #222;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.82rem;
    color: #777;
    line-height: 1.6;
}
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, #2A2A2A, transparent);
    margin: 1.5rem 0;
}
.compare-box {
    background: #141414;
    border: 1px solid #2A2A2A;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD BERT MODEL
# ─────────────────────────────────────────────
HF_MODEL_PATH = "dvybs/fake-news-detector-bert"
MAX_LENGTH    = 128

@st.cache_resource
def load_bert_model():
    try:
        tokenizer = BertTokenizer.from_pretrained(HF_MODEL_PATH)
        model     = BertForSequenceClassification.from_pretrained(HF_MODEL_PATH)
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model  = model.to(device)
        return model, tokenizer, device, "ok"
    except Exception as e:
        return None, None, None, str(e)


# ─────────────────────────────────────────────
# LOAD TFIDF MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_tfidf_model():
    try:
        with open("fake_news_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer, "ok"
    except Exception as e:
        return None, None, str(e)


# ─────────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────────
stemmer    = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [stemmer.stem(w) for w in text.split()
             if w not in stop_words and len(w) > 2]
    return " ".join(words)


# ─────────────────────────────────────────────
# PREDICTION FUNCTIONS
# ─────────────────────────────────────────────
def predict_with_bert(text, model, tokenizer, device):
    encoding = tokenizer(
        str(text),
        max_length     = MAX_LENGTH,
        padding        = 'max_length',
        truncation     = True,
        return_tensors = 'pt'
    )
    with torch.no_grad():
        outputs = model(
            input_ids      = encoding['input_ids'].to(device),
            attention_mask = encoding['attention_mask'].to(device)
        )
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    pred       = int(np.argmax(probs))
    verdict    = 'REAL' if pred == 1 else 'FAKE'
    fake_prob  = round(float(probs[0]) * 100, 1)
    real_prob  = round(float(probs[1]) * 100, 1)
    confidence = real_prob if pred == 1 else fake_prob

    return {
        "verdict":      verdict,
        "confidence":   confidence,
        "fake_prob":    fake_prob,
        "real_prob":    real_prob,
        "top_keywords": [],
        "model_used":   "BERT"
    }


def predict_with_tfidf(text, model, vectorizer):
    cleaned    = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    pred       = model.predict(vectorized)[0]
    verdict    = "REAL" if pred == 1 else "FAKE"

    fake_prob = real_prob = confidence = None
    if hasattr(model, "predict_proba"):
        proba     = model.predict_proba(vectorized)[0]
        fake_prob = round(proba[0] * 100, 1)
        real_prob = round(proba[1] * 100, 1)
        confidence = real_prob if pred == 1 else fake_prob

    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores  = vectorized.toarray()[0]
    top_indices   = np.argsort(tfidf_scores)[::-1][:12]
    top_keywords  = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]

    return {
        "verdict":      verdict,
        "confidence":   confidence,
        "fake_prob":    fake_prob,
        "real_prob":    real_prob,
        "top_keywords": top_keywords,
        "model_used":   "TF-IDF"
    }


# ─────────────────────────────────────────────
# LOAD BOTH MODELS AT STARTUP
# ─────────────────────────────────────────────
with st.spinner("Loading models..."):
    bert_model, bert_tokenizer, bert_device, bert_status = load_bert_model()
    tfidf_model, tfidf_vectorizer, tfidf_status = load_tfidf_model()

bert_available  = bert_status == "ok"
tfidf_available = tfidf_status == "ok"


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Fake News Detector")
    st.markdown("---")

    # ── MODEL SWITCHER ──
    st.markdown("**🤖 Select Model**")

    model_options = []
    if bert_available:
        model_options.append("BERT (Deep Learning)")
    if tfidf_available:
        model_options.append("TF-IDF (Machine Learning)")
    if bert_available and tfidf_available:
        model_options.append("Compare Both")

    if not model_options:
        st.error("No models available")
        selected_model = None
    else:
        selected_model = st.radio(
            "Model",
            model_options,
            label_visibility="collapsed"
        )

    # Model info cards
    if selected_model == "BERT (Deep Learning)":
        st.markdown("""
        <div class="sidebar-info">
        <b style="color:#7BA7E8;">BERT</b><br><br>
        110M parameter transformer model.<br>
        Understands full sentence context.<br>
        Trained on 6,000 article subset.<br><br>
        <b style="color:#AAA;">Accuracy:</b> ~99% on test set
        </div>
        """, unsafe_allow_html=True)
    elif selected_model == "TF-IDF (Machine Learning)":
        st.markdown("""
        <div class="sidebar-info">
        <b style="color:#8AFF9A;">TF-IDF</b><br><br>
        Classic ML with word frequency features.<br>
        Trained on full 44,000 articles.<br>
        Faster and more reliable on diverse news.<br><br>
        <b style="color:#AAA;">Accuracy:</b> ~95% on test set
        </div>
        """, unsafe_allow_html=True)
    elif selected_model == "Compare Both":
        st.markdown("""
        <div class="sidebar-info">
        <b style="color:#EF9F27;">Compare Mode</b><br><br>
        Runs both models simultaneously.<br>
        See how each model differs on the same article.<br><br>
        Great for understanding model behavior.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio("Navigation", ["Single Article", "Bulk Analysis", "About"], label_visibility="collapsed")

    st.markdown("---")
    # Status indicators
    st.markdown("**Model Status**")
    if bert_available:
        st.success("✅ BERT loaded")
    else:
        st.error("❌ BERT unavailable")
    if tfidf_available:
        st.success("✅ TF-IDF loaded")
    else:
        st.error("❌ TF-IDF unavailable")


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">Fake News<br>Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Paste a headline or article — get an instant credibility verdict. Switch models in the sidebar.</div>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

if not bert_available and not tfidf_available:
    st.error("No models available. Please check your setup.")
    st.stop()


# ─────────────────────────────────────────────
# RESULT RENDERER
# ─────────────────────────────────────────────
def render_result(result, label=""):
    verdict    = result["verdict"]
    confidence = result["confidence"]
    fake_prob  = result["fake_prob"]
    real_prob  = result["real_prob"]
    keywords   = result["top_keywords"]
    model_used = result["model_used"]

    if label:
        badge_class = "model-badge-bert" if model_used == "BERT" else "model-badge-tfidf"
        st.markdown(f'<div class="{badge_class}">🤖 {model_used}</div>', unsafe_allow_html=True)

    if verdict == "FAKE":
        st.markdown('<div class="verdict-fake">🚨 &nbsp; FAKE NEWS</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="verdict-real">✅ &nbsp; REAL NEWS</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        conf_display = f"{confidence:.1f}%" if confidence is not None else "N/A"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Confidence</div>
            <div class="metric-value">{conf_display}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        fp = f"{fake_prob:.1f}%" if fake_prob is not None else "N/A"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Fake probability</div>
            <div class="metric-value" style="color:#FF8A8A">{fp}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        rp = f"{real_prob:.1f}%" if real_prob is not None else "N/A"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Real probability</div>
            <div class="metric-value" style="color:#8AFF9A">{rp}</div>
        </div>""", unsafe_allow_html=True)

    if fake_prob is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 0.7))
        fig.patch.set_facecolor("#1A1A1A")
        ax.set_facecolor("#1A1A1A")
        ax.barh([""], [fake_prob], color="#E24B4A", height=0.5, label="Fake")
        ax.barh([""], [real_prob], left=[fake_prob], color="#639922", height=0.5, label="Real")
        ax.set_xlim(0, 100)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(loc="upper right", fontsize=7, framealpha=0,
                  labelcolor="#AAA", handlelength=1)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    if keywords:
        st.markdown("<br>**Top keywords detected:**", unsafe_allow_html=True)
        pills = " ".join([f'<span class="keyword-pill">{k}</span>' for k in keywords])
        st.markdown(pills, unsafe_allow_html=True)


# ═════════════════════════════════════════════
# PAGE 1: SINGLE ARTICLE
# ═════════════════════════════════════════════
if page == "Single Article":
    col_input, col_result = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.markdown("#### Paste your article")

        import random

        fake_examples = [
            "SHOCKING: Scientists prove that 5G towers are spreading a new virus. Government officials are hiding this truth from the public and mainstream media is complicit in the cover-up!",
            "BREAKING: NASA confirms alien spaceship has been spotted near Jupiter. World governments are on high alert and the United Nations has called an emergency secret meeting to discuss the findings.",
            "EXPOSED: The government has been secretly adding mind-control chemicals to tap water for decades. Whistleblower leaks documents proving the global conspiracy to control the population.",
            "MIRACLE CURE: Doctors confirm that drinking lemon water every morning completely cures cancer. Big Pharma has been hiding this simple remedy for years to protect their profits.",
            "URGENT: New study proves that mobile phones cause instant brain damage after just 10 minutes of use. Scientists urge everyone to throw away their phones immediately.",
            "CONFIRMED: Bill Gates has microchips inside COVID-19 vaccines to track every human being on the planet. Government insiders have leaked the proof that mainstream media refuses to show.",
            "DEEP STATE EXPOSED: Secret society of elite politicians has been running the world from underground bunkers. Anonymous source reveals shocking truth about world leaders.",
        ]

        real_examples = [
            "The Federal Reserve raised its benchmark interest rate by a quarter percentage point on Wednesday, the tenth increase since early last year, as officials try to cool the economy and bring inflation back to their 2% target.",
            "Apple Inc. reported record quarterly revenue of $89.5 billion on Thursday, driven by strong iPhone sales and continued growth in its services division, which includes the App Store, Apple Music, and iCloud.",
            "The United Nations Security Council met on Tuesday to discuss the ongoing humanitarian crisis, with member states calling for an immediate ceasefire and the delivery of aid to affected civilian populations.",
            "Scientists at Johns Hopkins University have published findings in the journal Nature showing that a newly developed mRNA vaccine demonstrated 94% efficacy in clinical trials involving 30,000 participants across multiple countries.",
            "Nigeria's central bank held its benchmark interest rate steady at 18.75% on Tuesday, citing the need to balance inflation control with economic growth, according to a statement from the monetary policy committee.",
            "The World Health Organization announced on Monday that global malaria cases declined by 12% in 2023 compared to the previous year, crediting expanded access to insecticide-treated bed nets and antimalarial drugs.",
            "Tesla reported better-than-expected earnings for the third quarter, with net income rising 17% year-on-year to $2.3 billion, as vehicle deliveries reached a new record of 435,000 units globally.",
        ]

        # ── Initialise session state keys ──
        if "article_text" not in st.session_state:
            st.session_state["article_text"] = ""
        if "fake_idx" not in st.session_state:
            st.session_state["fake_idx"] = 0
        if "real_idx" not in st.session_state:
            st.session_state["real_idx"] = 0

        ecol1, ecol2 = st.columns(2)
        with ecol1:
            if st.button("🔴 Load fake example"):
                # Pick a different example each time by cycling through the list
                idx = st.session_state["fake_idx"] % len(fake_examples)
                st.session_state["article_text"] = fake_examples[idx]
                st.session_state["fake_idx"] += 1
        with ecol2:
            if st.button("🟢 Load real example"):
                idx = st.session_state["real_idx"] % len(real_examples)
                st.session_state["article_text"] = real_examples[idx]
                st.session_state["real_idx"] += 1

        article = st.text_area(
            "News article",
            value       = st.session_state["article_text"],
            height      = 220,
            placeholder = "Paste a news headline or article excerpt here...",
            label_visibility = "collapsed",
            key         = "article_input"
        )

        analyze_btn = st.button("🔍 Analyze Article", use_container_width=True)

        st.markdown("""
        <div class="info-box">
        ⚠️ Always verify important news with trusted sources like Reuters, AP, or BBC.
        </div>
        """, unsafe_allow_html=True)

    with col_result:
        st.markdown("#### Result")

        if analyze_btn:
            if not article.strip():
                st.warning("Please enter some text first.")

            # ── SINGLE MODEL MODE ──
            elif selected_model in ["BERT (Deep Learning)", "TF-IDF (Machine Learning)"]:
                with st.spinner(f"Analyzing with {selected_model}..."):
                    if selected_model == "BERT (Deep Learning)":
                        result = predict_with_bert(article, bert_model, bert_tokenizer, bert_device)
                    else:
                        result = predict_with_tfidf(article, tfidf_model, tfidf_vectorizer)
                render_result(result, label=selected_model)

            # ── COMPARE BOTH MODE ──
            elif selected_model == "Compare Both":
                with st.spinner("Running both models..."):
                    bert_result  = predict_with_bert(article, bert_model, bert_tokenizer, bert_device)
                    tfidf_result = predict_with_tfidf(article, tfidf_model, tfidf_vectorizer)

                st.markdown("#### 🤖 BERT Result")
                render_result(bert_result, label="BERT")

                st.markdown("---")
                st.markdown("#### 📊 TF-IDF Result")
                render_result(tfidf_result, label="TF-IDF")

                # Agreement indicator
                st.markdown("---")
                if bert_result["verdict"] == tfidf_result["verdict"]:
                    st.success(f"✅ Both models agree — **{bert_result['verdict']}**")
                else:
                    st.warning(f"⚠️ Models disagree — BERT says **{bert_result['verdict']}**, TF-IDF says **{tfidf_result['verdict']}**")

        else:
            st.markdown("""
            <div style="background:#141414; border:1px dashed #2A2A2A; border-radius:12px;
                        padding:3rem 1.5rem; text-align:center; color:#444; margin-top:0.5rem;">
                <div style="font-size:2rem; margin-bottom:0.5rem;">🔍</div>
                <div style="font-family:'Syne',sans-serif; font-size:1rem;">
                    Your result will appear here
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═════════════════════════════════════════════
# PAGE 2: BULK ANALYSIS
# ═════════════════════════════════════════════
elif page == "Bulk Analysis":
    st.markdown("#### Analyze multiple headlines at once")
    st.caption("Enter one headline per line.")

    bulk_text = st.text_area(
        "Headlines",
        height      = 200,
        placeholder = "Headline 1\nHeadline 2\nHeadline 3\n...",
        label_visibility = "collapsed"
    )

    if st.button("🔍 Analyze All", use_container_width=False):
        lines = [l.strip() for l in bulk_text.strip().split("\n") if l.strip()]
        if not lines:
            st.warning("Please enter at least one headline.")
        else:
            with st.spinner(f"Analyzing {len(lines)} headlines..."):
                rows = []
                for line in lines:
                    if selected_model == "BERT (Deep Learning)" and bert_available:
                        r = predict_with_bert(line, bert_model, bert_tokenizer, bert_device)
                    elif selected_model == "TF-IDF (Machine Learning)" and tfidf_available:
                        r = predict_with_tfidf(line, tfidf_model, tfidf_vectorizer)
                    elif selected_model == "Compare Both":
                        r = predict_with_tfidf(line, tfidf_model, tfidf_vectorizer)
                    else:
                        r = predict_with_tfidf(line, tfidf_model, tfidf_vectorizer)

                    if r:
                        rows.append({
                            "Headline":   line[:80] + ("..." if len(line) > 80 else ""),
                            "Verdict":    r["verdict"],
                            "Confidence": f"{r['confidence']:.1f}%" if r["confidence"] else "N/A",
                            "Fake prob":  f"{r['fake_prob']:.1f}%" if r["fake_prob"] else "N/A",
                            "Real prob":  f"{r['real_prob']:.1f}%" if r["real_prob"] else "N/A",
                            "Model":      r["model_used"],
                        })

            results_df = pd.DataFrame(rows)
            fake_count = sum(1 for r in rows if r["Verdict"] == "FAKE")
            real_count = len(rows) - fake_count

            c1, c2, c3 = st.columns(3)
            c1.metric("Total analyzed", len(rows))
            c2.metric("🚨 Fake", fake_count)
            c3.metric("✅ Real", real_count)

            st.markdown("---")

            def style_verdict(val):
                if val == "FAKE":
                    return "background-color: #3D0B0B; color: #FFB3B3; font-weight: bold;"
                elif val == "REAL":
                    return "background-color: #0B1F0D; color: #B3FFBE; font-weight: bold;"
                return ""

            styled = results_df.style.map(style_verdict, subset=["Verdict"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

            if len(rows) > 1:
                fig2, ax2 = plt.subplots(figsize=(4, 4))
                fig2.patch.set_facecolor("#1A1A1A")
                ax2.set_facecolor("#1A1A1A")
                ax2.pie(
                    [fake_count, real_count],
                    labels     = ["Fake", "Real"],
                    colors     = ["#E24B4A", "#639922"],
                    autopct    = "%1.0f%%",
                    startangle = 90,
                    textprops  = {"color": "#F5F5F0", "fontsize": 12}
                )
                ax2.set_title("Breakdown", color="#F5F5F0", fontsize=13)
                st.pyplot(fig2, use_container_width=False)
                plt.close(fig2)


# ═════════════════════════════════════════════
# PAGE 3: ABOUT
# ═════════════════════════════════════════════
elif page == "About":
    st.markdown("#### About this project")
    st.markdown("""
    This **Fake News Detector** was built as a final year machine learning project using Python.

    **Dataset**
    [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
    — ~44,000 articles labeled as Fake or Real.

    **Models Available**
    - **TF-IDF + ML** — trained on full 44,000 articles, fast and reliable
    - **BERT** — 110M parameter transformer, fine-tuned on 6,000 articles, deep contextual understanding
    - **Compare Mode** — run both models simultaneously and see where they agree or disagree

    **Key Finding**
    Training data size significantly impacts model performance regardless of model complexity.
    The TF-IDF model trained on more data often outperforms BERT trained on a smaller subset.

    **Tech Stack**
    Python · BERT · scikit-learn · NLTK · Streamlit · Hugging Face · GitHub

    **Limitations**
    - Trained primarily on English language news
    - Does not browse the internet or fact-check in real time
    - Always verify important claims with authoritative sources
    """)

    st.markdown("---")
    st.markdown("**Model Comparison**")
    comparison = pd.DataFrame({
        "Feature":   ["Training data", "Accuracy", "Speed", "Context understanding", "Global news"],
        "TF-IDF":    ["44,000 articles", "~95%", "⚡ Fast", "❌ Limited", "⚠️ Partial"],
        "BERT":      ["6,000 articles", "~99%", "🐢 Slower", "✅ Full", "⚠️ Partial"],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**Project Roadmap**")
    roadmap = pd.DataFrame({
        "Phase": ["Phase 1", "Phase 2", "Phase 3", "Phase 4"],
        "Task": [
            "Train ML model in Google Colab",
            "Build Streamlit web app",
            "Upgrade to BERT transformer model",
            "Deploy app online with Hugging Face + Streamlit Cloud"
        ],
        "Status": ["✅ Done", "✅ Done", "✅ Done", "✅ Done"]
    })
    st.dataframe(roadmap, use_container_width=True, hide_index=True)
