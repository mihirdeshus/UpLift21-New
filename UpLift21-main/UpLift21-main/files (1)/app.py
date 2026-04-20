"""
app.py
UpLift21 — AI-Assisted Prenatal Screening System
Three-page Streamlit application: Assessment | Research | About
"""

import io
import warnings
import os
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="UpLift21",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load CSS ──────────────────────────────────────────────────────────────────
css_path = Path("styles.css")
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    required = ["models/best_model.pkl", "models/scaler.pkl", "models/model_name.pkl"]
    missing  = [p for p in required if not Path(p).exists()]
    if missing:
        return None, None, None, missing
    model      = joblib.load("models/best_model.pkl")
    scaler     = joblib.load("models/scaler.pkl")
    model_name = joblib.load("models/model_name.pkl")
    return model, scaler, model_name, []

model, scaler, model_name, missing_files = load_artifacts()

# ── Helpers ───────────────────────────────────────────────────────────────────
CLASS_LABELS  = ["Low", "Moderate", "High"]
BADGE_CLASSES = ["badge-low", "badge-mod", "badge-high"]
GAUGE_COLORS  = ["#00C6A7", "#F5A623", "#FF4D6A"]
BAR_COLORS    = ["#00C6A7", "#F5A623", "#FF4D6A"]

RECOMMENDATIONS = {
    0: ("Routine antenatal care is recommended. Screening markers are within "
        "expected ranges for this gestational age. Continue standard anomaly "
        "surveillance protocol. No immediate diagnostic escalation indicated "
        "based on current data."),
    1: ("Risk indicators fall in the intermediate zone. Review of first-trimester "
        "combined screening data by a specialist is advised. Consider repeat "
        "biochemical markers, extended NT assessment, or referral for non-invasive "
        "prenatal testing (NIPT). Clinical judgement should guide next steps."),
    2: ("Risk probability exceeds the optimised clinical threshold. Referral for "
        "confirmatory diagnostic testing is advised — NIPT as a first-line "
        "non-invasive step, or amniocentesis if NIPT is unavailable. This result "
        "is decision-support only and must be reviewed by a qualified clinician "
        "before any clinical action is taken."),
}


def make_pdf(inputs: dict, probs: np.ndarray, pred_class: int) -> bytes:
    """Generate a clinical PDF report using reportlab."""
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib import colors

    buf = io.BytesIO()
    styles = getSampleStyleSheet()

    title_s = ParagraphStyle("T", parent=styles["Title"],
                              fontSize=20, leading=24,
                              textColor=colors.HexColor("#0D1218"),
                              spaceAfter=4)
    head_s  = ParagraphStyle("H", parent=styles["Heading2"],
                              fontSize=11, textColor=colors.HexColor("#0D1218"),
                              spaceBefore=10, spaceAfter=4)
    body_s  = ParagraphStyle("B", parent=styles["Normal"],
                              fontSize=9.5, leading=15,
                              textColor=colors.HexColor("#333333"))
    mono_s  = ParagraphStyle("M", parent=styles["Normal"],
                              fontName="Courier", fontSize=9,
                              textColor=colors.HexColor("#444444"), leading=14)
    disc_s  = ParagraphStyle("D", parent=styles["Normal"],
                              fontSize=8, leading=12, fontName="Helvetica-Oblique",
                              textColor=colors.HexColor("#888888"))

    RULE = HRFlowable(width="100%", thickness=0.6,
                      color=colors.HexColor("#DDDDDD"), spaceAfter=6, spaceBefore=6)

    label_color = colors.HexColor(
        {"Low": "#1a7f37", "Moderate": "#9a6700", "High": "#d1242f"}[CLASS_LABELS[pred_class]]
    )
    label_s = ParagraphStyle("L", parent=styles["Normal"],
                              fontSize=14, fontName="Helvetica-Bold",
                              textColor=label_color)

    doc   = SimpleDocTemplate(buf,
                               rightMargin=22*mm, leftMargin=22*mm,
                               topMargin=22*mm, bottomMargin=22*mm)
    story = [
        Paragraph("UpLift21 — Prenatal Risk Assessment Report", title_s),
        RULE,
        Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%d %B %Y, %H:%M')}", body_s),
        Paragraph(f"<b>Model:</b> {model_name}", body_s),
        Spacer(1, 6*mm),
        RULE,
        Paragraph("Input Parameters", head_s),
    ]

    param_rows = [["Parameter", "Value"]] + [
        [k, str(v)] for k, v in inputs.items()
    ]
    param_table = Table(param_rows, colWidths=[90*mm, 70*mm])
    param_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#F0F4F8")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.HexColor("#333333")),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F8FAFB")]),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#DEDEDE")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    story.append(param_table)
    story.append(Spacer(1, 6*mm))
    story.append(RULE)
    story.append(Paragraph("Risk Assessment Result", head_s))
    story.append(Paragraph(
        f"Classification: <b>{CLASS_LABELS[pred_class]} Risk</b>", label_s
    ))
    story.append(Spacer(1, 3*mm))

    prob_rows = [["Risk Class", "Probability"]] + [
        [CLASS_LABELS[i], f"{probs[i]*100:.2f}%"] for i in range(3)
    ]
    prob_table = Table(prob_rows, colWidths=[90*mm, 70*mm])
    prob_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#F0F4F8")),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F8FAFB")]),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#DEDEDE")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 5*mm))
    story.append(RULE)
    story.append(Paragraph("Clinical Recommendation", head_s))
    story.append(Paragraph(RECOMMENDATIONS[pred_class], body_s))
    story.append(Spacer(1, 8*mm))
    story.append(RULE)
    story.append(Paragraph(
        "DISCLAIMER: UpLift21 is a research prototype and AI decision-support tool. "
        "It does not constitute a clinical diagnosis. All findings must be reviewed "
        "and interpreted by a qualified healthcare professional before any clinical "
        "action is taken. This tool is not for use in fetal sex determination and "
        "is designed in compliance with the PCPNDT Act, India.",
        disc_s,
    ))
    doc.build(story)
    buf.seek(0)
    return buf.read()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-brand">UpLift21</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-tag">Prenatal triage system</div>', unsafe_allow_html=True)

    page = st.radio("", ["Assessment", "Research", "About"],
                    label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    if model_name:
        st.markdown(f"""
        <div class="sb-info">
            MODEL<br>
            <span class="sb-info-val">{model_name}</span><br><br>
            CLASSES<br>
            <span class="sb-info-val">Low / Moderate / High</span><br><br>
            DATASET<br>
            <span class="sb-info-val">Synthetic (n=6000)</span><br><br>
            VERSION<br>
            <span class="sb-info-val">2.0</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="sb-info" style="color:#FF4D6A !important;">
            Models not loaded.<br>Run train_model.py first.
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# ASSESSMENT PAGE
# ════════════════════════════════════════════════════════════════════════════════
if page == "Assessment":

    st.markdown("""
    <div class="page-header">
        <div class="page-eyebrow">UpLift21 — Clinical Tool</div>
        <div class="page-title">Prenatal Risk Assessment</div>
        <div class="page-sub">
            Enter first-trimester screening values. The model returns a three-class
            Down syndrome risk stratification using biochemical and biophysical markers.
            All results are decision-support only.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if missing_files:
        st.error(f"Required model files not found: {', '.join(missing_files)}. "
                 f"Run generate_dataset.py then train_model.py first.")
        st.stop()

    col_form, col_result = st.columns([1.05, 1], gap="large")

    # ── Input form ────────────────────────────────────────────────────────────
    with col_form:
        st.markdown('<div class="sec-label">Maternal Parameters</div>', unsafe_allow_html=True)

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            maternal_age = st.number_input("Maternal Age (years)", 18, 50, 30, step=1)
        with r1c2:
            nasal_raw = st.selectbox("Nasal Bone", ["Present", "Absent"])
        nasal_binary = 1 if nasal_raw == "Present" else 0

        st.markdown('<div class="sec-label" style="margin-top:16px;">Ultrasound Markers</div>',
                    unsafe_allow_html=True)

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            nt  = st.number_input("NT Thickness (mm)", 0.5, 6.0, 1.8, step=0.1, format="%.1f")
        with r2c2:
            crl = st.number_input("CRL (mm)", 45.0, 90.0, 62.0, step=1.0, format="%.1f")

        fhr = st.number_input("Fetal Heart Rate (bpm)", 110.0, 180.0, 150.0, step=1.0, format="%.0f")

        st.markdown('<div class="sec-label" style="margin-top:16px;">Biochemical Markers</div>',
                    unsafe_allow_html=True)

        r3c1, r3c2 = st.columns(2)
        with r3c1:
            beta_hcg = st.number_input("Beta-hCG MoM", 0.1, 5.0, 1.0, step=0.05, format="%.2f")
        with r3c2:
            pappa    = st.number_input("PAPP-A MoM",   0.1, 3.0, 1.0, step=0.05, format="%.2f")

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("Run Assessment", use_container_width=True)

    # ── Results panel ─────────────────────────────────────────────────────────
    with col_result:
        if run_btn:
            X_raw    = np.array([[maternal_age, nt, crl, beta_hcg, pappa, fhr, nasal_binary]])
            X_input  = X_raw  # tree models: no scaling
            probs    = model.predict_proba(X_input)[0]
            pred_cls = int(np.argmax(probs))
            dominant = float(probs[pred_cls]) * 100

            color = GAUGE_COLORS[pred_cls]

            # Large probability display
            st.markdown(f"""
            <div class="card" style="text-align:center;padding:30px 24px;">
                <div class="prob-eyebrow">Dominant Risk Probability</div>
                <div class="prob-number" style="color:{color};">{dominant:.1f}%</div>
                <div style="margin-top:14px;">
                    <span class="badge {BADGE_CLASSES[pred_cls]}">{CLASS_LABELS[pred_cls]} Risk</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Plotly gauge
            try:
                import plotly.graph_objects as go
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=dominant,
                    number={"suffix": "%",
                            "font": {"color": color, "size": 30, "family": "Syne"}},
                    gauge={
                        "axis":        {"range": [0, 100],
                                        "tickfont": {"color": "#2A4A6A", "size": 9},
                                        "tickcolor": "#1A2332"},
                        "bgcolor":     "#0D1218",
                        "bordercolor": "#1A2332",
                        "steps": [
                            {"range": [0,   33],  "color": "rgba(0,198,167,0.07)"},
                            {"range": [33,  66],  "color": "rgba(245,166,35,0.07)"},
                            {"range": [66, 100],  "color": "rgba(255,77,106,0.07)"},
                        ],
                        "bar": {"color": color, "thickness": 0.22},
                    },
                ))
                fig.update_layout(
                    paper_bgcolor="#080C12",
                    plot_bgcolor="#080C12",
                    height=210,
                    margin=dict(l=20, r=20, t=10, b=10),
                    font=dict(family="DM Mono", color="#3A5A7A"),
                )
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})
            except ImportError:
                pass

            # Probability breakdown
            st.markdown('<div class="card"><div class="sec-label">Class Probabilities</div>',
                        unsafe_allow_html=True)
            for i, (lbl, p, bc) in enumerate(zip(CLASS_LABELS, probs, BAR_COLORS)):
                pct = float(p) * 100
                st.markdown(f"""
                <div class="prob-row">
                    <div class="prob-row-label" style="color:{bc};">{lbl}</div>
                    <div class="prob-bar-wrap">
                        <div class="prob-bar" style="width:{pct:.1f}%;background:{bc};"></div>
                    </div>
                    <div class="prob-pct">{pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Recommendation
            st.markdown(f"""
            <div class="rec-box">
                <div class="rec-label">Clinical Recommendation</div>
                <div class="rec-text">{RECOMMENDATIONS[pred_cls]}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── SHAP explanation ──────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="card"><div class="sec-label">Model Explanation (SHAP)</div>',
                        unsafe_allow_html=True)
            try:
                import shap
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                feat_names = [
                    "maternal_age", "nt_mm", "crl_mm",
                    "beta_hcg_mom", "pappa_mom", "fhr", "nasal_bone",
                ]
                n_feats = len(feat_names)

                explainer   = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_raw)

                # ── Robustly extract a plain Python float list ────────────────
                # shap_values shapes from sklearn tree models:
                #   list of (1, n_feats) arrays  — one per class (RandomForest)
                #   ndarray (1, n_feats, n_classes) — newer shap builds
                #   ndarray (1, n_feats)            — single-output fallback
                if isinstance(shap_values, list):
                    sv_flat = np.array(shap_values[pred_cls]).flatten().tolist()
                else:
                    arr = np.array(shap_values)
                    if arr.ndim == 3:          # (1, n_feats, n_classes)
                        sv_flat = arr[0, :, pred_cls].tolist()
                    elif arr.ndim == 2:        # (1, n_feats)
                        sv_flat = arr[0].tolist()
                    else:                      # (n_feats,)
                        sv_flat = arr.tolist()

                if len(sv_flat) != n_feats:
                    raise ValueError(
                        f"SHAP length {len(sv_flat)} != {n_feats} features"
                    )

                # Sort using Python built-ins — no numpy list indexing
                pairs   = sorted(zip(feat_names, sv_flat), key=lambda x: abs(x[1]))
                s_names = [p[0] for p in pairs]
                s_vals  = [p[1] for p in pairs]
                bar_c   = ["#FF4D6A" if v > 0 else "#00C6A7" for v in s_vals]

                fig_shap, ax_shap = plt.subplots(figsize=(7, 3.2),
                                                  facecolor="#0D1218")
                ax_shap.barh(s_names, s_vals, color=bar_c, height=0.55)
                ax_shap.axvline(0, color="#1A2332", linewidth=0.8)
                ax_shap.set_title(
                    f"SHAP — {CLASS_LABELS[pred_cls]} Risk class",
                    color="#C8D6E5", fontsize=10, fontweight="500",
                    fontfamily="DM Sans"
                )
                ax_shap.set_xlabel("SHAP value (impact on prediction)",
                                   color="#2A4A6A", fontsize=8)
                ax_shap.set_facecolor("#0D1218")
                ax_shap.tick_params(colors="#3A5A7A", labelsize=8)
                for sp in ax_shap.spines.values():
                    sp.set_edgecolor("#1A2332")
                ax_shap.grid(color="#1A2332", linewidth=0.4,
                             linestyle="--", alpha=0.6, axis="x")

                st.pyplot(fig_shap, use_container_width=True)
                plt.close(fig_shap)

                st.markdown("""
                <div style="font-family:'DM Mono',monospace;font-size:0.67rem;
                            color:#1A3A5A;letter-spacing:0.06em;margin-top:8px;">
                    Red bars increase risk probability. Teal bars decrease it.
                </div>
                """, unsafe_allow_html=True)

            except ImportError:
                st.markdown("""
                <div style="font-family:'DM Mono',monospace;font-size:0.75rem;color:#2A4A6A;">
                    Install shap to enable model explanations.
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"""
                <div style="font-family:'DM Mono',monospace;font-size:0.75rem;color:#2A4A6A;">
                    SHAP explanation unavailable: {str(e)[:120]}
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # PDF download
            st.markdown("<br>", unsafe_allow_html=True)
            try:
                param_dict = {
                    "Maternal Age":   f"{maternal_age} years",
                    "NT Thickness":   f"{nt:.1f} mm",
                    "CRL":            f"{crl:.1f} mm",
                    "Beta-hCG MoM":   f"{beta_hcg:.2f}",
                    "PAPP-A MoM":     f"{pappa:.2f}",
                    "Fetal HR":       f"{fhr:.0f} bpm",
                    "Nasal Bone":     nasal_raw,
                }
                pdf_bytes = make_pdf(param_dict, probs, pred_cls)
                fname = f"uplift21_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                st.download_button(
                    "Download Clinical Report (PDF)",
                    data=pdf_bytes,
                    file_name=fname,
                    mime="application/pdf",
                    use_container_width=True,
                )
            except ImportError:
                st.markdown("""
                <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#2A4A6A;">
                    Install reportlab to enable PDF export.
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="card" style="min-height:500px;display:flex;flex-direction:column;
                                     justify-content:center;align-items:center;text-align:center;">
                <div style="width:44px;height:44px;border:2px solid #1A2332;border-radius:50%;
                            margin-bottom:16px;"></div>
                <div style="font-family:'Syne',sans-serif;font-size:1.0rem;font-weight:600;
                            color:#1A2E40;margin-bottom:8px;">
                    Awaiting input
                </div>
                <div style="font-family:'DM Sans',sans-serif;font-size:0.82rem;
                            color:#1A3050;max-width:240px;line-height:1.6;">
                    Complete the form and run the assessment to see risk stratification results.
                </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# RESEARCH PAGE
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Research":

    st.markdown("""
    <div class="page-header">
        <div class="page-eyebrow">UpLift21 — Model Performance</div>
        <div class="page-title">Research & Methodology</div>
        <div class="page-sub">
            Training pipeline, evaluation metrics, feature analysis,
            and supporting literature.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metric strip
    try:
        results = joblib.load("models/all_results.pkl") if Path("models/all_results.pkl").exists() else None
        best_res = results[model_name] if results and model_name else None
        acc_str  = f"{best_res['acc']:.3f}"  if best_res else "—"
        auc_str  = f"{best_res['auc']:.3f}"  if best_res else "—"
        cv_str   = f"{best_res['cv_mean']:.3f}" if best_res else "—"
    except Exception:
        acc_str = auc_str = cv_str = "—"

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-tile">
            <div class="metric-val">{auc_str}</div>
            <div class="metric-lbl">ROC-AUC (OvR)</div>
        </div>
        <div class="metric-tile">
            <div class="metric-val">{acc_str}</div>
            <div class="metric-lbl">Accuracy</div>
        </div>
        <div class="metric-tile">
            <div class="metric-val">{cv_str}</div>
            <div class="metric-lbl">CV Mean (k=3)</div>
        </div>
        <div class="metric-tile">
            <div class="metric-val">6000</div>
            <div class="metric-lbl">Training Samples</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        # ROC Curves
        st.markdown('<div class="card"><div class="sec-label">ROC Curves (One-vs-Rest)</div>',
                    unsafe_allow_html=True)
        if Path("assets/roc_curve.png").exists():
            st.image("assets/roc_curve.png", use_container_width=True)
        else:
            st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:0.75rem;'
                        'color:#2A4A6A;">Run train_model.py to generate plots.</p>',
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Confusion matrix
        st.markdown('<div class="card"><div class="sec-label">Confusion Matrix</div>',
                    unsafe_allow_html=True)
        if Path("assets/confusion_matrix.png").exists():
            st.image("assets/confusion_matrix.png", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Literature
        st.markdown('<div class="card"><div class="sec-label">Supporting Literature</div>',
                    unsafe_allow_html=True)
        papers = [
            {
                "authors": "Goudarzi Z, Jafari M, Shirvani Z et al., 2025",
                "title":   "First and Second-Trimester Screening for Down Syndrome: "
                           "An Umbrella Review on Meta-Analyses",
                "journal": "Health Science Reports, 8(7): e70910",
                "finding": ("Combined ultrasound and serum screening across first and second "
                            "trimesters achieves sensitivity 0.935 and specificity 0.957. "
                            "Second trimester US+SS testing achieves sensitivity and "
                            "specificity both at 0.93 — the strongest combined performance."),
            },
            {
                "authors": "Yalcin E, Koc TK, Aslan S et al., 2025",
                "title":   "Artificial Intelligence in Prenatal Diagnosis: Down Syndrome "
                           "Risk Assessment with Gradient Boosting-Based Machine Learning",
                "journal": "Turkish Journal of Obstetrics and Gynecology, 22(2): 121-128",
                "finding": ("CatBoost achieves the highest accuracy at 95.31% on "
                            "first-trimester combined screening data (n=853). XGBoost and "
                            "LightGBM achieve 95.19% and 94.84% respectively. Tree-based "
                            "gradient boosting models outperform classical classifiers for "
                            "prenatal chromosomal risk classification."),
            },
        ]
        for p in papers:
            st.markdown(f"""
            <div class="lit-card">
                <div class="lit-authors">{p['authors']}</div>
                <div class="lit-title">{p['title']}</div>
                <div class="lit-journal">{p['journal']}</div>
                <div class="lit-finding">{p['finding']}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        # Feature importance
        st.markdown('<div class="card"><div class="sec-label">Feature Importance</div>',
                    unsafe_allow_html=True)
        if Path("assets/feature_importance.png").exists():
            st.image("assets/feature_importance.png", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Model comparison
        st.markdown('<div class="card"><div class="sec-label">Model Comparison</div>',
                    unsafe_allow_html=True)
        if Path("assets/model_comparison.png").exists():
            st.image("assets/model_comparison.png", use_container_width=True)
        if Path("assets/model_comparison.csv").exists():
            comp_df = pd.read_csv("assets/model_comparison.csv")
            st.markdown("<br>", unsafe_allow_html=True)
            for _, row in comp_df.iterrows():
                is_best = str(row["Model"]) == str(model_name)
                border  = "border-color:#00C6A7;" if is_best else ""
                badge   = ('<span style="font-family:\'DM Mono\',monospace;font-size:0.62rem;'
                           'color:#00C6A7;margin-left:8px;">best</span>') if is_best else ""
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                            padding:9px 14px;border:1px solid #1A2332;border-radius:6px;
                            margin-bottom:7px;{border}">
                    <span style="font-family:'DM Sans',sans-serif;font-size:0.84rem;
                                 color:#6A8AA8;">{row['Model']}{badge}</span>
                    <span style="font-family:'DM Mono',monospace;font-size:0.76rem;color:#3A6A8A;">
                        ACC {row['Accuracy']}</span>
                    <span style="font-family:'DM Mono',monospace;font-size:0.76rem;color:#00C6A7;">
                        AUC {row['ROC-AUC']}</span>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Input feature reference
        st.markdown('<div class="card"><div class="sec-label">Input Feature Reference</div>',
                    unsafe_allow_html=True)
        features_info = [
            ("nt_mm",        "Nuchal translucency (mm). Primary ultrasound DS marker. "
                             "Values above 2.5 mm indicate elevated chromosomal risk."),
            ("maternal_age", "Maternal age (years). DS incidence rises sharply above 35. "
                             "Age 40 carries ~1 in 100 risk vs ~1 in 1400 at age 25."),
            ("beta_hcg_mom", "Free beta-hCG in multiples of median. Elevated in trisomy 21. "
                             "Values >2.0 MoM are associated with increased DS risk."),
            ("pappa_mom",    "PAPP-A in multiples of median. Reduced in DS pregnancies. "
                             "Values <0.4 MoM are a significant risk indicator."),
            ("crl_mm",       "Crown-rump length (mm). Used for gestational age normalisation "
                             "of NT and MoM values. Measured at 11-13 weeks."),
            ("nasal_bone",   "Nasal bone presence at 11-13 weeks ultrasound. Absent in "
                             "approximately 60-70% of DS fetuses."),
            ("fhr",          "Fetal heart rate (bpm). Secondary biophysical marker. "
                             "Mild tachycardia has weak association with chromosomal anomalies."),
        ]
        rows_html = ""
        for fname, fdesc in features_info:
            rows_html += f"""
            <div style="display:flex;align-items:flex-start;gap:16px;padding:10px 0;
                        border-bottom:1px solid #0D1520;">
                <div style="font-family:'DM Mono',monospace;font-size:0.76rem;color:#4A7A9A;
                            width:110px;flex-shrink:0;padding-top:2px;">{fname}</div>
                <div style="font-family:'DM Sans',sans-serif;font-size:0.80rem;color:#2A4A6A;
                            line-height:1.55;">{fdesc}</div>
            </div>"""
        st.markdown(rows_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# ABOUT PAGE
# ════════════════════════════════════════════════════════════════════════════════
elif page == "About":

    st.markdown("""
    <div class="page-header">
        <div class="page-eyebrow">UpLift21 — Context</div>
        <div class="page-title">About This Project</div>
        <div class="page-sub">
            Problem framing, frugal science rationale, cost impact, and ethical framework.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="card card-accent">
            <div class="sec-label">The Problem</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:0.90rem;
                        color:#7A9AB8;line-height:1.72;">
                India records approximately
                <span style="color:#C8D6E5;font-weight:500;">30,000-35,000</span>
                Down syndrome births annually — among the highest absolute numbers globally
                due to population size and limited prenatal screening coverage.
                <br><br>
                Confirmatory prenatal diagnosis via amniocentesis costs
                <span style="color:#C8D6E5;font-weight:500;">₹15,000-35,000</span>
                per case in private facilities and is largely unavailable at primary
                health centre level. Existing first-trimester screening generates
                false positives that drive unnecessary invasive referrals, each
                carrying a 0.1-0.3% procedural miscarriage risk.
                <br><br>
                The gap is not scientific — it is one of access, triage quality,
                and cost of downstream testing.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="sec-label">Frugal Science Rationale</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:0.90rem;
                        color:#7A9AB8;line-height:1.72;">
                Developed under the
                <span style="color:#C8D6E5;">Frugal Science framework</span>
                (Manu Prakash Lab, Stanford University), UpLift21 operates
                entirely on existing clinical infrastructure.
                <br><br>
                No new ultrasound hardware. No cloud dependency. No specialist
                required on-site. A PHC-level clinician enters numerical values
                from an existing first-trimester ultrasound and blood report.
                The system returns a three-class risk stratification and routes
                only high-probability cases toward costly confirmatory testing.
                <br><br>
                The intervention is at the decision layer — the most scalable
                and lowest-cost point in the diagnostic pathway.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="sec-label">Ethical Framework</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:0.90rem;
                        color:#7A9AB8;line-height:1.72;">
                UpLift21 is a
                <span style="color:#C8D6E5;">decision-support tool</span>,
                not a diagnostic system. All outputs must be reviewed by a
                qualified clinician before any clinical action is taken.
                <br><br>
                The system does not perform and cannot be used for fetal sex
                determination. It is designed in compliance with the
                <span style="color:#C8D6E5;">PCPNDT Act, India</span>.
                <br><br>
                Deployment in any real clinical setting requires institutional
                ethics approval, prospective validation on regional cohort data,
                and informed patient consent protocols.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="sec-label">Cost Impact Model</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:0.88rem;
                        color:#5A8AAA;margin-bottom:16px;line-height:1.6;">
                Modelled on a district-level cohort of 10,000 pregnancies
                screened annually. Assumes average amniocentesis cost of ₹12,000.
            </div>
        """, unsafe_allow_html=True)

        cost_rows = [
            ("Without triage",    "~500 invasive tests",   "₹60,00,000", "#2A4A6A"),
            ("With UpLift21",     "~200 invasive tests",   "₹24,00,000", "#00C6A7"),
            ("Estimated saving",  "300 procedures avoided","₹36,00,000", "#F5A623"),
        ]
        for label_t, tests, cost, clr in cost_rows:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:10px 14px;border:1px solid #1A2332;border-radius:6px;
                        margin-bottom:8px;">
                <span style="font-family:'DM Sans',sans-serif;font-size:0.82rem;
                             color:#4A6A8A;width:140px;">{label_t}</span>
                <span style="font-family:'DM Mono',monospace;font-size:0.74rem;
                             color:#2A4A6A;">{tests}</span>
                <span style="font-family:'DM Mono',monospace;font-size:0.82rem;
                             font-weight:500;color:{clr};">{cost}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="sec-label">System Architecture</div>
        """, unsafe_allow_html=True)
        arch_steps = [
            ("Input",          "First-trimester numerical markers entered by PHC worker"),
            ("Preprocessing",  "RobustScaler applied where required; raw features for tree models"),
            ("Model",          "Gradient boosting classifier (best of RandomForest/XGBoost/LightGBM/CatBoost)"),
            ("Output",         "Three-class probability: Low / Moderate / High"),
            ("Triage",         "High risk → refer for NIPT or amniocentesis"),
            ("Report",         "PDF clinical report generated for patient record"),
        ]
        for step, desc in arch_steps:
            st.markdown(f"""
            <div style="display:flex;gap:14px;padding:9px 0;border-bottom:1px solid #0D1520;">
                <div style="font-family:'DM Mono',monospace;font-size:0.72rem;
                            color:#00C6A7;width:100px;flex-shrink:0;padding-top:1px;">{step}</div>
                <div style="font-family:'DM Sans',sans-serif;font-size:0.82rem;
                            color:#3A5A7A;line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="sec-label">Limitations</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:0.88rem;
                        color:#5A7A9A;line-height:1.72;">
                Training data is synthetic. Performance on real clinical cohorts
                will differ and requires prospective validation.
                <br><br>
                The model does not account for ethnicity-specific MoM medians,
                IVF pregnancies, twin gestations, or prior DS history — all of
                which are significant modifiers in clinical practice.
                <br><br>
                Operator variability in NT measurement is a known confounder
                not captured in numerical inputs alone.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        UpLift21 v2.0 &nbsp;|&nbsp; Frugal Science &nbsp;|&nbsp;
        Ashoka University &nbsp;|&nbsp; 2025 &nbsp;|&nbsp;
        Research prototype — not for clinical use without validation and ethics approval.
    </div>
    """, unsafe_allow_html=True)
