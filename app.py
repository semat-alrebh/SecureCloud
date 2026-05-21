"""
app.py — SecureFlow NIDS Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
import os
import base64

st.set_page_config(page_title="SecureFlow", page_icon="", layout="wide")

def get_logo_base64():
    for path in ['logo.jpg', 'logo.png', 'photo_2026-05-21_15-57-50.jpg']:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return base64.b64encode(f.read()).decode()
    return None

logo_b64 = get_logo_base64()

BG       = "#F5EEF0"
CARD_BG  = "#FFFFFF"
TEXT     = "#0D1F3C"
TEXT_SUB = "#4A5568"
BLUE     = "#1B3A6B"
RED      = "#C0392B"
BORDER   = "#D8CDD0"

CHART_COLORS = ["#1B3A6B","#2E5FA3","#4A7FC1","#6B9FD4","#8FBFE8","#B3D4F0"]

def chart_layout():
    return dict(
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font=dict(color=TEXT, family="sans-serif"),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    )

st.markdown(f"""
<style>
    .stApp, [data-testid="stAppViewContainer"], section.main {{
        background-color: {BG} !important;
    }}
    * {{ color: {TEXT} !important; }}
    .stSelectbox > div > div, .stRadio > div {{
        background-color: {CARD_BG} !important;
        border: 1px solid {BORDER} !important;
    }}
    [data-baseweb="popover"], [data-baseweb="menu"] {{
        background-color: {CARD_BG} !important;
        color: {TEXT} !important;
    }}
    li[role="option"] {{
        background-color: {CARD_BG} !important;
        color: {TEXT} !important;
    }}
    li[role="option"]:hover {{
        background-color: #E8E0E3 !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: {TEXT} !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        background-color: {BLUE} !important;
    }}
    .metric-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }}
    .metric-value {{ font-size: 2rem; font-weight: 700; color: {BLUE} !important; }}
    .metric-label {{ font-size: 0.85rem; color: {TEXT_SUB} !important; margin-top: 4px; }}
    .attack-badge {{
        background: {RED}; color: white !important;
        padding: 10px 28px; border-radius: 20px;
        font-size: 1.3rem; font-weight: bold; display: inline-block;
    }}
    .normal-badge {{
        background: #16A34A; color: white !important;
        padding: 10px 28px; border-radius: 20px;
        font-size: 1.3rem; font-weight: bold; display: inline-block;
    }}
    .header-row {{ display: flex; align-items: center; gap: 16px; margin-bottom: 4px; }}
    .header-logo {{ width: 56px; height: 56px; object-fit: contain; }}
    .header-title {{ font-size: 2.2rem; font-weight: 800; color: {BLUE} !important; margin: 0; }}
    .header-sub {{ color: {TEXT_SUB} !important; font-size: 0.95rem; margin-left: 72px; }}
    hr {{ border-color: {BORDER}; }}
    /* File uploader */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploadDropzone"] {{
        background-color: {CARD_BG} !important;
        border: 2px dashed {BORDER} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stFileUploader"] *,
    [data-testid="stFileUploadDropzone"] * {{
        color: {TEXT} !important;
        background-color: transparent !important;
    }}
    [data-testid="stFileUploader"] button {{
        background-color: {BLUE} !important;
        color: white !important;
        border: none !important;
    }}
    [data-testid="stFileUploader"] button * {{
        color: white !important;
    }}
    /* Radio buttons - just color the dot, keep label readable */
    [data-testid="stRadio"] label {{ color: {TEXT} !important; font-weight: 500; }}
    [data-testid="stRadio"] label span {{ color: {TEXT} !important; }}
    div[data-testid="stMetricValue"] {{ color: {BLUE} !important; }}
    /* Number inputs */
    [data-testid="stNumberInput"] input,
    [data-baseweb="input"] input,
    input[type="number"] {{
        background-color: {CARD_BG} !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER} !important;
    }}
    [data-baseweb="input"] {{
        background-color: {CARD_BG} !important;
        border: 1px solid {BORDER} !important;
    }}
    /* Slider */
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
        background-color: {BLUE} !important;
        border-color: {BLUE} !important;
    }}
    [data-testid="stSlider"] div[data-baseweb="slider"] > div > div {{
        background-color: {BLUE} !important;
    }}
    /* Primary button */
    .stButton > button[kind="primary"],
    .stButton > button {{
        background-color: {BLUE} !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
    }}
    .stButton > button:hover {{
        background-color: #2E5FA3 !important;
        color: white !important;
    }}
</style>
""", unsafe_allow_html=True)

# Header
if logo_b64:
    st.markdown(f"""
    <div class="header-row">
        <img src="data:image/jpeg;base64,{logo_b64}" class="header-logo"/>
        <h1 class="header-title">SecureFlow</h1>
    </div>
    <div class="header-sub">ML-Based Network Intrusion Detection System</div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="header-row">
        <h1 class="header-title">SecureFlow</h1>
    </div>
    <div class="header-sub">ML-Based Network Intrusion Detection System</div>
    """, unsafe_allow_html=True)

st.divider()

NSLKDD_COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
]

DATASET_STATS = {
    'Dataset':       ['NSL-KDD','CIC-IDS2017','UNSW-NB15'],
    'Total Records': [148517, 2830743, 257673],
    'Normal':        [77054, 2273097, 93000],
    'Attack':        [71463, 557646, 164673],
    'Features':      [44, 80, 46],
}

CORR_DATA = {
    'NSL-KDD': {
        'same_srv_rate': -0.709, 'dst_host_srv_count': -0.693,
        'logged_in': -0.664, 'dst_host_srv_serror_rate': 0.594,
        'serror_rate': 0.588, 'srv_serror_rate': 0.587,
        'count': 0.524, 'dst_host_count': 0.373
    },
    'CIC-IDS2017': {
        'Bwd Packet Length Std': 0.510, 'Bwd Packet Length Max': 0.492,
        'Bwd Packet Length Mean': 0.484, 'Avg Bwd Segment Size': 0.484,
        'Packet Length Std': 0.470, 'Max Packet Length': 0.454,
        'Packet Length Variance': 0.454, 'Fwd IAT Std': 0.423,
        'Packet Length Mean': 0.414, 'Average Packet Size': 0.413
    },
    'UNSW-NB15': {
        'sttl': 0.624, 'ct_state_ttl': 0.477,
        'ct_dst_sport_ltm': 0.372, 'swin': -0.365,
        'dload': -0.352, 'dwin': -0.339,
        'rate': 0.336, 'ct_src_dport_ltm': 0.319
    }
}

@st.cache_resource
def load_models():
    models = {}
    for key, path in [('nslkdd','model_nslkdd.pkl'),('unsw','model_unsw.pkl'),('cicids','model_cicids.pkl')]:
        if os.path.exists(path):
            with open(path,'rb') as f:
                models[key] = pickle.load(f)
    return models

all_models = load_models()

def detect_dataset(cols):
    cols = set(cols)
    scores = {
        'nslkdd': len(cols & {'serror_rate','same_srv_rate','dst_host_count','src_bytes','logged_in'}),
        'unsw':   len(cols & {'sttl','ct_state_ttl','dload','swin','rate'}),
        'cicids': len(cols & {'Bwd Packet Length Std','Fwd IAT Std','Flow Duration','Total Fwd Packets'}),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Model Performance", "Dataset Analysis", "Traffic Classifier"])

# ══════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════
with tab1:
    st.subheader("Algorithm Comparison")

    available = []
    if 'nslkdd' in all_models: available.append('NSL-KDD')
    if 'unsw'   in all_models: available.append('UNSW-NB15')
    if 'cicids' in all_models: available.append('CIC-IDS2017')

    if not available:
        st.warning("No trained models found. Please run train_model.py first.")
    else:
        ds_sel  = st.selectbox("Select Dataset", available, key='perf_ds')
        ds_key  = {'NSL-KDD':'nslkdd','UNSW-NB15':'unsw','CIC-IDS2017':'cicids'}[ds_sel]
        md      = all_models[ds_key]
        results = md['all_results']
        algos   = list(results.keys())
        best    = md['best_model_name']
        bm      = results[best]

        # Best model metrics
        st.markdown(f"**Best Model: {best}**")
        c1,c2,c3,c4 = st.columns(4)
        for col, label, key in zip(
            [c1,c2,c3,c4],
            ['Accuracy','Precision','Recall','F1-Score'],
            ['accuracy','precision','recall','f1']
        ):
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{bm[key]}%</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("")

        # Bar chart
        mk = st.selectbox("Metric", ['Accuracy','Precision','Recall','F1-Score'], key='mk')
        mk_key = {'Accuracy':'accuracy','Precision':'precision','Recall':'recall','F1-Score':'f1'}[mk]
        vals = [results[a][mk_key] for a in algos]

        fig_bar = go.Figure(go.Bar(
            x=algos, y=vals,
            marker_color=CHART_COLORS[:len(algos)],
            text=[f"{v}%" for v in vals],
            textposition='outside',
            textfont=dict(color=TEXT)
        ))
        fig_bar.update_layout(
            title=dict(text=f"{mk} — {ds_sel}", font=dict(color=TEXT)),
            yaxis_range=[0, 110],
            showlegend=False,
            **chart_layout()
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Radar
        st.subheader("Radar Chart — All Metrics")
        cats     = ['Accuracy','Precision','Recall','F1-Score']
        cat_keys = ['accuracy','precision','recall','f1']
        fig2 = go.Figure()
        for i, algo in enumerate(algos):
            v = [results[algo][k] for k in cat_keys] 
            v += v[:1]
            fig2.add_trace(go.Scatterpolar(
                r=v, theta=cats+[cats[0]],
                fill='toself', name=algo,
                line_color=CHART_COLORS[i % len(CHART_COLORS)],
                opacity=0.8
            ))
        fig2.update_layout(
            polar=dict(
                bgcolor=BG,
                radialaxis=dict(visible=True, range=[0,100], color=TEXT, gridcolor=BORDER),
                angularaxis=dict(color=TEXT, gridcolor=BORDER)
            ),
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            font=dict(color=TEXT),
            legend=dict(bgcolor=CARD_BG, bordercolor=BORDER, font=dict(color=TEXT), orientation='h', y=-0.2)
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Confusion matrix
        algo_sel = st.selectbox("Select Algorithm for Confusion Matrix", algos)
        cm = np.array(results[algo_sel]['confusion_matrix'])
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Normal','Attack'], y=['Normal','Attack'],
            text_auto=True,
            color_continuous_scale=[[0,'#EBF5FB'],[0.5,'#2E86C1'],[1,'#1B3A6B']],
            title=f"Confusion Matrix — {algo_sel}"
        )
        fig_cm.update_traces(textfont=dict(color='white', size=16))
        fig_cm.update_layout(
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            font=dict(color=TEXT, size=13),
            title=dict(font=dict(color=TEXT)),
            xaxis=dict(tickfont=dict(color=TEXT, size=13), title=dict(font=dict(color=TEXT))),
            yaxis=dict(tickfont=dict(color=TEXT, size=13), title=dict(font=dict(color=TEXT))),
            coloraxis_colorbar=dict(tickfont=dict(color=TEXT), title=dict(font=dict(color=TEXT)))
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        # Feature importance
        if md.get('top_features'):
            st.subheader("Top 10 Important Features")
            fi = md['top_features']
            fi_df = pd.DataFrame({'Feature': list(fi.keys()), 'Importance': list(fi.values())})
            fi_df = fi_df.sort_values('Importance', ascending=True)
            fig_fi = go.Figure(go.Bar(
                x=fi_df['Importance'], y=fi_df['Feature'],
                orientation='h',
                marker_color=CHART_COLORS[1],
                text=[f"{v:.4f}" for v in fi_df['Importance']],
                textposition='outside',
                textfont=dict(color=TEXT)
            ))
            fig_fi.update_layout(
                title=dict(text="", font=dict(color=TEXT)),
                yaxis=dict(tickfont=dict(color=TEXT, size=12), gridcolor=BORDER),
                xaxis=dict(tickfont=dict(color=TEXT), gridcolor=BORDER),
                plot_bgcolor=BG, paper_bgcolor=BG,
                font=dict(color=TEXT)
            )
            st.plotly_chart(fig_fi, use_container_width=True)

# ══════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════
with tab2:
    st.subheader("Dataset Overview")
    df_ds = pd.DataFrame(DATASET_STATS)

    c1,c2,c3 = st.columns(3)
    for col, ds in zip([c1,c2,c3], ['NSL-KDD','CIC-IDS2017','UNSW-NB15']):
        row = df_ds[df_ds['Dataset']==ds].iloc[0]
        col.markdown(f"""
        <div class="metric-card">
            <div style="font-size:1.2rem;font-weight:700;color:{BLUE}">{ds}</div>
            <div class="metric-value" style="font-size:1.4rem">{row['Total Records']:,}</div>
            <div class="metric-label">Total Records · {row['Features']} Features</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Bar chart
    st.subheader("Class Distribution per Dataset")
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name='Normal', x=df_ds['Dataset'], y=df_ds['Normal'],
                          marker_color=BLUE, text=df_ds['Normal'], textposition='outside',
                          textfont=dict(color=TEXT)))
    fig3.add_trace(go.Bar(name='Attack', x=df_ds['Dataset'], y=df_ds['Attack'],
                          marker_color=RED, text=df_ds['Attack'], textposition='outside',
                          textfont=dict(color=TEXT)))
    fig3.update_layout(
        barmode='group',
        title=dict(text="Normal vs Attack Records", font=dict(color=TEXT)),
        legend=dict(bgcolor=BG, font=dict(color=TEXT)),
        **chart_layout()
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Pie charts
    st.subheader("Attack Ratio per Dataset")
    cols = st.columns(3)
    for col, ds in zip(cols, ['NSL-KDD','CIC-IDS2017','UNSW-NB15']):
        row = df_ds[df_ds['Dataset']==ds].iloc[0]
        fig_pie = go.Figure(go.Pie(
            labels=['Normal','Attack'],
            values=[row['Normal'], row['Attack']],
            marker=dict(colors=[BLUE, RED]),
            textfont=dict(color='white'),
        ))
        fig_pie.update_layout(
            title=dict(text=ds, font=dict(color=TEXT)),
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            font=dict(color=TEXT),
            legend=dict(font=dict(color=TEXT), bgcolor=BG)
        )
        col.plotly_chart(fig_pie, use_container_width=True)

    # Correlation
    st.subheader("Top Correlated Features with Target")
    ds_choice = st.selectbox("Select Dataset", ['NSL-KDD','CIC-IDS2017','UNSW-NB15'])
    cd = CORR_DATA[ds_choice]
    cd_df = pd.DataFrame({'Feature': list(cd.keys()), 'Correlation': list(cd.values())})
    cd_df = cd_df.sort_values('Correlation')

    bar_colors = [RED if v < 0 else BLUE for v in cd_df['Correlation']]
    fig_corr = go.Figure(go.Bar(
        x=cd_df['Correlation'], y=cd_df['Feature'],
        orientation='h',
        marker_color=bar_colors,
        text=[f"{v:.3f}" for v in cd_df['Correlation']],
        textposition='outside',
        textfont=dict(color=TEXT)
    ))
    fig_corr.update_layout(
        title=dict(text=f"Feature Correlations — {ds_choice}", font=dict(color=TEXT)),
        **chart_layout()
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ══════════════════════════════════════════
# TAB 3
# ══════════════════════════════════════════
with tab3:
    st.subheader("Network Traffic Classifier")
    mode = st.radio("Input Mode", ["Upload File", "Manual Input"])

    if mode == "Upload File":
        uploaded = st.file_uploader("", type=['txt','parquet'])
        if uploaded:
            if not all_models:
                st.warning("No trained models found. Please run train_model.py first.")
            else:
                try:
                    fname = uploaded.name.lower()
                    # parquet files
                    if fname.endswith('.parquet'):
                        import io
                        df_up = pd.read_parquet(io.BytesIO(uploaded.read()))
                        detected = detect_dataset(df_up.columns)
                    # NSL-KDD txt files (no header)
                    elif 'kdd' in fname or fname.endswith('.txt'):
                        df_up = pd.read_csv(uploaded, header=None, names=NSLKDD_COLUMNS)
                        detected = 'nslkdd'
                    else:
                        detected = None

                    if detected and detected in all_models:
                        md = all_models[detected]
                        ds_label = {'nslkdd':'NSL-KDD','unsw':'UNSW-NB15','cicids':'CIC-IDS2017'}[detected]
                        st.info(f"Detected Dataset: **{ds_label}**")

                        # Algorithm selector
                        available_algos = list(md['all_results'].keys())
                        selected_algo = st.selectbox("Select Algorithm", available_algos,
                            index=available_algos.index(md['best_model_name']) if md['best_model_name'] in available_algos else 0)

                        # Get selected model
                        selected_model = md['all_models'][selected_algo]

                        drop_cols = ['label','Label','difficulty','attack_cat','class','Class']
                        df_up.drop(columns=[c for c in drop_cols if c in df_up.columns], inplace=True, errors='ignore')

                        for col in df_up.select_dtypes(include='object').columns:
                            if col in md['encoders']:
                                le = md['encoders'][col]
                                df_up[col] = df_up[col].astype(str).apply(
                                    lambda x: le.transform([x])[0] if x in le.classes_ else 0)
                            else:
                                df_up[col] = 0

                        for col in md['feature_names']:
                            if col not in df_up.columns:
                                df_up[col] = 0

                        X = df_up[md['feature_names']]
                        X_scaled = md['scaler'].transform(X)
                        preds = selected_model.predict(X_scaled)
                        probs = selected_model.predict_proba(X_scaled)[:,1]

                        n_attack = int((preds==1).sum())
                        n_normal = int((preds==0).sum())

                        c1,c2,c3 = st.columns(3)
                        c1.metric("Total Records", len(preds))
                        c2.metric("Attacks Detected", n_attack)
                        c3.metric("Normal Traffic", n_normal)

                        fig_r = go.Figure(go.Pie(
                            labels=['Normal','Attack'],
                            values=[n_normal, n_attack],
                            marker=dict(colors=[BLUE, RED]),
                            textfont=dict(color='white'),
                        ))
                        fig_r.update_layout(
                            title=dict(text="Classification Results", font=dict(color=TEXT)),
                            paper_bgcolor=BG, plot_bgcolor=BG,
                            font=dict(color=TEXT),
                            legend=dict(font=dict(color=TEXT), bgcolor=BG)
                        )
                        st.plotly_chart(fig_r, use_container_width=True)

                        result_df = pd.DataFrame({
                            'Record': range(1, len(preds)+1),
                            'Prediction': ['Attack' if p==1 else 'Normal' for p in preds],
                            'Confidence (%)': (probs*100).round(1)
                        })
                        st.dataframe(result_df.head(200), use_container_width=True)
                    else:
                        st.error("Could not detect dataset format. Please upload NSL-KDD, UNSW-NB15, or CIC-IDS2017 file.")
                except Exception as e:
                    st.error(f"Error reading file: {e}")

    else:
        if 'nslkdd' not in all_models:
            st.warning("NSL-KDD model not found. Please run train_model.py first.")
        else:
            md = all_models['nslkdd']
            col1,col2,col3 = st.columns(3)
            with col1:
                duration       = st.number_input("Duration", 0, 60000, 0)
                src_bytes      = st.number_input("Source Bytes", 0, 10000000, 200)
                dst_bytes      = st.number_input("Destination Bytes", 0, 10000000, 0)
                logged_in      = st.selectbox("Logged In", [0,1])
            with col2:
                serror_rate    = st.slider("SYN Error Rate", 0.0, 1.0, 0.0, 0.01)
                same_srv_rate  = st.slider("Same Service Rate", 0.0, 1.0, 1.0, 0.01)
                dst_host_count = st.number_input("Dst Host Count", 0, 255, 255)
                count          = st.number_input("Count", 0, 511, 1)
            with col3:
                protocol_type = st.selectbox("Protocol", ['tcp','udp','icmp'])
                service       = st.selectbox("Service", ['http','ftp','smtp','ssh','dns','other'])
                flag          = st.selectbox("Flag", ['SF','S0','REJ','RSTO','SH','RSTR','S1'])

            if st.button("Classify Traffic", type="primary"):
                fd = {f: 0.0 for f in md['feature_names']}
                fd.update({
                    'duration': duration, 'src_bytes': src_bytes, 'dst_bytes': dst_bytes,
                    'logged_in': logged_in, 'serror_rate': serror_rate,
                    'same_srv_rate': same_srv_rate, 'dst_host_count': dst_host_count,
                    'count': count,
                    'protocol_type': md['encoders']['protocol_type'].transform(
                        [protocol_type] if protocol_type in md['encoders']['protocol_type'].classes_ else ['tcp'])[0],
                    'service': md['encoders']['service'].transform(
                        [service] if service in md['encoders']['service'].classes_ else ['http'])[0],
                    'flag': md['encoders']['flag'].transform(
                        [flag] if flag in md['encoders']['flag'].classes_ else ['SF'])[0],
                })
                X = np.array([[fd[f] for f in md['feature_names']]])
                X_s = md['scaler'].transform(X)
                pred = md['best_model'].predict(X_s)[0]
                prob = md['best_model'].predict_proba(X_s)[0][1]

                st.markdown("---")
                if pred == 1:
                    st.markdown('<div style="text-align:center"><span class="attack-badge">ATTACK DETECTED</span></div>', unsafe_allow_html=True)
                    st.error(f"Confidence: {prob*100:.1f}%")
                else:
                    st.markdown('<div style="text-align:center"><span class="normal-badge">NORMAL TRAFFIC</span></div>', unsafe_allow_html=True)
                    st.success(f"Confidence: {(1-prob)*100:.1f}%")

                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob*100,
                    title={'text': "Attack Probability", 'font': {'color': TEXT}},
                    number={'font': {'color': TEXT}},
                    gauge={
                        'axis': {'range': [0,100], 'tickfont': {'color': TEXT}},
                        'bar': {'color': RED if prob > 0.5 else BLUE},
                        'bgcolor': BG,
                        'bordercolor': BORDER,
                        'steps': [
                            {'range': [0,40],   'color': '#D5E8D4'},
                            {'range': [40,70],  'color': '#FFE6CC'},
                            {'range': [70,100], 'color': '#F8CECC'},
                        ],
                    }
                ))
                fig_g.update_layout(paper_bgcolor=BG, font=dict(color=TEXT), height=300)
                st.plotly_chart(fig_g, use_container_width=True)
