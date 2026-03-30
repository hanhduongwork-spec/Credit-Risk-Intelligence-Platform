"""
Credit Risk Dashboard — Lending Club
Run: streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LogisticRegression

# ── PlattCalibrator phải định nghĩa ở đây để joblib load được ──
class PlattCalibrator:
    def __init__(self, base_model):
        self.base_model = base_model
        self.sig        = LogisticRegression(max_iter=1000)

    def fit(self, X_val, y_val):
        raw = self.base_model.predict_proba(X_val)[:, 1].reshape(-1, 1)
        self.sig.fit(raw, y_val)
        return self

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1].reshape(-1, 1)
        cal = self.sig.predict_proba(raw)[:, 1]
        return np.column_stack([1 - cal, cal])


st.set_page_config(page_title="Credit Risk Intelligence", page_icon="💳",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Mono', monospace; background-color: #0a0e1a; color: #e2e8f0; }
.stApp { background-color: #0a0e1a; }
.hero { background: linear-gradient(135deg,#0f172a,#1e293b,#0f172a); border:1px solid #1e3a5f; border-radius:16px; padding:40px 48px; margin-bottom:32px; position:relative; overflow:hidden; }
.hero::before { content:''; position:absolute; top:-50%; left:-50%; width:200%; height:200%;
  background: radial-gradient(circle at 30% 50%,rgba(56,189,248,.06),transparent 50%),radial-gradient(circle at 70% 50%,rgba(99,102,241,.06),transparent 50%); }
.hero-title { font-family:'Syne',sans-serif; font-size:2.4rem; font-weight:800;
  background:linear-gradient(90deg,#38bdf8,#818cf8,#38bdf8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0 0 8px; }
.hero-sub { color:#64748b; font-size:.85rem; letter-spacing:.1em; text-transform:uppercase; }
.section-header { font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:700; color:#f1f5f9;
  border-left:3px solid #38bdf8; padding-left:14px; margin:32px 0 20px; }
.metric-card { background:linear-gradient(135deg,#0f172a,#1e293b); border:1px solid #1e3a5f; border-radius:12px; padding:20px 24px; text-align:center; }
.metric-label { font-size:.72rem; color:#64748b; text-transform:uppercase; letter-spacing:.1em; margin-bottom:8px; }
.metric-value { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:700; color:#f1f5f9; }
.metric-value.positive { color:#34d399; } .metric-value.negative { color:#f87171; } .metric-value.highlight { color:#38bdf8; }
.stTabs [data-baseweb="tab-list"] { background:#0f172a; border-radius:10px; padding:4px; border:1px solid #1e293b; gap:4px; }
.stTabs [data-baseweb="tab"] { background:transparent; color:#64748b; border-radius:8px; font-family:'DM Mono',monospace; font-size:.8rem; padding:8px 20px; border:none; }
.stTabs [aria-selected="true"] { background:#1e3a5f !important; color:#38bdf8 !important; }
.stButton button { background:linear-gradient(135deg,#1d4ed8,#0ea5e9) !important; color:white !important;
  border:none !important; border-radius:8px !important; font-family:'Syne',sans-serif !important; font-weight:600 !important; }
.pd-gauge { background:#0f172a; border:1px solid #1e3a5f; border-radius:12px; padding:24px; margin:16px 0; }
.pd-bar-bg { background:#1e293b; border-radius:999px; height:12px; width:100%; overflow:hidden; }
.pd-bar-fill { height:100%; border-radius:999px; }
.decision-badge { display:inline-block; padding:6px 16px; border-radius:999px; font-family:'Syne',sans-serif; font-weight:700; font-size:.9rem; }
.approve { background:rgba(52,211,153,.15); color:#34d399; border:1px solid #34d399; }
.reject  { background:rgba(248,113,113,.15); color:#f87171; border:1px solid #f87171; }
.divider { border:none; border-top:1px solid #1e293b; margin:28px 0; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ──────────────────────────────────────────────
LGD_BASE     = 0.70
ORIG_FEE     = 0.03
GRADE_COLORS = {'A':'#34d399','B':'#38bdf8','C':'#fbbf24',
                'D':'#fb923c','E':'#f87171','F':'#e879f9','G':'#a78bfa'}

def tinh_total_income(loan_amnt, int_rate, term_months, fee=ORIG_FEE):
    r = int_rate/100/12; n = int(term_months)
    pmt = loan_amnt/n if r==0 else loan_amnt*(r*(1+r)**n)/((1+r)**n-1)
    return (pmt*n - loan_amnt) + loan_amnt*fee

def monte_carlo_profit(PD, loan_amnt, int_rate, term_months, lgd=LGD_BASE, N=10_000):
    income  = tinh_total_income(loan_amnt, int_rate, term_months)
    profits = np.where(np.random.binomial(1, PD, N)==1, -loan_amnt*lgd, income)
    return {'mean_profit': profits.mean(), 'std_profit': profits.std(),
            'var_95': np.percentile(profits, 5), 'prob_profit': (profits>0).mean()}

def style_chart(fig, axes):
    fig.patch.set_facecolor('#0f172a')
    for ax in (axes if isinstance(axes, list) else [axes]):
        ax.set_facecolor('#0f172a')
        ax.tick_params(colors='#64748b', labelsize=8)
        ax.xaxis.label.set_color('#64748b'); ax.yaxis.label.set_color('#64748b')
        ax.title.set_color('#e2e8f0')
        for s in ax.spines.values(): s.set_edgecolor('#1e3a5f')
        ax.grid(True, alpha=0.15, color='#334155')

def mcard(col, label, value, cls="highlight"):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div>'
                    f'<div class="metric-value {cls}">{value}</div></div>', unsafe_allow_html=True)

# ── LOAD ARTIFACTS ─────────────────────────────────────────
@st.cache_resource
def load_models():
    return (joblib.load('models/final_model.pkl'),
            joblib.load('models/cal_model.pkl'),
            joblib.load('models/selected_features.pkl'))

@st.cache_data
def load_data():
    return (pd.read_csv('data/result_with_pd.csv'),
            pd.read_csv('data/kb2_pricing.csv'),
            pd.read_csv('data/kb3_optimization.csv'),
            pd.read_csv('data/kb4_allocation.csv'))

try:
    model, cal_model, features = load_models()
    result_df, df_kb2, df_kb3, df_kb4 = load_data()

    # Tính optimal threshold từ kb3
    if 'profit_ratio' not in df_kb3.columns:
        df_kb3['profit_ratio'] = df_kb3['total_profit'] / df_kb3['total_loan']
    OPT_THRESHOLD = float(df_kb3.loc[df_kb3['profit_ratio'].idxmax(), 'threshold'])
except Exception as e:
    st.error(f"❌ Không load được artifacts: {e}")
    st.info("Chạy `python pipeline.py` trước để tạo models/ và data/")
    st.stop()

# ── HERO ───────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">Credit Risk Intelligence</div>
</div>
""", unsafe_allow_html=True)

# Overview metrics
total_profit = result_df['E_profit'].sum() if 'E_profit' in result_df.columns else 0
c1,c2,c3,c4,c5 = st.columns(5)
for col, lbl, val, cls in [
    (c1, "Tổng khoản vay",     f"{len(result_df):,}",                       "highlight"),
    (c2, "Tổng dư nợ",         f"${result_df['loan_amnt'].sum()/1e6:.1f}M", "highlight"),
    (c3, "PD trung bình",      f"{result_df['PD'].mean():.1%}",              "negative" if result_df['PD'].mean()>0.3 else "positive"),
    (c4, "E[Profit] danh mục", f"${total_profit/1e6:.1f}M",                  "positive" if total_profit>0 else "negative"),
    (c5, "Optimal Threshold",  f"PD ≤ {OPT_THRESHOLD:.2f}",                 "highlight"),
]: mcard(col, lbl, val, cls)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Dự đoán PD", "📊 KB1 — Optimal Rate",
    "🏷️ KB2 — Pricing Policy", "✅ KB3 — Approval Policy", "💰 KB4 — Capital Allocation"
])

# ── TAB 1: PREDICT PD ──────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Dự đoán xác suất vỡ nợ (PD)</div>', unsafe_allow_html=True)
    with st.form("pred"):
        c1,c2,c3 = st.columns(3)
        with c1:
            loan_amnt  = st.number_input("Số tiền vay ($)", 500, 40000, 10000, 500)
            int_rate   = st.number_input("Lãi suất (%)", 5.0, 30.0, 12.0, 0.1)
            term       = st.selectbox("Kỳ hạn (tháng)", [36, 60])
        with c2:
            annual_inc     = st.number_input("Thu nhập/năm ($)", 10000, 500000, 60000, 1000)
            dti            = st.number_input("DTI (%)", 0.0, 50.0, 15.0, 0.1)
            home_ownership = st.selectbox("Nhà ở", ["RENT","MORTGAGE","OWN","OTHER"])
        with c3:
            revol_util   = st.number_input("Revol Utilization (%)", 0.0, 100.0, 40.0, 1.0)
            revol_bal    = st.number_input("Revol Balance ($)", 0, 200000, 15000, 1000)
            pub_rec      = st.number_input("Public Records", 0, 10, 0)
        c4,c5,c6 = st.columns(3)
        with c4:
            open_acc  = st.number_input("Open Accounts", 1, 50, 8)
            total_acc = st.number_input("Total Accounts", 1, 100, 20)
        with c5:
            mort_acc             = st.number_input("Mortgage Accounts", 0, 20, 0)
            credit_age           = st.number_input("Credit Age (years)", 1, 50, 10)
            pub_rec_bankruptcies = st.number_input("Bankruptcies", 0, 5, 0)
        with c6:
            verification_status = st.selectbox("Verification", ["Not Verified","Source Verified","Verified"])
            purpose             = st.selectbox("Mục đích", ["debt_consolidation","credit_card",
                                               "home_improvement","other","major_purchase",
                                               "medical","small_business","car","vacation"])
            initial_list_status = st.selectbox("Initial List Status", ["w","f"])
        submitted = st.form_submit_button("🔮 Dự đoán", use_container_width=True)

    if submitted:
        inp = {
            "loan_amnt": loan_amnt, "int_rate": int_rate, "dti": dti,
            "pub_rec": pub_rec, "revol_bal": revol_bal, "revol_util": revol_util,
            "pub_rec_bankruptcies": pub_rec_bankruptcies, "open_acc": open_acc,
            "total_acc": total_acc, "mort_acc": mort_acc,
            "cbrt_annual_inc": np.cbrt(annual_inc), "cbrt_open_acc": np.cbrt(open_acc),
            "cbrt_revol_bal": np.cbrt(revol_bal), "cbrt_total_acc": np.cbrt(total_acc),
            "cbrt_mort_acc": np.cbrt(mort_acc), "term_months": term,
            "loan_to_revol_ratio": loan_amnt/(revol_util+1),
            "credit_age_years": credit_age,
            "total_bad_records": pub_rec + pub_rec_bankruptcies,
        }
        for opt in ["MORTGAGE","OTHER","OWN"]:
            inp[f"home_ownership_{opt}"] = int(home_ownership==opt)
        for opt in ["Source Verified","Verified"]:
            inp[f"verification_status_{opt}"] = int(verification_status==opt)
        for opt in ["credit_card","debt_consolidation","home_improvement","major_purchase",
                    "medical","other","small_business","vacation","car"]:
            inp[f"purpose_{opt}"] = int(purpose==opt)
        inp["initial_list_status_w"] = int(initial_list_status=="w")

        df_inp = pd.DataFrame([inp])
        for f in features:
            if f not in df_inp.columns: df_inp[f] = 0
        df_inp = df_inp[features]

        pd_score = cal_model.predict_proba(df_inp)[0, 1]
        mc       = monte_carlo_profit(pd_score, loan_amnt, int_rate, term)
        decision = pd_score <= OPT_THRESHOLD
        bar_col  = "#34d399" if decision else "#f87171"

        st.markdown(f"""
        <div class="pd-gauge">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
            <div>
              <span style="font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:{bar_col}">{pd_score:.1%}</span>
              <span style="color:#64748b;font-size:.8rem;margin-left:8px">Xác suất vỡ nợ</span>
            </div>
            <span class="decision-badge {'approve' if decision else 'reject'}">
              {'✅ PHÊ DUYỆT' if decision else '❌ TỪ CHỐI'}
            </span>
          </div>
          <div class="pd-bar-bg">
            <div class="pd-bar-fill" style="width:{min(pd_score*100,100):.1f}%;background:{bar_col}"></div>
          </div>
          <div style="display:flex;justify-content:space-between;margin-top:6px;color:#64748b;font-size:.7rem">
            <span>0%</span>
            <span style="color:#fbbf24">Threshold = {OPT_THRESHOLD:.0%}</span>
            <span>100%</span>
          </div>
        </div>""", unsafe_allow_html=True)

        r1,r2,r3,r4 = st.columns(4)
        for col, lbl, val, cls in [
            (r1,"E[Profit]",  f"${mc['mean_profit']:,.0f}", "positive" if mc['mean_profit']>0 else "negative"),
            (r2,"VaR 95%",    f"${mc['var_95']:,.0f}",      "negative"),
            (r3,"P(có lãi)",  f"{mc['prob_profit']:.1%}",   "positive" if mc['prob_profit']>0.5 else "negative"),
            (r4,"Std Profit", f"${mc['std_profit']:,.0f}",  "highlight"),
        ]: mcard(col, lbl, val, cls)

# ── TAB 2: KB1 ─────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">KB1 — Tìm lãi suất tối ưu cho khách hàng</div>', unsafe_allow_html=True)
    st.caption("Quét grid lãi suất → Monte Carlo 5k lần → tìm mức tối đa hóa E[Profit].")
    with st.form("kb1"):
        k1,k2,k3 = st.columns(3)
        with k1:
            kb1_loan = st.number_input("Số tiền vay ($)", 500, 40000, 10000, 500, key="k1l")
            kb1_term = st.selectbox("Kỳ hạn (tháng)", [36, 60], key="k1t")
        with k2:
            kb1_rmin = st.number_input("Lãi suất tối thiểu (%)", 5.0, 20.0, 6.0, 0.5, key="k1rn")
            kb1_rmax = st.number_input("Lãi suất tối đa (%)", 10.0, 30.0, 25.0, 0.5, key="k1rx")
        with k3:
            kb1_pd = st.slider("PD cơ sở (%)", 5, 60, 20, key="k1pd")
        kb1_go = st.form_submit_button("🔍 Tìm lãi suất tối ưu", use_container_width=True)

    if kb1_go:
        rows = []
        for rate in np.arange(kb1_rmin, kb1_rmax+0.5, 0.5):
            pd_adj = np.clip(kb1_pd/100 * (1+(rate-kb1_rmin)/100), 0.001, 0.999)
            mc = monte_carlo_profit(pd_adj, kb1_loan, rate, kb1_term, N=5_000)
            rows.append({'int_rate': rate, 'PD': pd_adj, **mc})
        df_r = pd.DataFrame(rows)
        best = df_r.loc[df_r['mean_profit'].idxmax()]

        m1,m2,m3 = st.columns(3)
        for col,lbl,val,cls in [
            (m1,"Lãi suất tối ưu",f"{best['int_rate']:.1f}%","highlight"),
            (m2,"E[Profit] tối đa",f"${best['mean_profit']:,.0f}","positive" if best['mean_profit']>0 else "negative"),
            (m3,"PD tại optimal",f"{best['PD']:.1%}","positive"),
        ]: mcard(col,lbl,val,cls)

        fig, axes = plt.subplots(1, 2, figsize=(12,4))
        style_chart(fig, list(axes))
        axes[0].plot(df_r['int_rate'], df_r['mean_profit'], color='#38bdf8', lw=2)
        axes[0].axvline(best['int_rate'], color='#f87171', ls='--', label=f"Optimal={best['int_rate']:.1f}%")
        axes[0].axhline(0, color='#475569', ls=':')
        axes[0].fill_between(df_r['int_rate'], df_r['mean_profit'], 0,
                              where=df_r['mean_profit']>0, alpha=0.15, color='#34d399')
        axes[0].fill_between(df_r['int_rate'], df_r['mean_profit'], 0,
                              where=df_r['mean_profit']<=0, alpha=0.15, color='#f87171')
        axes[0].set_title('E[Profit] theo Lãi suất'); axes[0].legend(labelcolor='#e2e8f0')
        axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))
        axes[1].plot(df_r['int_rate'], df_r['PD']*100, color='#fb923c', lw=2)
        axes[1].axvline(best['int_rate'], color='#f87171', ls='--')
        axes[1].set_title('PD (%) theo Lãi suất')
        axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x:.1f}%'))
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ── TAB 3: KB2 ─────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">KB2 — Pricing Policy theo Sub_Grade</div>', unsafe_allow_html=True)
    st.caption("Lãi suất tối ưu cho từng nhóm khách hàng A1→G5.")
    if 'grade' in df_kb2.columns:
        colors = [GRADE_COLORS.get(g,'#64748b') for g in df_kb2['grade']]
        fig, axes = plt.subplots(1, 3, figsize=(16,5))
        style_chart(fig, list(axes))
        axes[0].bar(df_kb2['sub_grade'], df_kb2['optimal_rate'], color=colors, alpha=0.85)
        axes[0].set_title('Lãi suất tối ưu (%)'); axes[0].tick_params(axis='x', rotation=90, labelsize=7)
        axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x:.0f}%'))
        if 'PD_at_optimal' in df_kb2.columns:
            axes[1].bar(df_kb2['sub_grade'], df_kb2['PD_at_optimal']*100, color=colors, alpha=0.85)
            axes[1].set_title('PD tại Optimal Rate (%)'); axes[1].tick_params(axis='x', rotation=90, labelsize=7)
            axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x:.1f}%'))
        if 'E_profit' in df_kb2.columns:
            axes[2].bar(df_kb2['sub_grade'], df_kb2['E_profit'], color=colors, alpha=0.85)
            axes[2].axhline(0, color='#475569', ls=':')
            axes[2].set_title('E[Profit] ($)'); axes[2].tick_params(axis='x', rotation=90, labelsize=7)
            axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))
        plt.tight_layout(); st.pyplot(fig); plt.close()

        summary = df_kb2.groupby('grade').agg(
            Rate_Min=('optimal_rate','min'), Rate_Max=('optimal_rate','max'),
            Avg_PD=('PD_at_optimal','mean'), Avg_Profit=('E_profit','mean')
        ).reset_index()
        st.dataframe(summary, use_container_width=True, hide_index=True)

# ── TAB 4: KB3 ─────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">KB3 — Portfolio Approval Policy</div>', unsafe_allow_html=True)
    st.caption("Ngưỡng PD tối ưu tối đa hóa Profit/Loan ratio — không bị lệch bởi quy mô danh mục.")
    if 'profit_ratio' in df_kb3.columns or 'total_profit' in df_kb3.columns:
        if 'profit_ratio' not in df_kb3.columns:
            df_kb3['profit_ratio'] = df_kb3['total_profit'] / df_kb3['total_loan']
        opt = df_kb3.loc[df_kb3['profit_ratio'].idxmax()]

        m1,m2,m3,m4 = st.columns(4)
        for col,lbl,val,cls in [
            (m1,"Threshold tối ưu",  f"PD ≤ {opt['threshold']:.2f}", "highlight"),
            (m2,"Approval Rate",     f"{opt['approval_rate']:.1f}%",  "positive"),
            (m3,"Profit/Loan Ratio", f"{opt['profit_ratio']*100:.2f}%","positive"),
            (m4,"Tổng E[Profit]",    f"${opt['total_profit']/1e6:.2f}M","positive" if opt['total_profit']>0 else "negative"),
        ]: mcard(col,lbl,val,cls)

        fig, axes = plt.subplots(1, 3, figsize=(18,4))
        style_chart(fig, list(axes))

        axes[0].plot(df_kb3['threshold'], df_kb3['profit_ratio']*100, color='#38bdf8', lw=2)
        axes[0].axvline(opt['threshold'], color='#f87171', ls='--', label=f"Optimal={opt['threshold']:.2f}")
        axes[0].set_xlabel('PD Threshold'); axes[0].set_ylabel('Profit/Loan (%)')
        axes[0].set_title('Profit/Loan Ratio (Metric chính)'); axes[0].legend(labelcolor='#e2e8f0')
        axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x:.2f}%'))

        axes[1].plot(df_kb3['threshold'], df_kb3['total_profit']/1e6, color='#818cf8', lw=2)
        axes[1].axvline(opt['threshold'], color='#f87171', ls='--')
        axes[1].axhline(0, color='#475569', ls=':')
        axes[1].fill_between(df_kb3['threshold'], df_kb3['total_profit']/1e6, 0,
                              where=df_kb3['total_profit']>0, alpha=0.15, color='#34d399')
        axes[1].set_xlabel('PD Threshold'); axes[1].set_ylabel('Total E[Profit] ($M)')
        axes[1].set_title('Tổng Lợi nhuận (Tham khảo)')
        axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.1f}M'))

        axes[2].plot(df_kb3['threshold'], df_kb3['approval_rate'], color='#fbbf24', lw=2)
        axes[2].axvline(opt['threshold'], color='#f87171', ls='--', label=f"{opt['approval_rate']:.1f}%")
        axes[2].set_xlabel('PD Threshold'); axes[2].set_ylabel('Approval Rate (%)')
        axes[2].set_title('Tỷ lệ phê duyệt'); axes[2].legend(labelcolor='#e2e8f0')

        plt.tight_layout(); st.pyplot(fig); plt.close()

# ── TAB 5: KB4 ─────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">KB4 — Capital Allocation theo Risk/Return</div>', unsafe_allow_html=True)
    st.caption("Phân bổ vốn tối ưu vào sub_grade có PD ≤ 35% và E[Profit] > 0.")
    if 'sub_grade' in df_kb4.columns:
        df_kb4['grade'] = df_kb4['sub_grade'].str[0]
        colors = [GRADE_COLORS.get(g,'#64748b') for g in df_kb4['grade']]
        total_exp = df_kb4['expected_total_profit'].sum() if 'expected_total_profit' in df_kb4.columns else 0

        a1,a2,a3 = st.columns(3)
        for col,lbl,val,cls in [
            (a1,"Sub_Grade đủ điều kiện", f"{len(df_kb4)}",           "highlight"),
            (a2,"Tổng vốn phân bổ",       f"${df_kb4['allocated_budget'].sum()/1e6:.1f}M" if 'allocated_budget' in df_kb4.columns else "N/A","highlight"),
            (a3,"Tổng E[Profit]",          f"${total_exp/1e6:.2f}M",  "positive" if total_exp>0 else "negative"),
        ]: mcard(col,lbl,val,cls)

        st.markdown("<br>", unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 3, figsize=(17,5))
        style_chart(fig, list(axes))

        if 'profit_ratio' in df_kb4.columns:
            axes[0].barh(df_kb4['sub_grade'], df_kb4['profit_ratio']*100, color=colors, alpha=0.85)
            axes[0].set_xlabel('Profit/Loan (%)'); axes[0].set_title('Return per $ Lent')
            axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x:.1f}%'))

        if 'allocated_budget' in df_kb4.columns:
            axes[1].bar(df_kb4['sub_grade'], df_kb4['allocated_budget']/1e6, color=colors, alpha=0.85)
            axes[1].set_ylabel('Budget ($M)'); axes[1].set_title('Phân bổ ngân sách')
            axes[1].tick_params(axis='x', rotation=90, labelsize=7)
            axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}M'))

        if 'avg_PD' in df_kb4.columns and 'profit_ratio' in df_kb4.columns:
            sz = df_kb4['weight']*3000 if 'weight' in df_kb4.columns else 200
            axes[2].scatter(df_kb4['avg_PD']*100, df_kb4['profit_ratio']*100,
                            s=sz, c=colors, alpha=0.8, edgecolors='#0f172a', lw=0.5)
            for _, row in df_kb4.iterrows():
                axes[2].annotate(row['sub_grade'], (row['avg_PD']*100, row['profit_ratio']*100),
                                 fontsize=7, ha='center', va='bottom', color='#94a3b8')
            axes[2].axvline(35, color='#f87171', ls='--', lw=1, label='Max PD=35%')
            axes[2].set_xlabel('Avg PD (%)'); axes[2].set_ylabel('Profit/Loan (%)')
            axes[2].set_title('Risk vs Return'); axes[2].legend(labelcolor='#e2e8f0', fontsize=8)

        plt.tight_layout(); st.pyplot(fig); plt.close()

        disp = [c for c in ['sub_grade','avg_PD','profit_ratio','weight',
                             'allocated_budget','n_loans_target','expected_total_profit']
                if c in df_kb4.columns]
        st.dataframe(df_kb4[disp], use_container_width=True, hide_index=True)

# Footer
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div style="text-align:center;color:#334155;font-size:.75rem;padding:8px 0">'
            'Credit Risk Intelligence · LightGBM + Optuna · Lending Club · FTU Business Analytics</div>',
            unsafe_allow_html=True)
