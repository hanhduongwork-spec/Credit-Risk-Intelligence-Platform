# Credit Risk Intelligence — Lending Club

> Dự đoán xác suất vỡ nợ (PD) và tối ưu hóa danh mục cho vay

---

## 1. Tổng quan

Dự án xây dựng hệ thống đánh giá rủi ro tín dụng cho dữ liệu Lending Club, bao gồm:

- **Mô hình dự đoán PD** sử dụng LightGBM + Optuna fine-tuning
- **Hiệu chuẩn xác suất** (Platt Scaling) để PD output đáng tin cậy
- **4 kịch bản tối ưu hóa** phục vụ ra quyết định kinh doanh
- **Dashboard Streamlit** trực quan hóa toàn bộ kết quả

---

## 🗂️ Cấu trúc dự án

```
credit-risk-lending-club/
├── app.py                              # Streamlit dashboard
├── pipeline.py                         # Pipeline huấn luyện
├── requirements.txt
├── README.md
├── .gitignore
├── lending_club_loan_cleaned.csv       # Data đã preprocessing
├── data/
│   ├── result_with_pd.csv              # PD + E[Profit] toàn danh mục
│   ├── kb2_pricing.csv                 # KB2: Pricing policy
│   ├── kb3_optimization.csv            # KB3: Approval threshold
│   └── kb4_allocation.csv              # KB4: Capital allocation
└── models/
    ├── final_model.pkl                 # LightGBM đã fine-tune
    ├── cal_model.pkl                   # Model đã hiệu chuẩn (Platt)
    └── selected_features.pkl           # Danh sách features
```

---

## 🚀 Cách chạy

### 1. Clone repo & cài thư viện
```bash
git clone https://github.com/<username>/credit-risk-lending-club.git
cd credit-risk-lending-club
pip install -r requirements.txt
```

### 2. Chạy pipeline (tạo model + artifacts)
```bash
python pipeline.py
```

### 3. Chạy Streamlit app
```bash
streamlit run app.py
```

---

## 📊 Pipeline Flow

```
lending_club_loan_cleaned.csv
    ↓ Feature Engineering
    ↓ Encode & Train/Test Split (80/20, stratified)
    ↓ Baseline: LightGBM / XGBoost / LogReg × Full / Top15
    ↓ Fine-tune LightGBM (Optuna, 20 trials)
    ↓ Evaluate (AUC, KS, Gini, F2)
    ↓ Calibration (Platt Scaling)
    ↓ Predict PD
    ├── KB1: Optimal Interest Rate (1 khách hàng)
    ├── KB2: Pricing Policy theo Sub_Grade
    ├── KB3: Portfolio Approval Policy
    └── KB4: Capital Allocation Risk/Return
    ↓ Save Artifacts → Streamlit
```

---

## 📈 Kết quả mô hình

| Metric | Giá trị |
|--------|---------|
| AUC-ROC | **0.7169** |
| KS Statistic | 0.3190 |
| Gini Coefficient | 0.4337 |

---

## 🎯 4 Kịch bản tối ưu hóa

| # | Câu hỏi kinh doanh | Phương pháp |
|---|---|---|
| **KB1** | Nên chào lãi suất bao nhiêu cho khách X? | Monte Carlo + Grid Search |
| **KB2** | Lãi suất tối thiểu cho từng sub_grade? | Analytical Profit Function |
| **KB3** | Từ chối khoản vay PD > bao nhiêu? | Profit/Loan Ratio Optimization |
| **KB4** | Phân bổ vốn vào sub_grade nào? | Greedy Risk/Return Allocation |

---

## ⚠️ Lưu ý

- `lending_club_loan_cleaned.csv` phải nằm cùng thư mục với `pipeline.py`
- Chạy `pipeline.py` trước để tạo `models/` và `data/` trước khi chạy app
- Raw data (`lending_club_loan.csv`) tải từ [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

---

*Group 1 K62DB · Credit Risk Modeling*
