import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Dự báo Năng suất Lúa (SVR)",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Ứng dụng Dự báo Năng suất Lúa bằng Mô hình SVR")
st.markdown("---")

# ----------------------------------------------------
# Các hàm xử lý dữ liệu (Dựa trên code gốc)
# ----------------------------------------------------
def bo_dau(s):
    """Bỏ dấu tiếng Việt phục vụ dò tìm tên cột"""
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.replace("đ", "d")

def tim_cot_muc_tieu(train, test):
    """Tự động tìm cột mục tiêu (năng suất)"""
    chi_co_o_train = [c for c in train.columns if c not in test.columns]
    if len(chi_co_o_train) == 1:
        return chi_co_o_train[0]
    
    tu_khoa = ["nang suat", "nangsuat", "nang_suat", "yield", "san luong"]
    ung_vien = chi_co_o_train if chi_co_o_train else list(train.columns)
    for c in ung_vien:
        if any(k in bo_dau(c) for k in tu_khoa):
            return c
    return None

def tach_ngay_thang(df):
    """Tách các cột datetime thành năm, tháng, ngày"""
    df = df.copy()
    for c in list(df.columns):
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c + "_nam"] = df[c].dt.year
            df[c + "_thang"] = df[c].dt.month
            df[c + "_ngay"] = df[c].dt.day
            df = df.drop(columns=[c])
    return df

def tien_xu_ly_data(train_df, test_df, target_col, drop_cols=[]):
    """Tiền xử lý toàn bộ dữ liệu Train và Test"""
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Chuẩn hóa tên cột
    train_df.columns = [str(c).strip() for c in train_df.columns]
    test_df.columns = [str(c).strip() for c in test_df.columns]

    # Loại bỏ dòng không có target
    y = pd.to_numeric(train_df[target_col], errors="coerce")
    hop_le = y.notna()
    train_df = train_df.loc[hop_le].reset_index(drop=True)
    y = y[hop_le].reset_index(drop=True)

    X = train_df.drop(columns=[target_col] + [c for c in drop_cols if c in train_df.columns])
    X = tach_ngay_thang(X)
    X_test = tach_ngay_thang(test_df).reindex(columns=X.columns)

    cat_cols = []
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            so = pd.to_numeric(X[c], errors="coerce")
            co_gia_tri = X[c].notna().sum()
            if co_gia_tri and so.notna().sum() >= 0.9 * co_gia_tri:
                X[c] = so
                X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
            else:
                cat_cols.append(c)

    for c in cat_cols:
        X[c] = X[c].fillna("missing").astype(str)
        X_test[c] = X_test[c].fillna("missing").astype(str)

    num_cols = [c for c in X.columns if c not in cat_cols]
    return X, y, X_test, num_cols, cat_cols

def build_svr_pipeline(num_cols, cat_cols):
    """Tạo Pipeline SVR với TransformedTargetRegressor"""
    tien_xu_ly = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    mo_hinh = TransformedTargetRegressor(
        regressor=Pipeline([("prep", tien_xu_ly), ("svr", SVR(kernel="rbf"))]),
        transformer=StandardScaler()
    )
    return mo_hinh


# ----------------------------------------------------
# Giao diện chính Streamlit (Tabs)
# ----------------------------------------------------
tab1, tab2 = st.tabs(["📊 Huấn luyện & Dự báo File", "✏️ Dự báo Nhập tay 1 mẫu"])

with tab1:
    st.header("1. Tải lên dữ liệu Excel")
    col1, col2 = st.sidebar.columns(2)
    
    train_file = st.file_uploader("Tải lên tập dữ liệu TRAIN (.xlsx)", type=["xlsx"])
    test_file = st.file_uploader("Tải lên tập dữ liệu TEST (.xlsx)", type=["xlsx"])

    if train_file and test_file:
        train_df = pd.read_excel(train_file)
        test_df = pd.read_excel(test_file)

        st.success(f"Đã tải thành công! Train: {train_df.shape}, Test: {test_df.shape}")

        # Tự động dò cột mục tiêu
        auto_target = tim_cot_muc_tieu(train_df, test_df)
        all_cols = list(train_df.columns)
        default_idx = all_cols.index(auto_target) if auto_target in all_cols else 0
        
        target_col = st.selectbox("Chọn cột mục tiêu (Target Column):", all_cols, index=default_idx)
        
        if st.button("🚀 Bắt đầu Huấn luyện & Dự báo", type="primary"):
            with st.spinner("Đang xử lý dữ liệu và chạy GridSearchCV..."):
                # Tiền xử lý
                X, y, X_test, num_cols, cat_cols = tien_xu_ly_data(train_df, test_df, target_col)
                
                # Lưu thông tin tính chất các cột vào Session State để dùng ở Tab 2
                st.session_state['X_sample'] = X.iloc[0:1]
                st.session_state['num_cols'] = num_cols
                st.session_state['cat_cols'] = cat_cols

                # Chia train / validation
                SEED = 42
                X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

                # Pipeline & Hyperparameter Tuning
                mo_hinh = build_svr_pipeline(num_cols, cat_cols)
                luoi_tham_so = {
                    "regressor__svr__C": [0.1, 1, 10, 100],
                    "regressor__svr__epsilon": [0.01, 0.05, 0.1, 0.2],
                    "regressor__svr__gamma": ["scale", 0.001, 0.01, 0.1],
                }

                cv = KFold(n_splits=min(5, len(X_tr)), shuffle=True, random_state=SEED)
                grid = GridSearchCV(mo_hinh, luoi_tham_so, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
                grid.fit(X_tr, y_tr)

                # Huấn luyện mô hình hoàn chỉnh trên toàn bộ Train
                final_model = grid.best_estimator_
                final_model.fit(X, y)
                st.session_state['trained_model'] = final_model
                st.session_state['feature_names'] = X.columns.tolist()

                # Đánh giá tập Validation
                pred_val = grid.predict(X_val)
                mae_val = mean_absolute_error(y_val, pred_val)
                rmse_val = np.sqrt(mean_squared_error(y_val, pred_val))
                r2_val = r2_score(y_val, pred_val)

            st.success("Tối ưu và Huấn luyện mô hình hoàn tất!")
            
            # Hiển thị tham số & độ đo
            st.subheader("📌 Tham số tốt nhất & Đánh giá trên tập validation (20%)")
            st.json(grid.best_params_)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE", f"{mae_val:.4f}")
            m2.metric("RMSE", f"{rmse_val:.4f}")
            m3.metric("R² Score", f"{r2_val:.4f}")

            # Hiển thị Permutation Importance
            st.subheader("📊 Mức độ quan trọng của đặc trưng (Permutation Importance)")
            with st.spinner("Đang tính toán mức độ quan trọng đặc trưng..."):
                pi = permutation_importance(
                    grid.best_estimator_, X_val, y_val, n_repeats=10,
                    random_state=SEED, scoring="neg_mean_absolute_error"
                )
                quan_trong = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=True)

                fig, ax = plt.subplots(figsize=(8, 5))
                quan_trong.tail(15).plot(kind="barh", ax=ax)
                ax.set_title("Top 15 Đặc trưng quan trọng (SVR)")
                ax.set_xlabel("Mức tăng MAE khi xáo trộn đặc trưng")
                st.pyplot(fig)

            # Dự báo dữ liệu Test
            st.subheader("🔮 Kết quả dự báo trên tập Test")
            du_bao_test = final_model.predict(X_test)
            
            ket_qua_df = test_df.copy()
            cot_du_bao = f"Du_bao_{target_col}"
            ket_qua_df[cot_du_bao] = np.round(du_bao_test, 4)

            if target_col in test_df.columns:
                thuc_te = pd.to_numeric(test_df[target_col], errors="coerce")
                ket_qua_df["Sai_so_tuyet_doi"] = np.round((thuc_te - du_bao_test).abs(), 4)

            st.dataframe(ket_qua_df)

            # Nút tải xuống kết quả Excel
            output_file = "KetQua_DuBao_SVR.xlsx"
            ket_qua_df.to_excel(output_file, index=False)
            with open(output_file, "rb") as f:
                st.download_button(
                    label="📥 Tải xuống kết quả dự báo (Excel)",
                    data=f,
                    file_name="KetQua_DuBao_SVR.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    else:
        st.info("Vui lòng tải lên cả 2 file Train và Test dạng Excel (.xlsx) ở khung trên để bắt đầu.")

with tab2:
    st.header("2. Dự báo trực tiếp 1 mẫu đầu vào")
    
    if 'trained_model' not in st.session_state:
        st.warning("Bạn cần chạy huấn luyện mô hình ở **Tab 1** trước khi sử dụng tính năng dự báo thủ công!")
    else:
        model = st.session_state['trained_model']
        feature_names = st.session_state['feature_names']
        num_cols = st.session_state['num_cols']
        cat_cols = st.session_state['cat_cols']
        X_sample = st.session_state['X_sample']

        st.markdown("Nhập các giá trị thông số để dự báo năng suất:")
        
        input_data = {}
        cols_ui = st.columns(3)
        
        for idx, col_name in enumerate(feature_names):
            with cols_ui[idx % 3]:
                if col_name in num_cols:
                    default_val = float(X_sample[col_name].iloc[0]) if col_name in X_sample else 0.0
                    input_data[col_name] = st.number_input(f"{col_name}", value=default_val)
                else:
                    input_data[col_name] = st.text_input(f"{col_name}", value="missing")

        if st.button("🔮 Dự báo ngay"):
            input_df = pd.DataFrame([input_data])
            pred_val = model.predict(input_df)[0]
            st.success(f"🌾 Kết quả dự báo Năng suất: **{pred_val:.4f}**")