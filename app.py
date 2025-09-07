import itertools
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# --------------------------
# DỮ LIỆU ĐẦU VÀO CỐ ĐỊNH
# --------------------------
lottery_results = [
    [7, 9, 10, 16, 19, 9],
    [6, 12, 14, 18, 25, 8],
    [1, 12, 13, 27, 28, 7],
    [2, 8, 16, 23, 24, 9],
    [5, 18, 20, 30, 34, 7],
    [1, 18, 25, 27, 33, 5],
    [13, 20, 23, 27, 35, 1],
    [9, 11, 27, 29, 35, 9],
    [10, 23, 24, 30, 33, 6],
    [28, 32, 33, 34, 35, 8],
    [1, 3, 7, 10, 24, 8],
    [3, 18, 20, 29, 34, 3],
    [5, 14, 15, 23, 34, 5],
    [3, 13, 19, 25, 29, 7],
    [3, 11, 12, 27, 32, 10],
    [6, 7, 13, 28, 29, 7],
    [6, 7, 10, 22, 34, 8],
    [1, 8, 14, 19, 23, 6],
    [14, 15, 17, 28, 31, 8],
    [2, 6, 20, 26, 32, 12],
    [2, 7, 9, 28, 31, 12],
    [1, 2, 12, 25, 31, 12],
    [11, 17, 24, 28, 30, 5],
    [9, 24, 29, 32, 35, 8],
    [8, 11, 17, 18, 20, 1],
    [1, 5, 8, 20, 24, 8],
    [5, 11, 16, 20, 34, 6],
    [5, 14, 26, 31, 35, 7],
    [12, 14, 19, 27, 34, 8],
    [13, 15, 19, 22, 27, 10],
    [2, 3, 12, 14, 22, 3],
    [7, 8, 22, 23, 28, 5],
    [9, 12, 22, 31, 35, 2],
    [3, 7, 12, 19, 26, 6],
    [7, 11, 13, 17, 25, 9],
    [2, 10, 20, 34, 35, 2],
    [7, 11, 17, 18, 25, 1],
    [1, 4, 8, 18, 29, 7],
    [5, 11, 16, 27, 34, 9],
    [2, 10, 12, 16, 34, 2],
    [5, 19, 20, 21, 30, 11],
    [4, 16, 24, 26, 31, 11],
    [21, 28, 31, 34, 35, 1],
    [9, 14, 28, 33, 35, 10],
    [8, 20, 24, 27, 28, 3],
    [2, 16, 18, 22, 30, 5],
    [7, 8, 18, 22, 33, 7],
    [9, 10, 19, 22, 24, 7],
    [16, 18, 19, 23, 25, 11],
    [1, 14, 16, 21, 25, 5],
    [17, 18, 27, 29, 33, 8],
    [3, 21, 28, 33, 34, 5],
    [9, 10, 11, 30, 35, 1],
    [7, 8, 18, 22, 33, 7],
    [9, 10, 19, 22, 24, 7],
    [16, 18, 19, 23, 25, 11],
    [3, 6, 11, 27, 33, 6],
    [1, 7, 12, 16, 23, 8],
    [1, 14, 19, 25, 27, 8],
    [10, 24, 28, 30, 32, 1],
    [1, 3, 28, 32, 33, 11],
    [15, 19, 25, 32, 33, 5],
    [1, 5, 16, 28, 29, 8],
    [20, 21, 29, 31, 33, 3]
]
st.subheader("📥 Nhập Kết quả kỳ mới nhất")

new_result_input = st.text_input("Nhập 6 số, cách nhau bằng dấu cách hoặc dấu '-':", "")

if new_result_input:
    try:
        new_result = [int(x) for x in new_result_input.replace('-', ' ').split()]
        if len(new_result) != 6:
            st.error("Vui lòng nhập đúng 6 số!")
        else:
            if new_result not in lottery_results:
                lottery_results.append(new_result)
                st.success(f"Đã thêm kết quả mới: {new_result}")
            else:
                st.warning("Kết quả này đã tồn tại trong dữ liệu.")
    except Exception as e:
        st.error(f"Lỗi khi xử lý input: {e}")
# --------------------------
# STREAMLIT GIAO DIỆN
# --------------------------
st.set_page_config(page_title="Dự đoán xổ số 5/35", layout="centered")
st.title("🎯 Dự đoán xổ số 5/35 bằng Machine Learning")

if st.button("🚀 Huấn luyện & Dự đoán kỳ tiếp theo"):
    # Chuẩn bị dữ liệu đầu vào, đầu ra
    data = [entry[:5] for entry in lottery_results]  # 5 số đầu là số chính

    mlb = MultiLabelBinarizer(classes=range(1, 36))  # Số từ 1 đến 35
    X = []
    y = []

    for i in range(len(data) - 1):
        X.append(data[i])
        y.append(data[i + 1])

    X_bin = mlb.fit_transform(X)
    y_bin = mlb.transform(y)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_bin, y_bin)

    # Dự đoán xác suất cho bộ số mới nhất
    last = mlb.transform([data[-1]])
    pred_prob = model.predict_proba(last)

    if isinstance(pred_prob, list):
    # mỗi p có shape (1, 2), p[0,1] là xác suất nhãn 1 cho từng nhãn
        probs = np.array([p[0,1] for p in pred_prob])  # lấy xác suất nhãn 1 cho từng số
        avg_prob = probs
    else:
        avg_prob = pred_prob[0]

    # Top 10 số có xác suất cao nhất
    top_n = min(10, len(avg_prob))
    top_indices = np.argsort(avg_prob)[-top_n:][::-1]
    predicted_numbers = [i + 1 for i in top_indices]

    st.subheader("🎉 Bộ số dự đoán (Top 10 số):")
    st.success(" - ".join(str(num) for num in predicted_numbers))

    # Hiển thị bảng xác suất top 10 số
    prob_table = [(i+1, round(avg_prob[i]*100, 2)) for i in top_indices]
    st.markdown("### 📊 Xác suất top 10 số:")
    st.table(pd.DataFrame(prob_table, columns=["Số", "Xác suất (%)"]))

    # --- Sinh các tổ hợp 6 số từ top 10 số ---
    combinations = list(itertools.combinations(predicted_numbers, 6))

    def combination_prob(comb, prob_dict):
        return sum(prob_dict[num] for num in comb)

    prob_dict = {num: avg_prob[num-1] for num in predicted_numbers}

    comb_probs = [(comb, combination_prob(comb, prob_dict)) for comb in combinations]

    # Hàm kiểm tra điều kiện chẵn/lẻ, số bé/lớn
    def is_valid_combination(comb):
        even_count = sum(1 for x in comb if x % 2 == 0)
        odd_count = 6 - even_count
        small_count = sum(1 for x in comb if x <= 18)
        large_count = 6 - small_count

        # Yêu cầu tối thiểu 2 số chẵn, 2 số lẻ, 2 số nhỏ, 2 số lớn
        return even_count >= 2 and odd_count >= 2 and small_count >= 2 and large_count >= 2

    valid_combs = [c for c in comb_probs if is_valid_combination(c[0])]

    # Sắp xếp và lấy 5 bộ số có tổng xác suất cao nhất
    valid_combs_sorted = sorted(valid_combs, key=lambda x: x[1], reverse=True)
    top_bets = valid_combs_sorted[:5]

    st.subheader("🎯 Các bộ số 6 số đề xuất để mua:")
    for comb, prob in top_bets:
        st.write(f"{' - '.join(map(str, comb))}  (Tổng xác suất: {round(prob*100, 2)}%)")

