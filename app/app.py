import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
from scipy import signal

st.set_page_config(layout="wide", page_title="Global Happiness Explorer")

st.title("Global Happiness Explorer")

@st.cache_data
def load_data(path="data/Happy_Updated.csv"):
    df = pd.read_csv(path, na_values=["", "NA"], encoding="latin-1")
    # 清理末尾空行
    df = df.dropna(subset=["Ladder_score","LGDP","Support","HLE","Freedom","Corruption"])
    df['Continent'] = df['Continent'].astype('category')
    return df

df = load_data()

st.sidebar.header("数据与模型设置")
st.sidebar.write(f"样本数: {df.shape[0]}；变量: {', '.join(df.columns)}")

response = st.sidebar.selectbox("选择响应变量 (Y)", ["Ladder_score"])
predictors_all = ["LGDP","Support","HLE","Freedom","Corruption"]
preds = st.sidebar.multiselect("选择自变量 (X)", predictors_all, default=predictors_all)

model_type = st.sidebar.selectbox("选择模型", ["线性回归 (OLS)", "岭回归 (Ridge)", "LASSO"])
test_size = st.sidebar.slider("测试集比例", 0.1, 0.5, 0.2)
random_state = st.sidebar.number_input("随机种子", value=42, step=1)

# 正则化参数
if model_type in ["岭回归 (Ridge)", "LASSO"]:
    alpha = st.sidebar.slider("正则化强度 (alpha)", 0.001, 5.0, 1.0)
else:
    alpha = 0.0

st.sidebar.markdown("---")
var_for_spectrum = st.sidebar.selectbox("频谱/频率图（选择变量）", ["Ladder_score"] + predictors_all, index=0)
show_corr = st.sidebar.checkbox("显示相关矩阵", value=True)

st.header("数据探索")
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("数据表（前20行）")
    st.dataframe(df.head(20))

with col2:
    st.subheader("基础统计")
    st.write(df[predictors_all + [response]].describe().T)

if show_corr:
    st.subheader("相关矩阵")
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(df[predictors_all + [response]].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

st.header("建模与评估")
if len(preds) == 0:
    st.warning("请至少选择一个自变量进行建模。")
else:
    X = pd.get_dummies(df[preds + ['Continent']], drop_first=True) if 'Continent' in preds else df[preds]
    y = df[response]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))
    if model_type == "线性回归 (OLS)":
        model = LinearRegression()
    elif model_type == "岭回归 (Ridge)":
        model = Ridge(alpha=alpha)
    else:
        model = Lasso(alpha=alpha)

    model.fit(X_train, y_train)
    preds_test = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds_test))
    mae = mean_absolute_error(y_test, preds_test)
    r2 = r2_score(y_test, preds_test)

    st.subheader("测试集评估")
    st.markdown(f"- RMSE: **{rmse:.3f}**  - MAE: **{mae:.3f}**  - R2: **{r2:.3f}**")

    st.subheader("系数")
    coef_df = pd.DataFrame({
        "term": X.columns,
        "estimate": model.coef_
    }).sort_values(by="estimate", key=lambda s: np.abs(s), ascending=False)
    st.dataframe(coef_df)

    st.subheader("拟合图与残差")
    fig2, ax2 = plt.subplots(1,2, figsize=(12,4))
    ax = ax2[0]
    ax.scatter(y_test, preds_test, alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("真实值")
    ax.set_ylabel("预测值")
    ax.set_title("真实 vs 预测")

    ax = ax2[1]
    ax.scatter(preds_test, y_test - preds_test, alpha=0.7)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel("预测值")
    ax.set_ylabel("残差")
    ax.set_title("残差 vs 预测")
    st.pyplot(fig2)

st.header("频谱图 （FFT / Periodogram）")
series = df[var_for_spectrum].dropna().values
# 若样本不规则/无时序意义，我们仍可展示变量数值在频域的频谱以供探索
freqs, pxx = signal.periodogram(series)
fig3, ax3 = plt.subplots()
ax3.semilogy(freqs, pxx)
ax3.set_xlabel("Frequency")
ax3.set_ylabel("Power spectral density")
ax3.set_title(f"Periodogram: {var_for_spectrum}")
st.pyplot(fig3)

st.markdown("注：本数据为横截面（各国）数据，频谱图更多用于探索数值分布中的频率成分，若你有时间序列数据（跨年），频谱图的解释会更有意义。")