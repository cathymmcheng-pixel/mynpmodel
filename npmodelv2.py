
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
from io import BytesIO

# ==========================================
# 0. 核心配置与常量
# ==========================================

st.set_page_config(layout="wide", page_title="赫双妥Model管理看板")

# 2025年 (平年)
DAYS_2025 = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
# 2026年 (平年)
DAYS_2026 = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# 合并两年的天数列表 (用于24个月的连续模拟)
DAYS_COMBINED = DAYS_2025 + DAYS_2026
MONTH_NAMES_25 = [f"25年{i}月" for i in range(1, 13)]
MONTH_NAMES_26 = [f"26年{i}月" for i in range(1, 13)]
MONTH_NAMES_COMBINED = MONTH_NAMES_25 + MONTH_NAMES_26
TOTAL_DAYS_2YEARS = sum(DAYS_COMBINED) # 730天
CYCLE_DAYS = 21

def get_month_ranges(days_list):
    """计算每个月在时间轴上的起始和结束天数索引"""
    starts = [0]
    for d in days_list[:-1]:
        starts.append(starts[-1] + d)
    ranges = []
    for i, start in enumerate(starts):
        ranges.append((start, start + days_list[i]))
    return ranges

MONTH_RANGES_25 = get_month_ranges(DAYS_2025)
MONTH_RANGES_COMBINED = get_month_ranges(DAYS_COMBINED)

# ==========================================
# 1. 核心算法逻辑
# ==========================================

def simulate_sales_continuous(pure_new_list, trans_new_list, X, Y, days_config=DAYS_COMBINED, month_ranges=MONTH_RANGES_COMBINED):
    """
    通用连续模拟函数：支持12个月或24个月
    """
    total_days = sum(days_config)
    # 初始化每日销量数组
    daily_big = np.zeros(total_days + 200)
    daily_small = np.zeros(total_days + 200)

    limit_months = len(pure_new_list)

    # --- 纯新患者逻辑 ---
    for m_idx in range(limit_months):
        count = pure_new_list[m_idx]
        if pd.isna(count): count = 0

        days_in_m = days_config[m_idx]
        start_day_m = month_ranges[m_idx][0]
        daily_inflow = count / days_in_m

        for d in range(days_in_m):
            entry_day = start_day_m + d

            # Day 0: 大支
            if entry_day < total_days:
                daily_big[entry_day] += daily_inflow

            # Day 21...: 小支
            full_doses = int(np.floor(X).item() if hasattr(X, 'item') else np.floor(X))
            remainder = X - full_doses
            first_small_day = entry_day + CYCLE_DAYS

            for k in range(full_doses):
                dose_day = first_small_day + k * CYCLE_DAYS
                if dose_day < total_days:
                    daily_small[dose_day] += daily_inflow
            if remainder > 0:
                dose_day = first_small_day + full_doses * CYCLE_DAYS
                if dose_day < total_days:
                    daily_small[dose_day] += daily_inflow * remainder

    # --- 转新患者逻辑 ---
    for m_idx in range(limit_months):
        count = trans_new_list[m_idx]
        if pd.isna(count): count = 0

        days_in_m = days_config[m_idx]
        start_day_m = month_ranges[m_idx][0]
        daily_inflow = count / days_in_m

        for d in range(days_in_m):
            entry_day = start_day_m + d

            full_doses = int(np.floor(Y).item() if hasattr(Y, 'item') else np.floor(Y))
            remainder = Y - full_doses
            first_small_day = entry_day 

            for k in range(full_doses):
                dose_day = first_small_day + k * CYCLE_DAYS
                if dose_day < total_days:
                    daily_small[dose_day] += daily_inflow
            if remainder > 0:
                dose_day = first_small_day + full_doses * CYCLE_DAYS
                if dose_day < total_days:
                    daily_small[dose_day] += daily_inflow * remainder

    # 汇总月度数据
    monthly_big = []
    monthly_small = []
    for start, end in month_ranges:
        monthly_big.append(np.sum(daily_big[start:end]))
        monthly_small.append(np.sum(daily_small[start:end]))

    return np.array(monthly_big), np.array(monthly_small)

def simulate_separated_continuous(pure_list, trans_list, X, Y, days_config, month_ranges):
    zeros = np.zeros(len(pure_list))
    b_pure, s_pure = simulate_sales_continuous(pure_list, zeros, X, 0, days_config, month_ranges)
    _, s_trans = simulate_sales_continuous(zeros, trans_list, 0, Y, days_config, month_ranges)
    return b_pure, s_pure, s_trans

# 计算准确度辅助函数
def calculate_metrics(actual, predicted):
    mask = ~np.isnan(actual)
    if np.sum(mask) == 0: return 0, 0
    act_filtered = actual[mask]
    pred_filtered = predicted[mask]
    total_diff = np.sum(pred_filtered - act_filtered)
    ss_res = np.sum((act_filtered - pred_filtered) ** 2)
    ss_tot = np.sum((act_filtered - np.mean(act_filtered)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    return total_diff, r2

# ==========================================
# 2. 界面 - 侧边栏与头部
# ==========================================

with st.sidebar:
    st.header("操作面板")
    uploaded_file = st.file_uploader("上传CSV销量数据", type=["csv"])
    st.markdown("---")
    st.markdown("**说明**：请确保上传的CSV包含以下行名：\n- `纯新患者数`\n- `转新患者数`\n- `赫大支实际纯销`\n- `赫小支实际纯销`\n\n列名为 `1月` 至 `12月`。")

st.title("📊 赫双妥 Model 管理看板")

# ==========================================
# 3. 2025年拟合分析
# ==========================================

st.markdown("---")
st.header("✨ 2025年拟合分析")
st.markdown("---")

st.header("A. 模型基础与数据导入")
with st.expander("查看算法模型基础假设", expanded=False):
    st.markdown("""
    **产品规格：**
    * **赫大支 (15ml)**：用于患者首次使用。
    * **赫小支 (10ml)**：用于患者维持治疗。
    * **周期**：均为 21 天。

    **患者行为逻辑：**
    1.  **纯新患者 (Pure New)**：Day 0 贡献 1 大支; Day 21 起贡献小支 (上限 X)。
    2.  **转新患者 (Transferred New)**：Day 0 起贡献小支 (上限 Y)。
    """)

# 全局变量初始化
pure_new_25 = np.zeros(12)
trans_new_25 = np.zeros(12)
actual_big_25 = np.zeros(12)
actual_small_25 = np.zeros(12)
data_loaded = False

if uploaded_file is not None:
    try:
        try:
            df = pd.read_csv(uploaded_file, index_col=0)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, index_col=0, encoding='gbk')

        cols = [c for c in df.columns if "月" in c]
        if len(cols) == 12:
            pure_new_25 = df.loc['纯新患者数', cols].values.astype(float)
            trans_new_25 = df.loc['转新患者数', cols].values.astype(float)
            actual_big_25 = df.loc['赫大支实际纯销', cols].values.astype(float)
            actual_small_25 = df.loc['赫小支实际纯销', cols].values.astype(float)
            data_loaded = True
            st.success("✅ 2025 数据加载成功")
            with st.expander("查看原始数据"):
                st.dataframe(df.style.format("{:.0f}"))
        else:
            st.error("CSV文件格式错误：未找到12个月份列")
    except Exception as e:
        st.error(f"读取文件出错: {e}")
else:
    st.info("请先上传数据文件。")

# --- 中屏 (B) ---
st.header("B. 智能拟合 (Fitting)")

col_b1, col_b2 = st.columns([1, 2])
best_x_fit = 7.0
best_y_fit = 4.2

if data_loaded:
    with col_b1:
        st.subheader("参数设置")
        ratio_input = st.number_input("输入 Y 与 X 的关系 (Y = ? % of X)", min_value=10.0, max_value=200.0, value=60.0, step=5.0)
        ratio = ratio_input / 100.0
        fit_mode = st.radio("选择拟合目标", ("全年总量拟合最准", "全年趋势拟合最准"))

        if st.button("开始拟合求解"):
            with st.spinner("正在寻找最佳参数..."):
                def objective(x_val):
                    y_val = x_val * ratio
                    # 仅跑2025年12个月
                    _, pred_small = simulate_sales_continuous(
                        pure_new_25, trans_new_25, x_val, y_val, 
                        days_config=DAYS_2025, month_ranges=MONTH_RANGES_25
                    )

                    if fit_mode == "全年总量拟合最准":
                        return abs(np.sum(pred_small) - np.sum(actual_small_25))
                    else:
                        return np.sum((pred_small - actual_small_25) ** 2)

                res = minimize(objective, x0=10.0, bounds=[(0.0, 50.0)], method='L-BFGS-B')
                best_x_fit = res.x[0]
                best_y_fit = best_x_fit * ratio

                st.session_state['fit_x'] = best_x_fit
                st.session_state['fit_y'] = best_y_fit

    if 'fit_x' in st.session_state:
        best_x_fit = st.session_state['fit_x']
        best_y_fit = st.session_state['fit_y']

        # 拟合结果展示
        _, fit_pred_small = simulate_sales_continuous(
            pure_new_25, trans_new_25, best_x_fit, best_y_fit, 
            days_config=DAYS_2025, month_ranges=MONTH_RANGES_25
        )
        _, r2_s = calculate_metrics(actual_small_25, fit_pred_small)

        with col_b2:
            st.subheader("拟合结果")
            st.markdown(f"**最佳 X:** `{best_x_fit:.4f}` | **最佳 Y:** `{best_y_fit:.4f}` | **R²:** `{r2_s:.4f}`")
            fig_fit = go.Figure()
            fig_fit.add_trace(go.Scatter(x=MONTH_NAMES_25, y=actual_small_25, name='实际', line=dict(color='blue')))
            fig_fit.add_trace(go.Scatter(x=MONTH_NAMES_25, y=fit_pred_small, name='预测', line=dict(color='orange', dash='dash')))
            fig_fit.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_fit, use_container_width=True)

# --- 下屏 (C) ---
st.header("C. 2025年详细报表")

col_c1, col_c2 = st.columns(2)
with col_c1:
    user_x_25 = st.number_input("2025年 X (纯新小支数)", value=best_x_fit, format="%.2f", key="x_25")
with col_c2:
    user_y_25 = st.number_input("2025年 Y (转新小支数)", value=best_y_fit, format="%.2f", key="y_25")

if data_loaded:
    # 1. 计算2025数据 (仅12个月)
    b_pure_25, s_pure_25, s_trans_25 = simulate_separated_continuous(
        pure_new_25, trans_new_25, user_x_25, user_y_25,
        days_config=DAYS_2025, month_ranges=MONTH_RANGES_25
    )

    # 2. 构建2025报表
    with np.errstate(divide='ignore', invalid='ignore'):
        # 准确度计算
        acc_big_m_25 = b_pure_25 / actual_big_25
        acc_big_c_25 = np.cumsum(b_pure_25) / np.cumsum(np.nan_to_num(actual_big_25))

        s_total_25 = s_pure_25 + s_trans_25
        acc_small_m_25 = s_total_25 / actual_small_25
        acc_small_c_25 = np.cumsum(s_total_25) / np.cumsum(np.nan_to_num(actual_small_25))

    df_25_data = {
        '纯新患者数': pure_new_25,
        '纯新贡献赫大支': b_pure_25,
        '纯新贡献赫小支': s_pure_25,
        '转新患者数': trans_new_25,
        '转新贡献赫小支': s_trans_25,
        '赫大支预测总计': b_pure_25,
        '赫大支实际纯销': actual_big_25,
        '赫大支差值': b_pure_25 - actual_big_25,
        '赫大支当月预测准确度': acc_big_m_25,
        '赫大支累计预测准确度': acc_big_c_25,
        '赫小支预测总计': s_total_25,
        '赫小支实际纯销': actual_small_25,
        '赫小支差值': s_total_25 - actual_small_25,
        '赫小支当月预测准确度': acc_small_m_25,
        '赫小支累计预测准确度': acc_small_c_25
    }

    df_25 = pd.DataFrame(df_25_data, index=MONTH_NAMES_25).T

    # 3. 添加Total列 (针对2025)
    df_25['Y25全年total'] = df_25.sum(axis=1)
    # 修正Total准确度
    t_pred_b = df_25.loc['赫大支预测总计', 'Y25全年total']
    t_act_b = df_25.loc['赫大支实际纯销', 'Y25全年total']
    t_acc_b = t_pred_b / t_act_b if t_act_b != 0 else 0
    df_25.loc['赫大支当月预测准确度', 'Y25全年total'] = t_acc_b
    df_25.loc['赫大支累计预测准确度', 'Y25全年total'] = t_acc_b

    t_pred_s = df_25.loc['赫小支预测总计', 'Y25全年total']
    t_act_s = df_25.loc['赫小支实际纯销', 'Y25全年total']
    t_acc_s = t_pred_s / t_act_s if t_act_s != 0 else 0
    df_25.loc['赫小支当月预测准确度', 'Y25全年total'] = t_acc_s
    df_25.loc['赫小支累计预测准确度', 'Y25全年total'] = t_acc_s

    # 4. 展示样式
    acc_rows = ['赫大支当月预测准确度', '赫大支累计预测准确度', '赫小支当月预测准确度', '赫小支累计预测准确度']
    diff_rows = ['赫大支差值', '赫小支差值']

    def style_excel(df):
        return df.style\
            .format("{:.1f}")\
            .format("{:.1%}", subset=pd.IndexSlice[acc_rows, :], na_rep="-")\
            .format("{:.0f}", subset=pd.IndexSlice[~df.index.isin(acc_rows), :], na_rep="-")\
            .background_gradient(cmap="RdBu", axis=1, subset=pd.IndexSlice[diff_rows, :])\
            .apply(lambda s: ['color: blue; font-weight: bold' if s.name in acc_rows else '' for _ in s], axis=1)

    st.subheader("2025年 数据表")
    st.dataframe(style_excel(df_25))

    # 5. 可视化 (仅2025)
    v25_c1, v25_c2 = st.columns(2)
    with v25_c1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=MONTH_NAMES_25, y=actual_big_25, name='实际纯销', mode='lines+markers'))
        fig1.add_trace(go.Scatter(x=MONTH_NAMES_25, y=b_pure_25, name='预测纯销', mode='lines+markers', line=dict(dash='dash')))
        fig1.add_bar(x=MONTH_NAMES_25, y=b_pure_25 - actual_big_25, name='差值', marker_color='gray', opacity=0.3)
        fig1.update_layout(title="图一：2025 赫大支 差异分析")
        st.plotly_chart(fig1, use_container_width=True)

    with v25_c2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=MONTH_NAMES_25, y=actual_small_25, name='实际纯销', mode='lines+markers'))
        fig2.add_trace(go.Scatter(x=MONTH_NAMES_25, y=s_total_25, name='预测纯销', mode='lines+markers', line=dict(dash='dash')))
        fig2.add_bar(x=MONTH_NAMES_25, y=s_total_25 - actual_small_25, name='差值', marker_color='gray', opacity=0.3)
        fig2.update_layout(title="图二：2025 赫小支 差异分析")
        st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# 4. 2026年预测分析
# ==========================================

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.header("🚀 2026年预测分析")
st.markdown("---")

st.header("D. 2026年预测交互入口")

if not data_loaded:
    st.warning("请先在顶部上传2025年历史数据。")
    st.stop()

# --- D屏：参数输入区域 ---
st.subheader("1. 设定预测参数")
col_d1, col_d2 = st.columns(2)

with col_d1:
    user_x_26 = st.number_input("预测 X (25/26年一致)", value=user_x_25, format="%.2f")
with col_d2:
    user_y_26 = st.number_input("预测 Y (25/26年一致)", value=user_y_25, format="%.2f")

# --- 2. 横向输入表格 (转置) ---
st.subheader("2. 输入2026年每月预估数据")
st.caption("提示：您可以直接从Excel复制一整行数据（12个月），然后点击表格中第一个单元格进行粘贴。")

# 创建横向结构：列为月份，行为指标
transposed_data = {
    '指标': ['纯新患者数', '转新患者数', '赫大支实际纯销 (选填)', '赫小支实际纯销 (选填)', '每月销售指标 (金额, 元)'],
}
# 初始化默认值
default_pure = [1500] * 12
default_trans = [500] * 12
default_none = [None] * 12
# 默认销售指标分配 (示例值)
default_target = [160000] * 6 + [250000] * 6 

for i in range(12):
    col_name = f"26年{i+1}月"
    # 构建每一列的数据
    transposed_data[col_name] = [
        default_pure[i],
        default_trans[i],
        default_none[i],
        default_none[i],
        default_target[i]
    ]

df_transposed_template = pd.DataFrame(transposed_data)

# 配置列编辑权限
column_config = {
    '指标': st.column_config.TextColumn(disabled=True, width="medium"),
}
# 设置月份列为数字输入
for i in range(12):
    col_name = f"26年{i+1}月"
    column_config[col_name] = st.column_config.NumberColumn(required=False)

# 展示编辑器
edited_transposed = st.data_editor(
    df_transposed_template,
    column_config=column_config,
    hide_index=True,
    use_container_width=True,
    num_rows="fixed", # 禁止添加行，便于复制粘贴结构稳定
    key="editor_2026"
)

# --- 解析转置后的数据 ---
# 行顺序：0:纯新, 1:转新, 2:大支实际, 3:小支实际, 4:销售指标
pure_new_26 = edited_transposed.iloc[0, 1:].fillna(0).values.astype(float)
trans_new_26 = edited_transposed.iloc[1, 1:].fillna(0).values.astype(float)
actual_big_26 = edited_transposed.iloc[2, 1:].values.astype(float) # 保持None
actual_small_26 = edited_transposed.iloc[3, 1:].values.astype(float) # 保持None
targets_26 = edited_transposed.iloc[4, 1:].fillna(0).values.astype(float)

# --- 核心计算：24个月连续模拟 ---
input_pure_24 = np.concatenate([pure_new_25, pure_new_26])
input_trans_24 = np.concatenate([trans_new_25, trans_new_26])
input_act_big_24 = np.concatenate([actual_big_25, actual_big_26])
input_act_small_24 = np.concatenate([actual_small_25, actual_small_26])

# 预测
pred_big_24, pred_small_24 = simulate_sales_continuous(
    input_pure_24, input_trans_24, user_x_26, user_y_26,
    days_config=DAYS_COMBINED, month_ranges=MONTH_RANGES_COMBINED
)

# 分解贡献
b_pure_24, s_pure_24, s_trans_24 = simulate_separated_continuous(
    input_pure_24, input_trans_24, user_x_26, user_y_26,
    days_config=DAYS_COMBINED, month_ranges=MONTH_RANGES_COMBINED
)

# --- 结果展示 ---
st.subheader("3. 2025-2026 全景数据结果")

# 1. 财务指标
idx_26_start = 12
idx_26_h1_end = 18
idx_26_h2_end = 24

pred_small_h1 = np.sum(pred_small_24[idx_26_start:idx_26_h1_end])
pred_large_h1 = np.sum(pred_big_24[idx_26_start:idx_26_h1_end])
pred_small_h2 = np.sum(pred_small_24[idx_26_h1_end:idx_26_h2_end])
pred_large_h2 = np.sum(pred_big_24[idx_26_h1_end:idx_26_h2_end])

# 考核价
price_large = 8489.27
price_small = 6232.07

# 收入预测
rev_total_h1 = (pred_small_h1 * price_small) + (pred_large_h1 * price_large)
rev_total_h2 = (pred_small_h2 * price_small) + (pred_large_h2 * price_large)

# 从用户输入的行中提取H1和H2的指标总和
target_h1_sum = np.sum(targets_26[0:6])
target_h2_sum = np.sum(targets_26[6:12])

ach_rate_h1 = rev_total_h1 / target_h1_sum if target_h1_sum > 0 else 0
ach_rate_h2 = rev_total_h2 / target_h2_sum if target_h2_sum > 0 else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("2026 H1 预测总收入", f"¥{rev_total_h1/10000:,.1f} 万")
m1.metric("2026 H1 指标完成率", f"{ach_rate_h1:.1%}", help=f"H1指标: ¥{target_h1_sum/10000:.1f}万")
m2.metric("2026 H2 预测总收入", f"¥{rev_total_h2/10000:,.1f} 万")
m2.metric("2026 H2 指标完成率", f"{ach_rate_h2:.1%}", help=f"H2指标: ¥{target_h2_sum/10000:.1f}万")

# 2. 24个月大表
with np.errstate(divide='ignore', invalid='ignore'):
    acc_big_month_24 = b_pure_24 / input_act_big_24
    acc_big_cum_24 = np.cumsum(b_pure_24) / np.cumsum(np.nan_to_num(input_act_big_24))

    s_pred_total_24 = s_pure_24 + s_trans_24
    acc_small_month_24 = s_pred_total_24 / input_act_small_24
    acc_small_cum_24 = np.cumsum(s_pred_total_24) / np.cumsum(np.nan_to_num(input_act_small_24))

# Handle NaN for visualization in table
acc_big_month_24[np.isnan(input_act_big_24)] = np.nan
acc_big_cum_24[np.isnan(input_act_big_24)] = np.nan
acc_small_month_24[np.isnan(input_act_small_24)] = np.nan
acc_small_cum_24[np.isnan(input_act_small_24)] = np.nan

full_df_data = {
    '纯新患者数': input_pure_24,
    '纯新贡献赫大支': b_pure_24,
    '纯新贡献赫小支': s_pure_24,
    '转新患者数': input_trans_24,
    '转新贡献赫小支': s_trans_24,
    '赫大支预测总计': b_pure_24,
    '赫大支实际纯销': input_act_big_24,
    '赫大支差值': b_pure_24 - input_act_big_24,
    '赫大支当月预测准确度': acc_big_month_24,
    '赫大支累计预测准确度': acc_big_cum_24,
    '赫小支预测总计': s_pred_total_24,
    '赫小支实际纯销': input_act_small_24,
    '赫小支差值': s_pred_total_24 - input_act_small_24,
    '赫小支当月预测准确度': acc_small_month_24,
    '赫小支累计预测准确度': acc_small_cum_24
}

df_full = pd.DataFrame(full_df_data, index=MONTH_NAMES_COMBINED).T

# 【修改点】Y26 全年 Total 列计算
# 1. 提取2026年的数据 (后12列)
df_26_part = df_full.iloc[:, 12:24]
# 2. 常规行：直接求和 (2026 Sum)
y26_totals = df_26_part.sum(axis=1)

# 3. 准确度行计算

# A. 当月预测准确度 (Monthly Accuracy) - 保持不变
# 逻辑：Total Pred 2026 / Total Actual 2026
t_pred_b_26 = y26_totals['赫大支预测总计']
t_act_b_26 = y26_totals['赫大支实际纯销']
t_acc_b_monthly = t_pred_b_26 / t_act_b_26 if t_act_b_26 != 0 else 0

t_pred_s_26 = y26_totals['赫小支预测总计']
t_act_s_26 = y26_totals['赫小支实际纯销']
t_acc_s_monthly = t_pred_s_26 / t_act_s_26 if t_act_s_26 != 0 else 0

y26_totals['赫大支当月预测准确度'] = t_acc_b_monthly
y26_totals['赫小支当月预测准确度'] = t_acc_s_monthly

# B. 累计预测准确度 (Cumulative Accuracy) - 【核心修改】
# 逻辑：从2025年1月开始累计求和 (Pred All / Actual All)
total_pred_b_all = df_full.loc['赫大支预测总计'].sum()
total_act_b_all = df_full.loc['赫大支实际纯销'].sum()
t_acc_b_cum = total_pred_b_all / total_act_b_all if total_act_b_all != 0 else 0

total_pred_s_all = df_full.loc['赫小支预测总计'].sum()
total_act_s_all = df_full.loc['赫小支实际纯销'].sum()
t_acc_s_cum = total_pred_s_all / total_act_s_all if total_act_s_all != 0 else 0

y26_totals['赫大支累计预测准确度'] = t_acc_b_cum
y26_totals['赫小支累计预测准确度'] = t_acc_s_cum

# 4. 添加新列
df_full['Y26全年total'] = y26_totals

# 展示表格
st.dataframe(style_excel(df_full))

# 下载
def to_excel_full(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='2025-2026预测')
    return output.getvalue()

st.download_button(
    "📥 下载 2025-2026 完整预测报表",
    data=to_excel_full(df_full),
    file_name="赫双妥_2025_2026_Forecast.xlsx"
)

# 3. 趋势图
st.subheader("4. 趋势可视化 (2025-2026)")
v_col1, v_col2 = st.columns(2)

with v_col1:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=MONTH_NAMES_COMBINED, y=input_act_big_24, name='实际纯销', mode='lines+markers', connectgaps=False))
    fig1.add_trace(go.Scatter(x=MONTH_NAMES_COMBINED, y=b_pure_24, name='预测纯销', mode='lines+markers', line=dict(dash='dash')))
    fig1.add_vline(x=11.5, line_width=1, line_dash="dot", annotation_text="2026 Start")
    fig1.update_layout(title="图一：赫大支 (Large Vial) 预测 vs 实际")
    st.plotly_chart(fig1, use_container_width=True)

with v_col2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=MONTH_NAMES_COMBINED, y=input_act_small_24, name='实际纯销', mode='lines+markers', connectgaps=False))
    fig2.add_trace(go.Scatter(x=MONTH_NAMES_COMBINED, y=s_pred_total_24, name='预测纯销', mode='lines+markers', line=dict(dash='dash')))
    fig2.add_vline(x=11.5, line_width=1, line_dash="dot", annotation_text="2026 Start")
    fig2.update_layout(title="图二：赫小支 (Small Vial) 预测 vs 实际")
    st.plotly_chart(fig2, use_container_width=True)
