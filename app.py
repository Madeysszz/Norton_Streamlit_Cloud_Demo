import pandas as pd
import plotly.express as px
import streamlit as st

from demo_data import build_demo_data


COLORS = {"retail": "#B3652A", "cobrandgen": "#225F86"}
MODELS = {"Konsensus": "consensus_score", "Isolation Forest": "if_percentile",
          "ECOD": "ecod_percentile", "LOF": "lof_percentile"}


@st.cache_data
def load_data():
    data, _ = build_demo_data(seed=2026)
    return data


def chart(figure):
    figure.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(figure, width="stretch")


@st.cache_data
def read_table(name):
    _, tables = build_demo_data(seed=2026)
    return tables.get(name, pd.DataFrame())


def overview(data):
    st.header("Executive Overview")
    left, middle, right = st.columns(3)
    left.metric("Log teramati", f"{len(data):,}")
    middle.metric("Total nilai demonstrasi", f"{data.amount.sum():,.0f}")
    right.metric("Akun demo", f"{data.account_key.nunique():,}")
    st.caption("Seluruh nilai memakai unit demonstrasi fiktif. Akun, status, waktu, dan nilai dibuat secara terprogram untuk contoh tampilan.")
    left, right = st.columns(2)
    with left:
        counts = data.groupby("channel").size().rename("log_count").reset_index()
        chart(px.bar(counts, x="channel", y="log_count", color="channel", color_discrete_map=COLORS,
                     title="Distribusi log antar kanal", labels={"channel": "Kanal", "log_count": "Jumlah log"}))
    with right:
        statuses = data.groupby(["channel", "status_snapshot"]).size().rename("log_count").reset_index()
        chart(px.bar(statuses, x="status_snapshot", y="log_count", color="channel", barmode="group",
                     color_discrete_map=COLORS, title="Status pada snapshot ekspor",
                     labels={"status_snapshot": "Status", "log_count": "Jumlah log"}))
    daily = data.assign(date_utc=data.created_at.dt.floor("D")).groupby(["date_utc", "channel"]).size().rename("log_count").reset_index()
    chart(px.line(daily, x="date_utc", y="log_count", color="channel", markers=True,
                  color_discrete_map=COLORS, title="Log harian yang teramati",
                  labels={"date_utc": "Tanggal UTC", "log_count": "Jumlah log"}))
    st.caption("Grafik menunjukkan aktivitas sintetis pada Januari 2030. Status merupakan kategori demonstrasi; data ini tidak menggambarkan aktivitas pelanggan perusahaan.")


def monitoring(data, score_column, threshold):
    st.header("Anomaly Monitoring")
    flagged = data[data[score_column].ge(threshold)]
    left, middle, right = st.columns(3)
    left.metric("Log melewati ambang", f"{len(flagged):,}")
    middle.metric("Proporsi pada filter", f"{len(flagged) / len(data):.2%}")
    right.metric("Ambang skor", f"{threshold:.2f}")
    st.caption("Skor IF, ECOD, dan LOF pada demo merupakan simulasi, tanpa pelatihan model. Konsensus adalah rerata tiga persentil simulasi dalam kanal, sebelum filter dashboard.")
    chart(px.histogram(data, x=score_column, color="channel", nbins=40, barmode="overlay", opacity=.65,
                       color_discrete_map=COLORS, title="Distribusi skor investigasi",
                       labels={score_column: "Skor dalam kanal"}))
    daily = flagged.assign(date_utc=flagged.created_at.dt.floor("D")).groupby(["date_utc", "channel"]).size().rename("flagged_count").reset_index()
    chart(px.bar(daily, x="date_utc", y="flagged_count", color="channel", color_discrete_map=COLORS,
                 title="Log melewati ambang menurut waktu", labels={"date_utc": "Tanggal UTC", "flagged_count": "Jumlah log"}))
    counts = data.groupby("channel").size().rename("log_count").to_frame()
    counts["flagged_count"] = flagged.groupby("channel").size().reindex(counts.index, fill_value=0)
    counts["flagged_rate"] = counts.flagged_count / counts.log_count
    st.dataframe(counts.reset_index(), hide_index=True, width="stretch")
    st.subheader("Alasan statistik pada kasus berperingkat tinggi")
    st.dataframe(flagged.sort_values(score_column, ascending=False)[["log_key", "channel", score_column, "reasons"]].head(20),
                 hide_index=True, width="stretch")


def comparison(data, selected_channels):
    st.header("Model Comparison")
    st.info("Tabel dihitung dari skor simulasi pada seluruh data demo kanal terpilih. Agreement menunjukkan kesamaan peringkat simulasi. Tidak ada eksperimen model Norton atau pengukuran performa aktual dalam demo ini.")
    long = data.melt(id_vars=["channel"], value_vars=["if_percentile", "ecod_percentile", "lof_percentile"],
                     var_name="model", value_name="score")
    chart(px.histogram(long, x="score", color="model", facet_col="channel", nbins=30, barmode="overlay", opacity=.55,
                       title="Persentil model pada log yang difilter", labels={"score": "Persentil dalam kanal", "model": "Model"}))
    for name, title in [("model_agreement", "Top-K overlap, Jaccard, dan korelasi peringkat"),
                        ("model_stability", "Stabilitas seed, parameter, dan sampel"),
                        ("threshold_sensitivity", "Sensitivitas ambang"),
                        ("experiment_parameters", "Konfigurasi simulasi")]:
        st.subheader(title)
        table = read_table(name)
        if "channel" in table:
            table = table[table.channel.isin(selected_channels)]
        if table.empty:
            st.caption("Uji stabilitas model tidak dijalankan pada demo ini." if name == "model_stability" else "Tidak ada ringkasan simulasi untuk pilihan ini.")
        else:
            st.dataframe(table, hide_index=True, width="stretch")
    st.caption("Konfigurasi menandai seluruh skor sebagai simulated_scores dan trained=False. Kuota investigasi hanya memperagakan cara kerja antrean.")


def investigation(data, all_data, score_column):
    st.header("Transaction Investigation")
    st.caption("Unit investigasi adalah satu log sintetis. Alasan dan bukti dibuat untuk memperagakan interaksi investigasi; tidak ada data pelanggan asli.")
    ranked = data.sort_values([score_column, "log_key"], ascending=[False, True])
    columns = ["log_key", "channel", "created_at", "status_snapshot", "amount", score_column,
               "anomaly_rank", "investigation_priority", "reasons"]
    st.dataframe(ranked[columns].head(500), hide_index=True, width="stretch")
    st.caption("Tabel menampilkan maksimal 500 log sesuai skor terpilih. anomaly_rank dan investigation_priority mengikuti konsensus snapshot di dalam kanal.")
    selection = st.selectbox("Pilih log untuk melihat bukti", ranked.log_key.head(500).tolist())
    case = ranked[ranked.log_key.eq(selection)].iloc[0]
    st.write(f"Akun pseudonim: {case.account_key}")
    st.write(f"Alasan: {case.reasons}")
    st.write(f"Bukti: {case.supporting_evidence}")
    context = all_data[all_data.account_key.eq(case.account_key)].sort_values("created_at")
    chart(px.scatter(context, x="created_at", y="amount", color="status_snapshot", hover_data=["log_key"],
                     title="Konteks seluruh log akun pada snapshot",
                     labels={"created_at": "Waktu UTC", "amount": "Nilai tercatat", "status_snapshot": "Status snapshot"}))
    evidence = ["log_key", "created_at", "amount", "status_snapshot", "prior_history_count",
                "seconds_since_previous_log", "prior_channel_median_amount"]
    st.dataframe(context[evidence], hide_index=True, width="stretch")
    st.caption("Konteks akun demo memakai seluruh periode data sintetis agar riwayat tidak terpotong oleh filter.")


def main():
    st.set_page_config(page_title="Norton — Demo Deployment", layout="wide", initial_sidebar_state="expanded")
    st.title("Norton — Demo Deployment")
    st.info("Demo publik menggunakan data sintetis yang dibuat secara terprogram. Skor berlabel IF, ECOD, dan LOF disimulasikan; bukan hasil pelatihan atau performa aktual model Norton.")
    data = load_data()
    if data.empty:
        st.error("Data demonstrasi belum tersedia. Silakan muat ulang aplikasi.")
        st.stop()
    page = st.sidebar.radio("Halaman", ["Executive Overview", "Anomaly Monitoring", "Model Comparison", "Transaction Investigation"])
    selected_channels = st.sidebar.multiselect("Kanal", sorted(data.channel.unique()), default=sorted(data.channel.unique()))
    selected_statuses = st.sidebar.multiselect("Status snapshot", sorted(data.status_snapshot.unique()), default=sorted(data.status_snapshot.unique()))
    date_range = st.sidebar.date_input("Periode UTC", value=(data.created_at.min().date(), data.created_at.max().date()),
                                      min_value=data.created_at.min().date(), max_value=data.created_at.max().date())
    model = st.sidebar.selectbox("Skor investigasi", list(MODELS))
    threshold = st.sidebar.slider("Ambang skor investigasi", .50, 1.00, .99, .005)
    st.sidebar.caption("Data dan identifier seluruhnya sintetis. Prioritas hanya contoh antrean peninjauan; demo tidak membuktikan fraud atau akurasi model.")
    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        st.info("Pilih tanggal awal dan akhir.")
        st.stop()
    mask = (data.channel.isin(selected_channels) & data.status_snapshot.isin(selected_statuses)
            & data.created_at.dt.date.ge(date_range[0]) & data.created_at.dt.date.le(date_range[1]))
    filtered = data[mask]
    if filtered.empty:
        st.info("Tidak ada log pada kombinasi filter ini.")
        st.stop()
    score_column = MODELS[model]
    if page == "Executive Overview":
        overview(filtered)
    elif page == "Anomaly Monitoring":
        monitoring(filtered, score_column, threshold)
    elif page == "Model Comparison":
        comparison(filtered, selected_channels)
    else:
        investigation(filtered, data, score_column)


if __name__ == "__main__":
    main()
