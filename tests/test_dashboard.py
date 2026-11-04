from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest


DEMO_ROOT = Path(__file__).resolve().parents[1]
APP = DEMO_ROOT / "app.py"
MODEL_COLUMNS = {
    "Konsensus": "consensus_score", "Isolation Forest": "if_percentile",
    "ECOD": "ecod_percentile", "LOF": "lof_percentile",
}


@pytest.fixture
def demo_generator(monkeypatch):
    monkeypatch.syspath_prepend(str(DEMO_ROOT))
    from demo_data import build_demo_data
    return build_demo_data


@pytest.fixture
def demo_data(demo_generator):
    return demo_generator(seed=2026)[0]


@pytest.fixture
def cloud_dashboard(monkeypatch):
    monkeypatch.syspath_prepend(str(DEMO_ROOT))
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    return app


def test_demo_generator_is_reproducible_for_seed(demo_generator):
    first, first_tables = demo_generator(seed=2026)
    second, second_tables = demo_generator(seed=2026)
    changed, _ = demo_generator(seed=2027)
    pd.testing.assert_frame_equal(first, second)
    assert first_tables.keys() == second_tables.keys()
    for name in first_tables:
        pd.testing.assert_frame_equal(first_tables[name], second_tables[name])
    assert not first.equals(changed)


def test_demo_generator_scores_ranks_and_counts_reconcile(demo_generator):
    data, tables = demo_generator(seed=2026)
    assert len(data) == 128
    assert data.groupby("channel").size().to_dict() == {"cobrandgen": 64, "retail": 64}
    assert data.log_key.is_unique
    assert data.created_at.notna().all()
    assert str(data.created_at.dt.tz) == "UTC"
    numeric = data[["amount", "anomaly_rank", *MODEL_COLUMNS.values()]]
    assert np.isfinite(numeric.to_numpy(dtype=float)).all()
    assert data.amount.ge(0).all()
    assert data[list(MODEL_COLUMNS.values())].ge(0).all().all()
    assert data[list(MODEL_COLUMNS.values())].le(1).all().all()
    assert np.allclose(
        data.consensus_score,
        data[["if_percentile", "ecod_percentile", "lof_percentile"]].mean(axis=1),
    )
    for _, group in data.groupby("channel"):
        assert sorted(group.anomaly_rank.tolist()) == list(range(1, 65))
        assert group.investigation_priority.value_counts().to_dict() == {
            "rendah": 60, "sedang": 3, "tinggi": 1,
        }
        ranked = group.sort_values("anomaly_rank")
        assert ranked.consensus_score.is_monotonic_decreasing
    assert {"model_agreement", "model_stability", "threshold_sensitivity", "experiment_parameters"} <= tables.keys()


@pytest.mark.parametrize("page", [
    "Executive Overview", "Anomaly Monitoring", "Model Comparison", "Transaction Investigation",
])
def test_cloud_demo_pages_render_with_synthetic_disclosure(cloud_dashboard, demo_data, page):
    cloud_dashboard.sidebar.radio[0].set_value(page).run()
    assert not cloud_dashboard.exception
    assert cloud_dashboard.header[0].value == page
    assert any("sintetis" in item.value.lower() for item in cloud_dashboard.info)
    if page == "Executive Overview":
        assert [item.value for item in cloud_dashboard.metric] == [
            f"{len(demo_data):,}", f"{demo_data.amount.sum():,.0f}", f"{demo_data.account_key.nunique():,}",
        ]
    else:
        assert len(cloud_dashboard.dataframe) > 0


def test_cloud_demo_channel_status_and_date_filters_match_generated_data(cloud_dashboard, demo_data):
    channel = "retail"
    expected = demo_data.loc[demo_data.channel.eq(channel)]
    cloud_dashboard.sidebar.multiselect[0].set_value([channel]).run()
    assert not cloud_dashboard.exception
    assert cloud_dashboard.metric[0].value == f"{len(expected):,}"
    status = sorted(expected.status_snapshot.unique())[0]
    expected = expected.loc[expected.status_snapshot.eq(status)]
    cloud_dashboard.sidebar.multiselect[1].set_value([status]).run()
    assert not cloud_dashboard.exception
    assert cloud_dashboard.metric[0].value == f"{len(expected):,}"
    dates = sorted(expected.created_at.dt.date.unique())
    assert len(dates) > 2
    lower, upper = dates[1], dates[-2]
    filtered = expected.loc[expected.created_at.dt.date.between(lower, upper)]
    assert 0 < len(filtered) < len(expected)
    cloud_dashboard.sidebar.date_input[0].set_value((lower, upper)).run()
    assert not cloud_dashboard.exception
    assert cloud_dashboard.metric[0].value == f"{len(filtered):,}"
    assert cloud_dashboard.metric[1].value == f"{filtered.amount.sum():,.0f}"


@pytest.mark.parametrize("model,score_column", MODEL_COLUMNS.items())
def test_cloud_demo_model_and_threshold_use_generated_scores(cloud_dashboard, demo_data, model, score_column):
    cloud_dashboard.sidebar.radio[0].set_value("Anomaly Monitoring").run()
    cloud_dashboard.sidebar.selectbox[0].set_value(model).run()
    cloud_dashboard.sidebar.slider[0].set_value(.8).run()
    assert not cloud_dashboard.exception
    expected = demo_data.loc[demo_data[score_column].ge(.8)]
    assert cloud_dashboard.metric[0].value == f"{len(expected):,}"
    assert cloud_dashboard.metric[1].value == f"{len(expected) / len(demo_data):.2%}"
    cases = cloud_dashboard.dataframe[1].value
    assert score_column in cases.columns
    assert len(cases) == min(20, len(expected))
    assert cases[score_column].ge(.8).all()
    assert set(cases.log_key) <= set(expected.log_key)


def test_cloud_demo_threshold_changes_queue_size(cloud_dashboard, demo_data):
    cloud_dashboard.sidebar.radio[0].set_value("Anomaly Monitoring").run()
    cloud_dashboard.sidebar.selectbox[0].set_value("Isolation Forest").run()
    counts = []
    for threshold in [.5, 1.0]:
        cloud_dashboard.sidebar.slider[0].set_value(threshold).run()
        assert not cloud_dashboard.exception
        expected_count = int(demo_data.if_percentile.ge(threshold).sum())
        assert cloud_dashboard.metric[0].value == f"{expected_count:,}"
        counts.append(expected_count)
    assert counts[0] > counts[1]


def test_cloud_demo_empty_channel_filter_stops_cleanly(cloud_dashboard):
    cloud_dashboard.sidebar.multiselect[0].set_value([]).run()
    assert not cloud_dashboard.exception
    assert any("Tidak ada log" in item.value for item in cloud_dashboard.info)
    assert len(cloud_dashboard.metric) == 0


def test_cloud_demo_investigation_keeps_account_context_outside_date_filter(cloud_dashboard, demo_data):
    grouped = demo_data.groupby("account_key")
    account = next(key for key, group in grouped if group.created_at.dt.date.nunique() > 1)
    expected = grouped.get_group(account)
    chosen = expected.sort_values("created_at").iloc[-1]
    selected_date = chosen.created_at.date()
    cloud_dashboard.sidebar.multiselect[0].set_value([chosen.channel]).run()
    cloud_dashboard.sidebar.date_input[0].set_value((selected_date, selected_date)).run()
    cloud_dashboard.sidebar.radio[0].set_value("Transaction Investigation").run()
    assert not cloud_dashboard.exception
    selection = next(item for item in cloud_dashboard.selectbox if item.label == "Pilih log untuk melihat bukti")
    selection.set_value(chosen.log_key).run()
    assert not cloud_dashboard.exception
    visible = cloud_dashboard.dataframe[0].value
    context = cloud_dashboard.dataframe[1].value
    assert visible.created_at.dt.date.eq(selected_date).all()
    assert set(context.log_key) == set(expected.log_key)
    assert context.created_at.dt.date.ne(selected_date).any()


def test_cloud_demo_ignores_private_data_environment_and_csv(monkeypatch, tmp_path, demo_data):
    private = tmp_path / "private-artifacts"
    logs = private / "data/processed"
    logs.mkdir(parents=True)
    (logs / "sanitized_logs.csv").write_text("PRIVATE_SENTINEL", encoding="utf-8")
    monkeypatch.setenv("NORTON_DATA_ROOT", str(private))
    monkeypatch.syspath_prepend(str(DEMO_ROOT))

    def forbid_csv(*args, **kwargs):
        raise AssertionError("Public demo must not read CSV artifacts")

    monkeypatch.setattr(pd, "read_csv", forbid_csv)
    st.cache_data.clear()
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert app.metric[0].value == f"{len(demo_data):,}"
    for page in ["Anomaly Monitoring", "Model Comparison", "Transaction Investigation"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
    assert not any("PRIVATE_SENTINEL" in str(item.value) for item in app.markdown)


def test_cloud_demo_empty_generated_data_shows_error(monkeypatch):
    monkeypatch.syspath_prepend(str(DEMO_ROOT))
    import demo_data as generator
    monkeypatch.setattr(generator, "build_demo_data", lambda seed=2026: (pd.DataFrame(), {}))
    st.cache_data.clear()
    try:
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        assert not app.exception
        assert app.error[0].value == "Data demonstrasi belum tersedia. Silakan muat ulang aplikasi."
        assert len(app.metric) == 0
    finally:
        st.cache_data.clear()
