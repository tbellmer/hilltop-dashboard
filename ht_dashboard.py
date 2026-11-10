from contextlib import closing
from datetime import date
from pathlib import Path
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st


# ------------------------------------------------------------
# Settings: change this to your existing SQLite database file.
# ------------------------------------------------------------

DEFAULT_DB_PATH = str(
    Path(__file__).resolve().parent / "bvr_capacity.sqlite"
)

DEFAULT_START_DATE = date(2026, 10, 9)

WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

ACTIVITY_CAPACITY = {
    "pickleball": 32,
    "basketball": 60,
    "floor": 100,
    "track": 30,
}


def load_data(db_path, start_date):
    """Read the requested records using a short-lived connection."""
    query = """
        SELECT
            capture_date,
            capture_time,
            CASE
                WHEN pickleball_closed = 0 THEN pickleball
                ELSE NULL
            END AS pickleball,
            CASE
                WHEN basketball_closed = 0 THEN basketball
                ELSE NULL
            END AS basketball,
            floor,
            track
        FROM bvr_capacity_hourly
        WHERE capture_date >= ?
        ORDER BY capture_date, capture_time
    """

    # Read-only mode also prevents accidentally creating a new DB.
    db_uri = db_path.resolve().as_uri() + "?mode=ro"

    with closing(
        sqlite3.connect(db_uri, uri=True, timeout=10)
    ) as connection:
        return pd.read_sql_query(
            query,
            connection,
            params=(start_date.isoformat(),),
        )


def prepare_data(df):
    """Create date, weekday, and hour columns for reporting."""
    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["capture_date"].astype(str).str.strip()
        + " "
        + df["capture_time"].astype(str).str.strip(),
        errors="coerce",
    )

    for column in ["pickleball", "basketball", "floor", "track"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for activity_name, capacity in ACTIVITY_CAPACITY.items():
        df[f"{activity_name}_pct"] = (
            df[activity_name] / capacity * 100
        )

    invalid_timestamps = int(df["timestamp"].isna().sum())
    df = df.dropna(subset=["timestamp"]).copy()

    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.day_name()

    return df, invalid_timestamps


def make_heatmap(table, title, color_label, number_format=".1f"):
    """Render a heatmap with hours displayed as HH:00."""
    display = table.copy()
    display.index = display.index.astype(str)
    display.columns = [
        f"{int(hour):02d}:30" for hour in display.columns
    ]

    fig = px.imshow(
        display,
        aspect="auto",
        text_auto=number_format,
        color_continuous_scale="YlOrRd",
        labels={
            "x": "Capture-hour bucket",
            "y": "",
            "color": color_label,
        },
        title=title,
        zmin=0,
    )

    fig.update_traces(hoverongaps=False)
    fig.update_layout(
        height=max(350, 35 * len(display.index) + 150)
    )

    return fig


def main():
    st.set_page_config(
        page_title="Hilltop Activity",
        page_icon="🏓",
        layout="wide",
    )

    st.title("Hilltop Activity Dashboard")

    st.caption(
        "Hourly captured activity counts, not unique daily visitors. "
    )

    # --------------------------------------------------------
    # Database controls
    # --------------------------------------------------------
    st.sidebar.header("Database")

    db_path_text = st.sidebar.text_input(
        "SQLite database path",
        value=DEFAULT_DB_PATH,
    )

    load_start_date = st.sidebar.date_input(
        "Load records starting",
        value=DEFAULT_START_DATE,
        key="load_start_date",
    )

    # Any interaction reruns the app. This button explicitly
    # provides a way to reload new database records.
    st.sidebar.button("Refresh data")

    db_path = Path(db_path_text.strip()).expanduser()

    if not db_path.is_file():
        st.error(
            "Database file not found. Enter the full path "
            "to your existing SQLite database in the sidebar."
        )
        st.code(str(db_path))
        st.stop()

    try:
        raw_df = load_data(db_path, load_start_date)
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        st.error(f"Could not read the database: {exc}")
        st.stop()

    if raw_df.empty:
        st.warning(
            "Your query returned no records. "
            "Try an earlier starting date."
        )
        st.stop()

    df, invalid_timestamps = prepare_data(raw_df)

    if invalid_timestamps:
        st.warning(
            f"Skipped {invalid_timestamps} records with "
            "invalid capture dates or times."
        )

    if df.empty:
        st.warning("No records have valid capture timestamps.")
        st.stop()

    st.sidebar.caption(
        f"Loaded {len(df):,} records. "
        "Database access is read-only."
    )

    # --------------------------------------------------------
    # Chart filters
    # --------------------------------------------------------
    st.sidebar.header("Chart filters")

    activity = st.sidebar.selectbox(
        "Activity",
        options=["pickleball", "basketball", "floor", "track"],
        format_func=str.title,
    )

    start_column, end_column = st.columns(2)

    with start_column:
        chart_start = st.date_input(
            "Chart start date",
            value=df["date"].min(),
        )

    with end_column:
        chart_end = st.date_input(
            "Chart end date",
            value=df["date"].max(),
        )

    selected_days = st.sidebar.multiselect(
        "Weekdays",
        options=WEEKDAYS,
        default=WEEKDAYS,
    )

    hour_start, hour_end = st.sidebar.slider(
        "Capture hours",
        min_value=0,
        max_value=23,
        value=(5, 22),
        step=1,
    )

    if chart_start > chart_end:
        st.warning("Chart start date must be on or before end date.")
        st.stop()

    filtered = df.loc[
        df["date"].between(chart_start, chart_end)
        & df["weekday"].isin(selected_days)
        & df["hour"].between(hour_start, hour_end)
    ].copy()

    filtered = filtered.dropna(subset=[activity])

    if filtered.empty:
        st.warning("No observations match the chart filters.")
        st.stop()

    # One observation per date/hour prevents extra captures
    # on one day from receiving extra weight.
    hourly = (
        filtered.groupby(
            ["date", "weekday", "hour"],
            as_index=False,
        )[activity]
        .mean()
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    st.subheader(f"{activity.title()} summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Observed days",
        f"{hourly['date'].nunique():,}",
    )

    col2.metric(
        "Observed date/hour slots",
        f"{len(hourly):,}",
    )

    col3.metric(
        "Average captured count",
        f"{hourly[activity].mean():.1f}",
    )

    col4.metric(
        "Highest captured count",
        f"{filtered[activity].max():g}",
    )

    latest_capture = filtered["timestamp"].max()

    st.caption(
        f"Latest capture in this selection: "
        f"{latest_capture:%Y-%m-%d %H:%M}. "
        "Some dates may contain only part of the day."
    )

    # --------------------------------------------------------
    # Typical range by hour
    # --------------------------------------------------------
    st.subheader("Typical range by hour")

    st.caption(
        "Each box shows the spread of captured counts for that "
        "reporting hour. The line inside the box is the median. "
        "Individual points beyond the box are unusually high or "
        "low observations."
    )

    box_fig = px.box(
        filtered,
        x="hour",
        y=activity,
        labels={
            "hour": "Capture-hour bucket",
            activity: "Captured count",
        },
    )

    available_hours = sorted(filtered["hour"].unique())

    box_fig.update_xaxes(
        tickmode="array",
        tickvals=available_hours,
        ticktext=[
            f"{hour:02d}:30" for hour in available_hours
        ],
    )

    box_fig.update_yaxes(rangemode="tozero")

    st.plotly_chart(
        box_fig,
        use_container_width=True,
    )

    st.caption(
        "A narrow box means the counts for that time slot have been "
        "similar across observed days. A wide box or distant points "
        "means activity has varied more."
    )

    # --------------------------------------------------------
    # Weekday/hour heatmap
    # --------------------------------------------------------
    st.subheader("Typical activity by weekday and hour")

    day_order = [
        day for day in WEEKDAYS if day in selected_days
    ]
    hour_order = list(range(hour_start, hour_end + 1))

    averages = hourly.pivot_table(
        index="weekday",
        columns="hour",
        values=activity,
        aggfunc="mean",
    ).reindex(
        index=day_order,
        columns=hour_order,
    )

    sample_counts = hourly.pivot_table(
        index="weekday",
        columns="hour",
        values=activity,
        aggfunc="count",
    ).reindex(
        index=day_order,
        columns=hour_order,
    )

    st.plotly_chart(
        make_heatmap(
            averages,
            title="Average captured count",
            color_label="Average count",
        ),
        use_container_width=True,
    )

    st.caption(
        "Darker cells indicate higher counts. "
        "Blank cells mean no observations, not zero activity. "
        "For example, captures at 09:48 and 09:59 are grouped "
        "into the 09:30 capture-hour bucket."
    )

    with st.expander("How many observed days support each cell?"):
        display_counts = sample_counts.fillna(0).astype(int)
        display_counts.columns = [
            f"{hour:02d}:30" for hour in hour_order
        ]

        st.dataframe(
            display_counts,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # Weekday comparison
    # --------------------------------------------------------
    st.subheader("Compare hourly patterns across weekdays")

    lines = (
        hourly.groupby(
            ["weekday", "hour"],
            as_index=False,
        )[activity]
        .mean()
    )

    # Reindex to keep missing hours as gaps in the chart.
    full_index = pd.MultiIndex.from_product(
        [day_order, hour_order],
        names=["weekday", "hour"],
    )

    lines = (
        lines.set_index(["weekday", "hour"])
        .reindex(full_index)
        .reset_index()
    )

    line_fig = px.line(
        lines,
        x="hour",
        y=activity,
        color="weekday",
        markers=True,
        category_orders={"weekday": WEEKDAYS},
        labels={
            "hour": "Capture hour",
            activity: "Average captured count",
            "weekday": "Weekday",
        },
    )

    line_fig.update_traces(connectgaps=False)
    line_fig.update_xaxes(
        tickmode="array",
        tickvals=hour_order,
        ticktext=[f"{hour:02d}:30" for hour in hour_order],
    )
    line_fig.update_yaxes(rangemode="tozero")

    st.plotly_chart(
        line_fig,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Individual dates
    # --------------------------------------------------------
    st.subheader("Activity by individual date")

    daily = hourly.pivot_table(
        index="date",
        columns="hour",
        values=activity,
        aggfunc="mean",
    ).sort_index().reindex(columns=hour_order)

    st.plotly_chart(
        make_heatmap(
            daily,
            title="Captured count by date",
            color_label="Count",
        ),
        use_container_width=True,
    )



    # --------------------------------------------------------
    # Pickleball versus basketball
    # --------------------------------------------------------
    st.subheader("Pickleball versus basketball")

    comparison = filtered[
        [
            "timestamp",
            "hour",
            "pickleball_pct",
            "basketball_pct",
        ]
    ].dropna(
        subset=["pickleball_pct", "basketball_pct"]
    )

    if comparison.empty:
        st.info(
            "No observations have both pickleball and basketball open."
        )
    else:
        comparison_by_hour = (
            comparison.groupby("hour", as_index=False)[
                ["pickleball_pct", "basketball_pct"]
            ]
            .mean()
        )

        comparison_long = comparison_by_hour.melt(
            id_vars=["hour"],
            value_vars=[
                "pickleball_pct",
                "basketball_pct",
            ],
            var_name="activity",
            value_name="percent_utilized",
        )

        comparison_long["activity"] = (
            comparison_long["activity"]
            .str.replace("_pct", "", regex=False)
            .str.title()
        )

        comparison_fig = px.bar(
            comparison_long,
            x="hour",
            y="percent_utilized",
            color="activity",
            barmode="group",
            labels={
                "hour": "Capture-hour bucket",
                "percent_utilized": "Average percent utilized",
                "activity": "Activity",
            },
        )

        available_hours = sorted(comparison["hour"].unique())

        comparison_fig.update_xaxes(
            tickmode="array",
            tickvals=available_hours,
            ticktext=[
                f"{hour:02d}:30"
                for hour in available_hours
            ],
        )

        comparison_fig.update_yaxes(
            rangemode="tozero",
            ticksuffix="%",
        )

	comparison_fig.update_traces(
    	    texttemplate="%{y:.2f}%",
            textposition="outside",
	)

        st.plotly_chart(
            comparison_fig,
            use_container_width=True,
        )

        st.caption(
            "This chart includes only observations where both pickleball "
            "and basketball were open. Grouped bars make it easy to "
            "compare the two activities at the same reporting time."
        )


    # --------------------------------------------------------
    # Source records and optional export
    # --------------------------------------------------------
    output_columns = [
        "capture_date",
        "capture_time",
        "pickleball",
        "basketball",
        "floor",
        "track"
    ]

    export_df = filtered.sort_values("timestamp")[output_columns]

    with st.expander("View filtered query results"):
        st.dataframe(
            export_df,
            use_container_width=True,
            hide_index=True,
        )

    st.download_button(
        label="Download filtered results as CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="hilltop_filtered_activity.csv",
        mime="text/csv",
    )

    st.sidebar.info(
        "Click Refresh data after your collector saves new records. "
        "This dashboard does not automatically poll the database."
    )


if __name__ == "__main__":
    main()