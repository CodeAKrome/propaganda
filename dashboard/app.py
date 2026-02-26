#!/usr/bin/env python3
"""
Dashboard for MongoDB RSS News Data
Run with: streamlit run dashboard/app.py
"""

import os
import sys
from datetime import datetime, timedelta
from collections import Counter

import streamlit as st
import pandas as pd
import pymongo
import altair as alt
from bson import ObjectId

# MongoDB connection config (same as mongo2chroma.py)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017")
MONGO_DB = "rssnews"
MONGO_COLL = "articles"


@st.cache_resource
def get_mongo_client():
    """Create MongoDB client connection."""
    return pymongo.MongoClient(MONGO_URI)


@st.cache_resource
def get_collection():
    """Get MongoDB collection."""
    client = get_mongo_client()
    return client[MONGO_DB][MONGO_COLL]


@st.cache_data(ttl=300)
def get_total_records():
    """Get total number of records."""
    coll = get_collection()
    return coll.count_documents({})


@st.cache_data(ttl=300)
def get_records_by_source():
    """Get record counts grouped by source."""
    coll = get_collection()
    pipeline = [
        {"$group": {"_id": "$source", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    results = list(coll.aggregate(pipeline))
    return pd.DataFrame([
        {"source": r["_id"] or "Unknown", "count": r["count"]}
        for r in results
    ])


@st.cache_data(ttl=300)
def get_records_over_time():
    """Get records over time with counts."""
    coll = get_collection()
    pipeline = [
        {"$match": {"published": {"$exists": True, "$ne": None}}},
        {"$group": {
            "_id": {
                "$dateToString": {"format": "%Y-%m-%d", "date": "$published"}
            },
            "count": {"$sum": 1}
        }},
        {"$sort": {"_id": 1}}
    ]
    results = list(coll.aggregate(pipeline))
    return pd.DataFrame([
        {"date": r["_id"], "count": r["count"]}
        for r in results
    ])


@st.cache_data(ttl=300)
def get_records_by_source_and_date():
    """Get records grouped by source and date."""
    coll = get_collection()
    pipeline = [
        {"$match": {"published": {"$exists": True, "$ne": None}}},
        {"$group": {
            "_id": {
                "source": "$source",
                "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$published"}}
            },
            "count": {"$sum": 1}
        }},
        {"$sort": {"_id.date": 1}}
    ]
    results = list(coll.aggregate(pipeline))
    return pd.DataFrame([
        {"source": r["_id"]["source"] or "Unknown", "date": r["_id"]["date"], "count": r["count"]}
        for r in results
    ])


@st.cache_data(ttl=300)
def get_all_sources():
    """Get list of all unique sources."""
    coll = get_collection()
    pipeline = [
        {"$group": {"_id": "$source"}},
        {"$sort": {"_id": 1}}
    ]
    results = list(coll.aggregate(pipeline))
    return sorted([r["_id"] or "Unknown" for r in results])


@st.cache_data(ttl=300)
def get_bias_by_source_and_date():
    """Get bias averages grouped by source and date."""
    coll = get_collection()
    pipeline = [
        {
            "$match": {
                "published": {"$exists": True, "$ne": None},
                "bias": {"$exists": True, "$ne": None}
            }
        },
        {
            "$project": {
                "source": 1,
                "published": 1,
                "bias": 1,
                "date_str": {"$dateToString": {"format": "%Y-%m-%d", "date": "$published"}}
            }
        },
        {
            "$group": {
                "_id": {
                    "source": "$source",
                    "date": "$date_str"
                },
                "count": {"$sum": 1},
                "avg_dir_L": {"$avg": "$bias.dir.L"},
                "avg_dir_C": {"$avg": "$bias.dir.C"},
                "avg_dir_R": {"$avg": "$bias.dir.R"},
                "avg_deg_L": {"$avg": "$bias.deg.L"},
                "avg_deg_M": {"$avg": "$bias.deg.M"},
                "avg_deg_H": {"$avg": "$bias.deg.H"}
            }
        },
        {"$sort": {"_id.date": 1, "_id.source": 1}}
    ]
    results = list(coll.aggregate(pipeline))
    return pd.DataFrame([
        {
            "source": r["_id"]["source"] or "Unknown",
            "date": r["_id"]["date"],
            "count": r["count"],
            "dir_L": r.get("avg_dir_L", 0) or 0,
            "dir_C": r.get("avg_dir_C", 0) or 0,
            "dir_R": r.get("avg_dir_R", 0) or 0,
            "deg_L": r.get("avg_deg_L", 0) or 0,
            "deg_M": r.get("avg_deg_M", 0) or 0,
            "deg_H": r.get("avg_deg_H", 0) or 0
        }
        for r in results
    ])


@st.cache_data(ttl=300)
def get_bias_by_day_of_week():
    """Get bias averages grouped by source and day of week."""
    coll = get_collection()
    pipeline = [
        {
            "$match": {
                "published": {"$exists": True, "$ne": None},
                "bias": {"$exists": True, "$ne": None}
            }
        },
        {
            "$project": {
                "source": 1,
                "bias": 1,
                "dayofweek": {"$dayOfWeek": "$published"}
            }
        },
        {
            "$group": {
                "_id": {
                    "source": "$source",
                    "dayofweek": "$dayofweek"
                },
                "count": {"$sum": 1},
                "avg_dir_L": {"$avg": "$bias.dir.L"},
                "avg_dir_C": {"$avg": "$bias.dir.C"},
                "avg_dir_R": {"$avg": "$bias.dir.R"}
            }
        }
    ]
    results = list(coll.aggregate(pipeline))
    # Map day numbers to names (MongoDB: 1=Sunday, 2=Monday, ..., 7=Saturday)
    day_names = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday", 7: "Saturday"}
    return pd.DataFrame([
        {
            "source": r["_id"]["source"] or "Unknown",
            "dayofweek": r["_id"]["dayofweek"],
            "day_name": day_names.get(r["_id"]["dayofweek"], "Unknown"),
            "count": r["count"],
            "dir_L": r.get("avg_dir_L", 0) or 0,
            "dir_C": r.get("avg_dir_C", 0) or 0,
            "dir_R": r.get("avg_dir_R", 0) or 0
        }
        for r in results
    ])


@st.cache_data(ttl=300)
def get_overall_bias():
    """Get overall bias across all records."""
    coll = get_collection()
    pipeline = [
        {
            "$match": {
                "bias": {"$exists": True, "$ne": None}
            }
        },
        {
            "$group": {
                "_id": None,
                "total_articles": {"$sum": 1},
                "avg_dir_L": {"$avg": "$bias.dir.L"},
                "avg_dir_C": {"$avg": "$bias.dir.C"},
                "avg_dir_R": {"$avg": "$bias.dir.R"},
                "avg_deg_L": {"$avg": "$bias.deg.L"},
                "avg_deg_M": {"$avg": "$bias.deg.M"},
                "avg_deg_H": {"$avg": "$bias.deg.H"}
            }
        }
    ]
    results = list(coll.aggregate(pipeline))
    if results:
        r = results[0]
        return pd.DataFrame([{
            "total_articles": r["total_articles"],
            "Left": r.get("avg_dir_L", 0) or 0,
            "Center": r.get("avg_dir_C", 0) or 0,
            "Right": r.get("avg_dir_R", 0) or 0,
            "Deg_Low": r.get("avg_deg_L", 0) or 0,
            "Deg_Medium": r.get("avg_deg_M", 0) or 0,
            "Deg_High": r.get("avg_deg_H", 0) or 0
        }])
    return pd.DataFrame()


@st.cache_data(ttl=60)
def get_sample_bias_records(limit=10):
    """Get sample bias records for inspection."""
    coll = get_collection()
    pipeline = [
        {
            "$match": {
                "bias": {"$exists": True, "$ne": None},
                "source": {"$exists": True, "$ne": None}
            }
        },
        {
            "$project": {
                "_id": 1,
                "title": 1,
                "source": 1,
                "published": 1,
                "bias": 1
            }
        },
        {"$limit": limit}
    ]
    results = list(coll.aggregate(pipeline))
    records = []
    for r in results:
        bias = r.get("bias", {})
        # Handle both object and string formats
        if isinstance(bias, str):
            try:
                import json
                bias = json.loads(bias)
            except:
                bias = {}
        
        records.append({
            "_id": str(r.get("_id", ""))[:12],
            "title": (r.get("title", "") or "")[:60] + "..." if len(r.get("title", "") or "") > 60 else r.get("title", ""),
            "source": r.get("source", "Unknown"),
            "dir_L": bias.get("dir", {}).get("L", 0) or 0,
            "dir_C": bias.get("dir", {}).get("C", 0) or 0,
            "dir_R": bias.get("dir", {}).get("R", 0) or 0,
            "deg_L": bias.get("deg", {}).get("L", 0) or 0,
            "deg_M": bias.get("deg", {}).get("M", 0) or 0,
            "deg_H": bias.get("deg", {}).get("H", 0) or 0,
            "reason": (bias.get("reason", "") or "")[:100]
        })
    return pd.DataFrame(records)


def main():
    st.set_page_config(
        page_title="RSS News Dashboard",
        page_icon="📰",
        layout="wide"
    )
    
    st.title("📰 RSS News Dashboard")
    st.markdown("---")
    
    # Widget 1: Total Records
    st.subheader("📊 Total Records")
    total = get_total_records()
    st.metric(label="Total Articles", value=f"{total:,}")
    
    # Widget 1b: Overall Bias Overview
    overall_bias = get_overall_bias()
    if not overall_bias.empty:
        st.subheader("🎯 Overall Bias Across All Sources")
        row = overall_bias.iloc[0]
        
        cols = st.columns(4)
        cols[0].metric("Total Analyzed", f"{int(row['total_articles']):,}")
        cols[1].metric("Left Score", f"{row['Left']:.3f}", delta="🔵 LEFT" if row['Left'] >= 0.5 else None, delta_color="inverse")
        cols[2].metric("Center Score", f"{row['Center']:.3f}", delta="🟢 CENTER" if row['Center'] >= 0.5 else None, delta_color="inverse")
        cols[3].metric("Right Score", f"{row['Right']:.3f}", delta="🔴 RIGHT" if row['Right'] >= 0.5 else None, delta_color="inverse")
        
        # Pie chart for bias direction
        bias_pie_data = pd.DataFrame({
            "Direction": ["Left", "Center", "Right"],
            "Score": [row['Left'], row['Center'], row['Right']]
        })
        
        pie_chart = alt.Chart(bias_pie_data).mark_arc(innerRadius=50).encode(
            alt.Theta(field="Score", type="quantitative"),
            alt.Color(field="Direction", type="nominal", scale=alt.Scale(domain=["Left", "Center", "Right"], range=["#3b82f6", "#22c55e", "#ef4444"])),
            tooltip=[alt.Tooltip("Direction:N"), alt.Tooltip("Score:Q", format=".3f")]
        ).properties(
            width=300,
            height=300,
            title="Overall Bias Distribution"
        )
        st.altair_chart(pie_chart, use_container_width=False)
        
        # Degree bar chart
        deg_data = pd.DataFrame({
            "Degree": ["Low", "Medium", "High"],
            "Score": [row['Deg_Low'], row['Deg_Medium'], row['Deg_High']]
        })
        deg_chart = alt.Chart(deg_data).mark_bar(color="#6366f1").encode(
            alt.X("Degree:N", title="Degree", sort=["Low", "Medium", "High"]),
            alt.Y("Score:Q", title="Average Score", scale=alt.Scale(domain=[0, 1])),
            tooltip=["Degree:N", alt.Tooltip("Score:Q", format=".3f")]
        ).properties(
            width=300,
            height=200,
            title="Bias Degree (Strength)"
        )
        st.altair_chart(deg_chart, use_container_width=False)
        
        # Determine overall status
        if row['Left'] >= 0.5:
            overall_status = "🔵 LEFT BIASED"
        elif row['Right'] >= 0.5:
            overall_status = "🔴 RIGHT BIASED"
        elif row['Center'] >= 0.5:
            overall_status = "🟢 CENTERED"
        else:
            overall_status = "⚪ NEUTRAL"
        st.markdown(f"**Overall Assessment: {overall_status}** (threshold ≥0.5)")
        
        # Show sample raw bias records
        with st.expander("📋 Sample Bias Records (Raw)"):
            sample_records = get_sample_bias_records(10)
            if not sample_records.empty:
                st.dataframe(
                    sample_records,
                    column_config={
                        "_id": st.column_config.TextColumn("ID", width="small"),
                        "title": st.column_config.TextColumn("Title", width="large"),
                        "source": st.column_config.TextColumn("Source", width="medium"),
                        "dir_L": st.column_config.NumberColumn("Left", format="%.3f"),
                        "dir_C": st.column_config.NumberColumn("Center", format="%.3f"),
                        "dir_R": st.column_config.NumberColumn("Right", format="%.3f"),
                        "deg_L": st.column_config.NumberColumn("Deg:L", format="%.3f"),
                        "deg_M": st.column_config.NumberColumn("Deg:M", format="%.3f"),
                        "deg_H": st.column_config.NumberColumn("Deg:H", format="%.3f"),
                        "reason": st.column_config.TextColumn("Reason", width="medium")
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.info("No bias records found")
    else:
        st.info("No bias data available. Run bias analysis on articles first.")
    
    st.markdown("---")
    
    # Widget 2: Records by Source
    st.subheader("📡 Records by Source")
    source_df = get_records_by_source()
    
    if not source_df.empty:
        # Display as table
        st.dataframe(
            source_df,
            column_config={
                "source": st.column_config.TextColumn("Source", width="medium"),
                "count": st.column_config.NumberColumn("Count", format="%d")
            },
            hide_index=True,
            width='stretch'
        )
        
        # Also show as bar chart
        st.bar_chart(source_df.set_index("source")["count"])
    else:
        st.info("No source data available")
    
    st.markdown("---")
    
    # Widget 3: Records Over Time (heatmap style)
    st.subheader("📈 Records Over Time")
    time_df = get_records_over_time()
    
    if not time_df.empty:
        # Convert to datetime for better handling with explicit format
        time_df["date"] = pd.to_datetime(time_df["date"], format="%Y-%m-%d", errors='coerce')
        
        # Filter out invalid dates (NaT) and dates before 2000
        time_df = time_df[time_df["date"].notna() & (time_df["date"] >= pd.Timestamp("2000-01-01"))]
        
        if not time_df.empty:
            # Add color based on count (normalize for color mapping)
            max_count = time_df["count"].max()
            time_df["intensity"] = time_df["count"] / max_count
            
            # Create color-coded bar chart
            st.bar_chart(
                time_df.set_index("date")["count"],
                color=["#4e79a7"]
            )
            
            # Show date range
            date_range = f"{time_df['date'].min().strftime('%Y-%m-%d')} to {time_df['date'].max().strftime('%Y-%m-%d')}"
            st.caption(f"Date range: {date_range}")
        else:
            st.info("No valid temporal data available")
    else:
        st.info("No temporal data available")
    
    st.markdown("---")
    
    # Widget 3b: Calendar Heatmap
    st.subheader("📅 Calendar Heatmap")
    
    if not time_df.empty:
        # Prepare data for calendar heatmap
        calendar_df = time_df.copy()
        calendar_df["year"] = calendar_df["date"].dt.year
        calendar_df["month"] = calendar_df["date"].dt.month
        calendar_df["day"] = calendar_df["date"].dt.day
        calendar_df["week"] = calendar_df["date"].dt.isocalendar().week
        calendar_df["dayofweek"] = calendar_df["date"].dt.dayofweek  # 0=Monday, 6=Sunday
        
        # Create calendar heatmap using Altair
        alt_chart = alt.Chart(calendar_df).mark_rect().encode(
            alt.X("dayofweek:O", title="Day of Week", axis=alt.Axis(labelAngle=0)),
            alt.Y("week:O", title="Week", sort="descending"),
            alt.Color("count:Q", scale=alt.Scale(scheme="blues"), title="Articles"),
            tooltip=["date:T", "count:Q"]
        ).properties(
            width=600,
            height=300,
            title="Articles per Day (Calendar Heatmap)"
        )
        
        st.altair_chart(alt_chart, width='stretch')
        
        st.caption("📅 Calendar view showing article density - darker blue = more articles")
    else:
        st.info("No temporal data available for heatmap")
    
    st.markdown("---")
    
    # Widget 4: Records by Source and Date (multi-select)
    st.subheader("📊 Records by Source Over Time")
    
    all_sources = get_all_sources()
    
    if all_sources:
        # Multi-select for sources
        selected_sources = st.multiselect(
            "Select sources to compare:",
            options=all_sources,
            default=all_sources[:5] if len(all_sources) > 5 else all_sources
        )
        
        if selected_sources:
            # Get data for selected sources
            source_date_df = get_records_by_source_and_date()
            
            if not source_date_df.empty:
                # Filter by selected sources
                filtered_df = source_date_df[source_date_df["source"].isin(selected_sources)]
                
                if not filtered_df.empty:
                    # Convert dates with error handling (use .copy() to avoid SettingWithCopyWarning)
                    filtered_df = filtered_df.copy()
                    filtered_df["date"] = pd.to_datetime(filtered_df["date"], format="%Y-%m-%d", errors='coerce')
                    filtered_df = filtered_df[filtered_df["date"].notna()]
                    
                    if not filtered_df.empty:
                        pivot_df = filtered_df.pivot(
                            index="date",
                            columns="source",
                            values="count"
                        ).fillna(0)
                        
                        # Line chart
                        st.line_chart(pivot_df)
                        
                        # Show raw data toggle
                        if st.checkbox("Show raw data"):
                            st.dataframe(
                                filtered_df.sort_values(["source", "date"]),
                                column_config={
                                    "source": st.column_config.TextColumn("Source"),
                                    "date": st.column_config.DateColumn("Date"),
                                    "count": st.column_config.NumberColumn("Count")
                                },
                                width='stretch'
                            )
                else:
                    st.info("No data for selected sources")
        else:
            st.info("Please select at least one source")
    else:
        st.info("No sources available")
    
    # Sidebar with refresh option
    st.sidebar.title("Options")
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Connected to: {MONGO_URI}")
    st.sidebar.caption(f"Database: {MONGO_DB}")
    st.sidebar.caption(f"Collection: {MONGO_COLL}")

    # ============================================================
    # NEW: Bias Charts by Source with Expand Widget
    # ============================================================
    st.markdown("---")
    st.subheader("📊 Volume & Bias by Source")
    
    # Get all data
    all_sources = get_all_sources()
    bias_df = get_bias_by_source_and_date()
    bias_by_day_df = get_bias_by_day_of_week()
    
    if not bias_df.empty and all_sources:
        # --- Summary: Bias Overview by Source ---
        st.markdown("**Bias Summary by Source** (≥0.5 = biased)")
        summary_data = []
        for source in all_sources:
            source_bias = bias_df[bias_df["source"] == source]
            if not source_bias.empty:
                avg_left = source_bias["dir_L"].mean()
                avg_center = source_bias["dir_C"].mean()
                avg_right = source_bias["dir_R"].mean()
                total_articles = source_bias["count"].sum()
                
                # Determine dominant bias
                if avg_left >= 0.5:
                    status = "🔵 LEFT"
                elif avg_right >= 0.5:
                    status = "🔴 RIGHT"
                elif avg_center >= 0.5:
                    status = "🟢 CENTER"
                else:
                    status = "⚪ NEUTRAL"
                
                summary_data.append({
                    "Source": source,
                    "Articles": total_articles,
                    "Left": round(avg_left, 3),
                    "Center": round(avg_center, 3),
                    "Right": round(avg_right, 3),
                    "Status": status
                })
        
        summary_df = pd.DataFrame(summary_data)
        if not summary_df.empty:
            st.dataframe(
                summary_df,
                column_config={
                    "Source": st.column_config.TextColumn("Source", width="medium"),
                    "Articles": st.column_config.NumberColumn("Articles", format="%d"),
                    "Left": st.column_config.NumberColumn("Left", format="%.3f"),
                    "Center": st.column_config.NumberColumn("Center", format="%.3f"),
                    "Right": st.column_config.NumberColumn("Right", format="%.3f"),
                    "Status": st.column_config.TextColumn("Status")
                },
                hide_index=True,
                use_container_width=True
            )
        
        # Create expand widget to show sources progressively
        # Show 4 initially, rest hidden until expanded
        show_all = st.checkbox("Show all sources", value=False)
        
        # Limit to 4 initially unless show_all is checked
        display_sources = all_sources if show_all else all_sources[:4]
        
        for source in display_sources:
            # Calculate overall bias for this source (average across all dates)
            source_bias = bias_df[bias_df["source"] == source]
            if not source_bias.empty:
                avg_left = source_bias["dir_L"].mean()
                avg_center = source_bias["dir_C"].mean()
                avg_right = source_bias["dir_R"].mean()
                
                # Determine bias status: if any direction >= 0.5, show as biased
                bias_label = ""
                if avg_left >= 0.5:
                    bias_label = " 🔵 LEFT"
                elif avg_right >= 0.5:
                    bias_label = " 🔴 RIGHT"
                elif avg_center >= 0.5:
                    bias_label = " 🟢 CENTER"
                else:
                    bias_label = " ⚪ NEUTRAL"
            else:
                bias_label = ""
            
            with st.expander(f"📈 {source}{bias_label}", expanded=False):
                # Filter data for this source
                source_time_df = bias_df[bias_df["source"] == source].copy()
                source_day_df = bias_by_day_df[bias_by_day_df["source"] == source].copy()
                
                if source_time_df.empty:
                    st.info(f"No bias data available for {source}")
                    continue
                
                # Convert dates
                source_time_df["date"] = pd.to_datetime(source_time_df["date"], format="%Y-%m-%d", errors='coerce')
                source_time_df = source_time_df[source_time_df["date"].notna()]
                
                if source_time_df.empty:
                    st.info(f"No temporal data available for {source}")
                    continue
                
                # --- Chart 1: Volume over time ---
                st.markdown("**Volume Over Time**")
                volume_chart = alt.Chart(source_time_df).mark_bar(color="#4e79a7").encode(
                    alt.X("date:T", title="Date"),
                    alt.Y("count:Q", title="Article Count"),
                    tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("count:Q", title="Articles")]
                ).properties(
                    width=600,
                    height=150,
                    title=f"{source} - Article Volume"
                )
                st.altair_chart(volume_chart, use_container_width=True)
                
                # --- Chart 2: Bias Direction over time ---
                st.markdown("**Bias Direction Over Time**")
                # Melt the dataframe for easier plotting
                bias_melted = source_time_df.melt(
                    id_vars=["date"],
                    value_vars=["dir_L", "dir_C", "dir_R"],
                    var_name="Direction",
                    value_name="Score"
                ).replace({
                    "dir_L": "Left",
                    "dir_C": "Center", 
                    "dir_R": "Right"
                })
                
                bias_line_chart = alt.Chart(bias_melted).mark_line(point=True).encode(
                    alt.X("date:T", title="Date"),
                    alt.Y("Score:Q", title="Bias Score", scale=alt.Scale(domain=[0, 1])),
                    alt.Color("Direction:N", scale=alt.Scale(domain=["Left", "Center", "Right"], range=["#3b82f6", "#22c55e", "#ef4444"])),
                    tooltip=[alt.Tooltip("date:T", title="Date"), "Direction:N", alt.Tooltip("Score:Q", format=".3f")]
                ).properties(
                    width=600,
                    height=200,
                    title=f"{source} - Bias Direction Over Time"
                ).interactive()
                st.altair_chart(bias_line_chart, use_container_width=True)
                
                # --- Chart 3: Bias by Day of Week ---
                if not source_day_df.empty:
                    st.markdown("**Bias by Day of Week**")
                    # Reorder days correctly
                    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    source_day_df["day_name"] = pd.Categorical(source_day_df["day_name"], categories=day_order, ordered=True)
                    source_day_df = source_day_df.sort_values("dayofweek")
                    
                    # Melt for direction
                    day_melted = source_day_df.melt(
                        id_vars=["day_name", "dayofweek"],
                        value_vars=["dir_L", "dir_C", "dir_R"],
                        var_name="Direction",
                        value_name="Score"
                    ).replace({
                        "dir_L": "Left",
                        "dir_C": "Center",
                        "dir_R": "Right"
                    })
                    
                    day_chart = alt.Chart(day_melted).mark_bar(point=True).encode(
                        alt.X("day_name:N", title="Day of Week", sort=day_order),
                        alt.Y("Score:Q", title="Bias Score", scale=alt.Scale(domain=[0, 1])),
                        alt.Color("Direction:N", scale=alt.Scale(domain=["Left", "Center", "Right"], range=["#3b82f6", "#22c55e", "#ef4444"])),
                        tooltip=["day_name:N", "Direction:N", alt.Tooltip("Score:Q", format=".3f")]
                    ).properties(
                        width=500,
                        height=200,
                        title=f"{source} - Bias by Day of Week"
                    )
                    st.altair_chart(day_chart, use_container_width=True)
                
                # Show data table
                with st.expander("View Raw Data"):
                    st.dataframe(
                        source_time_df.sort_values("date"),
                        column_config={
                            "source": st.column_config.TextColumn("Source"),
                            "date": st.column_config.DateColumn("Date"),
                            "count": st.column_config.NumberColumn("Articles"),
                            "dir_L": st.column_config.NumberColumn("Left", format="%.3f"),
                            "dir_C": st.column_config.NumberColumn("Center", format="%.3f"),
                            "dir_R": st.column_config.NumberColumn("Right", format="%.3f"),
                            "deg_L": st.column_config.NumberColumn("Deg: Low", format="%.3f"),
                            "deg_M": st.column_config.NumberColumn("Deg: Med", format="%.3f"),
                            "deg_H": st.column_config.NumberColumn("Deg: High", format="%.3f")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
        
        if not show_all and len(all_sources) > 4:
            st.caption(f"Showing 4 of {len(all_sources)} sources. Check 'Show all sources' to display all.")
    else:
        st.info("No bias data available. Run bias analysis on articles first.")


if __name__ == "__main__":
    main()
