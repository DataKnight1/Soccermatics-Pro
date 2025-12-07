# Enzo Fernández - World Cup 2022 Analysis

**Twelve Soccermatics Pro - Project 1: Plotting Actions and Telling Stories**

A comprehensive data-driven analysis of Enzo Fernández's performance during Argentina's 2022 FIFA World Cup victory. This project is the first exercise of the Soccermatics Pro course, designed to analyze a specific player's actions, support the analysis with data visualization, and contextualize their performance against peers.

![Course Logo](https://images.squarespace-cdn.com/content/v1/5ebd6f2be3bec9264595f15f/c38d561d-8e75-4a72-a05a-ae006c1d6e2c/TwelveLogo+3.png?format=1500w)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://enzofernandezproject1.streamlit.app/)


---

## 📋 Project Goal

**The Assignment:**
1.  Think of a player who interests you.
2.  Identify important actions they performed and explain why.
3.  Plot the actions and describe how the data supports or contradicts your analysis.
4.  Collect statistics for the tournament and compare them to other players.
5.  Write a short, coach-readable text using at most two figures.

**My Analysis:**
I assessed Enzo Fernández's impact as Argentina's "Missing Piece," hypothesizing that his introduction provided the essential link between defense and attack that was absent in the opening match.

---

## 🔬 Key Findings (Data-Driven)

> **Hypothesis**: After entering the starting XI, Enzo Fernández acted as Argentina's main midfield connector and progression hub, performing at a significantly higher level than other central midfielders in terms of ball progression and volume.

| Metric | Value (p90) | Percentile | Insight |
|--------|-------------|------------|---------|
| **Progressive Passes** | 30.47 | **Top Tier** | Elite verticality, functioning as a deep playmaker. |
| **Deep Progressions** | 11.44 | **Top 10** | Consistently breaking the first line of pressure. |
| **Pressures** | 17.25 | **High** | High work rate, winning the ball back centrally. |
| **Pass Completion** | 88.5% | **High** | Rare combination of high risk (progression) and high security. |

**Conclusion**: The data unequivocally supports the hypothesis. Enzo functioned as a "Right-Sided Pivot," directing 31.8% of his progressive actions to the right half-space to feed Lionel Messi, while maintaining the defensive volume of a box-to-box midfielder.

---

## 🏗️ Project Structure

The project follows a modular production-ready structure as defined in `STRUCTURE.md`:

```
Project 1/
├── app.py                          # Streamlit application (Interactive Dashboard)
├── pipeline_runner.py              # Main automated pipeline script
├── README.md                       # This documentation
│
├── data/                           # StatsBomb open data storage
│
├── output/
│   ├── figures/                    # Generated visualization images (PNG)
│   ├── insights/                   # CSV exports of analysis results
│   └── metrics/                    # Computed player metrics
│
└── src/                            # Source code modules
    ├── core/                       # Configuration and settings
    ├── data/                       # Data loading and processing
    ├── analysis/                   # Metrics and statistical calculations
    └── visualizations/             # Plotting functions (heatmaps, radar, scatter)
```

---

## 🚀 Quick Start

### 1. Installation

Ensure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt
```

### 2. Run the Analysis Pipeline

This is the core script. It loads data, calculates metrics, and generates all visualization figures.

```bash
python pipeline_runner.py
```

### 3. Interactive Exploration

Launch the Streamlit dashboard to interact with the data and explore different comparisons.

[**👉 Click here to view the Live App**](https://enzofernandezproject1.streamlit.app/)


```bash
streamlit run app.py
```

---

## 📊 Visualizations

The pipeline generates high-quality visual evidence stored in `output/figures/`:

*   **Performance Profile**: A pizza chart highlighting his percentile dominance.
*   **Progression Heatmap**: Showing his tendency to progress down the right flank.
*   **Similarity Analysis**: Identifying his statistical "twins" (e.g., comparable to Bernardo Silva in creation, but with higher defensive output).

---

## 📝 Deliverables

*   **Interactive Dashboard**: The Streamlit application (`app.py`), allowing for dynamic data exploration.
*   **Figures**: All generated charts in `output/figures/`.

---

## 👤 Author

**Tiago**
*   **Course**: Twelve Soccermatics Pro
*   **Module**: Project 1 - Plotting Actions and Telling Stories
*   **Date**: December 2025

---
*Data provided by StatsBomb Open Data.*
