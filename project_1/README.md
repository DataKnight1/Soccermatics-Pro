<div align="center">

<img src="https://images.squarespace-cdn.com/content/v1/5ebd6f2be3bec9264595f15f/c38d561d-8e75-4a72-a05a-ae006c1d6e2c/TwelveLogo+3.png?format=1500w" alt="Twelve Football" width="180"/>

# ⚽ Enzo Fernández - World Cup 2022 Analysis

### *Argentina's Missing Piece: A Data-Driven Performance Study*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B.svg)](https://soccermatics-pro--enzo-fernandez-project1.streamlit.app/)
[![StatsBomb](https://img.shields.io/badge/Data-StatsBomb_Open-green.svg)](https://github.com/statsbomb/open-data)

**Course:** Twelve Soccermatics Pro | **Project:** Plotting Actions and Telling Stories

[🚀 Live Dashboard](https://soccermatics-pro--enzo-fernandez-project1.streamlit.app/) • [📊 Analysis](#-key-findings-data-driven) • [💻 Quick Start](#-quick-start)

</div>

---

## 📑 Table of Contents

- [Project Goal](#-project-goal)
- [Key Findings](#-key-findings-data-driven)
- [Project Structure](#️-project-structure)
- [Quick Start](#-quick-start)
- [Visualizations](#-visualizations)
- [Author](#-author)

---

## 📋 Project Goal

**The Assignment:**
1. Think of a player who interests you
2. Identify important actions they performed and explain why
3. Plot the actions and describe how the data supports or contradicts your analysis
4. Collect statistics for the tournament and compare them to other players
5. Write a short, coach-readable text using at most two figures

**My Analysis:**
I assessed Enzo Fernández's impact as Argentina's **"Missing Piece"**, hypothesizing that his introduction provided the essential link between defense and attack that was absent in the opening match against Saudi Arabia.

---

## 🔬 Key Findings (Data-Driven)

> **Hypothesis**: After entering the starting XI, Enzo Fernández acted as Argentina's main midfield connector and progression hub, performing at a significantly higher level than other central midfielders in terms of ball progression and volume.

### Executive Summary

| Metric | Value | Rank | Insight |
|--------|-------|------|---------|
| **Deep Progressions (per 90)** | 11.44 | #2 | Elite volume |
| **Progression Accuracy** | 88.5% | Top 10% | Reliable outlet |
| **Right-Sided Tendency** | 31% | Unique | Tactical bias |
| **Primary Targets** | Molina & Messi | - | Strategic overload |

### Key Insights

1. **Volume & Reliability:** Ranked **#2** in Deep Progressions (11.44/90) with **88.5%** accuracy, providing a constant, safe outlet for defenders.

2. **Right-Sided Bias:** Unlike typical central midfielders, Enzo directed **31%** of his progressions to the deep right flank, specifically targeting **Molina** and **Messi**.

3. **Tactical Impact:** This right-sided overload allowed Argentina to break low blocks by isolating Messi in favorable 1v1 situations against shifted defenses.

### Conclusion

The data unequivocally supports the hypothesis. Enzo functioned as a **"Right-Sided Pivot,"** directing his progressive actions to unlock the right half-space while maintaining elite defensive volume.

---

## 🗂️ Project Structure

```
project_1/
├── app.py                          # Streamlit Interactive Dashboard
├── pipeline_runner.py              # Automated analysis pipeline
├── README.md                       # This documentation
│
├── data/                           # StatsBomb open data
│   ├── competitions/
│   ├── matches/
│   └── events/
│
├── output/
│   ├── figures/                    # PNG visualizations
│   ├── insights/                   # CSV analysis results
│   └── metrics/                    # Computed player metrics
│
└── src/                            # Source code modules
    ├── core/                       # Configuration & settings
    ├── data/                       # Data loading & processing
    ├── analysis/                   # Metrics & calculations
    └── visualizations/             # Plotting functions
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### 1. Installation

Clone the repository and install dependencies:

```bash
# Navigate to project directory
cd project_1

# Install required packages
pip install -r requirements.txt
```

### 2. Run the Analysis Pipeline

Execute the automated pipeline to generate all visualizations and metrics:

```bash
python pipeline_runner.py
```

This will:
- ✅ Load StatsBomb data for the 2022 World Cup
- ✅ Calculate player metrics and rankings
- ✅ Generate all visualization figures
- ✅ Export insights to CSV files

### 3. Launch Interactive Dashboard

Start the Streamlit application for dynamic exploration:

```bash
streamlit run app.py
```

Or visit the **[Live App](https://soccermatics-pro--enzo-fernandez-project1.streamlit.app/)** deployed on Streamlit Cloud.

---

## 📊 Visualizations

The pipeline generates high-quality analytical visualizations stored in `output/figures/`:

### Available Charts

- **🍕 Performance Profile (Pizza Chart)**: Percentile rankings across key metrics
- **🗺️ Progression Heatmap**: Spatial distribution showing right-sided bias
- **📈 Similarity Analysis**: Statistical comparison with similar players (e.g., Bernardo Silva)
- **⚡ Action Maps**: Pass networks and progressive action sequences
- **📊 Tournament Rankings**: Position among all central midfielders

### Example Outputs

All visualizations are designed to be **coach-readable**, emphasizing clarity and actionable insights over complexity.


---


<div align="center">

*Data provided by [StatsBomb Open Data](https://github.com/statsbomb/open-data)*

**Built with ⚽ for football analytics**

[⬆ Back to Top](#-enzo-fernández---world-cup-2022-analysis)

</div>
