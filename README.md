# 📶 5G Network QoS Monitoring & Prediction

An end-to-end platform that monitors mobile network performance, analyses 4G/5G radio and traffic indicators, and predicts the Quality of Service (QoS) of a connection using Machine Learning.

---

## ✨ Features

- 📊 **Data Analysis** — Explore and visualize 5G KPI datasets (correlations, trends)
- 🤖 **Machine Learning** — Logistic Regression and Random Forest QoS classifiers
- 📡 **Real-time Monitoring** — Live window tracking throughput, latency, packet loss and RSRP signal
- 🌐 **Web Dashboard** — Flask web application to follow the network state in the browser
- 💬 **AI Telecom Assistant** — Chatbot specialized in 4G/5G networks, powered by LLaMA 3.3 (Groq API)

---

## 🗂️ Project Structure

```
5G-Network-Qos-Monitoring/
│
├── projet.ipynb              # Main notebook (analysis, training, evaluation)
├── generate_dataset1.py      # Script that generates the KPI dataset
├── app_web.py                # Web dashboard (Flask)
├── monitoring_realtime.py    # Real-time monitoring window (Matplotlib)
├── agent_telecom.py          # AI telecom assistant (Flask + Groq API)
│
├── dataset1_5g_network_kpi.csv   # Raw 5G KPI data
├── df_5g_clean.csv               # Cleaned KPI data
├── df_ho_clean.csv               # Handover data
├── ds1_processed.csv             # Processed data used for training
├── network_logs_1.csv            # Network logs
│
├── correlation_heatmap.png
├── dashboard_networklogs.png
├── dashboard_qos.png
├── evaluation_avancee.png
├── evolution_kpis.png
└── resultats_ml.png
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/safamarzouk/5G-Network-Qos-Monitoring.git
cd 5G-Network-Qos-Monitoring
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install pandas numpy scikit-learn matplotlib seaborn flask flask-cors jupyter requests
```

---

## ▶️ How to Run

### 📓 Jupyter Notebook (full analysis)

```bash
python -m jupyter notebook projet.ipynb
```

### 🌐 Web Dashboard

```bash
python app_web.py
```

Then open: http://localhost:5000

### 📊 Real-time Monitoring Window

```bash
python monitoring_realtime.py
```

### 💬 AI Telecom Assistant

The assistant needs a free Groq API key (https://console.groq.com). Never write your key directly in the code before publishing it.

```bash
python agent_telecom.py
```

Then open: http://localhost:5001

---

## 🧠 Machine Learning Results

| Model | Accuracy | AUC |
|-------|----------|-----|
| Logistic Regression | 93.9% | 0.980 |
| Random Forest | 100% | 1.000 |

Most important feature: **RSRP signal** (about 84% of the importance).

> ⚠️ Scores this high usually come from clearly separable, generated data: they demonstrate the method rather than performance on real network traffic.

---

## 📈 QoS Thresholds

| KPI | Good QoS threshold |
|-----|--------------------|
| Throughput | > 30 Mbps |
| Latency | < 50 ms |
| Packet Loss | < 2% |
| RSRP Signal | > -100 dBm |

---

## 🛠️ Technologies

- Python 3.x
- Scikit-learn — Machine Learning
- Pandas / NumPy — Data processing
- Matplotlib / Seaborn — Data visualization
- Flask — Web framework
- Groq API (LLaMA 3.3) — AI Telecom Assistant
- Jupyter Notebook — Analysis and reporting

---

## 👥 Authors

Project developed by two telecommunications engineering students:

- **Safa Marzouk** — [@safamarzouk](https://github.com/safamarzouk)
- **Aymen Gharbi** — [@Aym3nGharbi](https://github.com/Aym3nGharbi)
