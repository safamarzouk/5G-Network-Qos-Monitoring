📶 5G Network QoS Monitoring & Prediction

An intelligent platform that monitors mobile network performance, analyses 4G/5G radio and traffic indicators, and predicts whether a connection delivers a good Quality of Service (QoS), using Machine Learning.

✨ Highlights
🔍 Exploratory analysis of 5G KPI datasets with correlation heatmaps and trend charts
🧠 QoS classification with Logistic Regression and Random Forest
⏱️ Live monitoring window following throughput, latency, packet loss and RSRP signal strength
🖥️ Flask web dashboard to visualise the network state in the browser
🤖 Telecom assistant (LLaMA 3.3 through the Groq API) that answers questions about 4G/5G networks
📁 Repository Layout
5G-Network-Qos-Monitoring/
│
├── projet.ipynb              # Notebook: data analysis, ML training and evaluation
├── generate_dataset1.py      # Script that builds the KPI dataset
├── app_web.py                # Web dashboard (Flask)
├── monitoring_realtime.py    # Real-time monitoring window (Matplotlib)
├── agent_telecom.py          # AI telecom assistant (Flask + Groq API)
│
├── dataset1_5g_network_kpi.csv   # Raw 5G KPI data
├── df_5g_clean.csv               # Cleaned KPI data
├── df_ho_clean.csv               # Handover data
├── ds1_processed.csv             # Processed dataset used for training
├── network_logs_1.csv            # Network logs
│
└── *.png                     # Charts: heatmap, dashboards, KPI evolution, ML results
🚀 Getting Started
1. Get the code
bash
git clone https://github.com/safamarzouk/5G-Network-Qos-Monitoring.git
cd 5G-Network-Qos-Monitoring
2. Create a virtual environment and install the libraries
bash
python -m venv venv
venv\Scripts\activate
python -m pip install pandas numpy scikit-learn matplotlib seaborn flask flask-cors jupyter requests
▶️ Usage
What you want	Command	Where
Full analysis notebook	python -m jupyter notebook projet.ipynb	Opens in your browser
Web dashboard	python app_web.py	http://localhost:5000
Real-time monitoring	python monitoring_realtime.py	Desktop window
Telecom assistant	python agent_telecom.py	http://localhost:5001

🔑 The assistant needs a free Groq API key. Never write your key directly in the code before publishing it.

📊 Model Performance
Model	Accuracy	AUC
Logistic Regression	93.9 %	0.980
Random Forest	100 %	1.000

The most influential feature is the RSRP signal (about 84 % of the importance).

⚠️ Scores this high usually come from clearly separable, generated data: they demonstrate the method rather than performance on real network traffic.

What counts as a good QoS?
Indicator	Target
Throughput	above 30 Mbps
Latency	below 50 ms
Packet loss	below 2 %
RSRP signal	above -100 dBm
🧰 Built With

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Flask · Groq API (LLaMA 3.3) · Jupyter Notebook

👥 Authors

Project developed by a team of two telecommunications engineering students:

Safa Marzouk (@safamarzouk)
Aymen Gharbi (@Aym3nGharbi)