from flask import Flask, render_template_string, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings, threading, time
warnings.filterwarnings('ignore')


# CHARGER MODÈLE

df = pd.read_csv('df_5g_clean.csv')
features_ml = ['throughput_mbps', 'latency_ms', 'packet_loss_pct',
                'rsrp_dbm', 'rsrq_db', 'handover_count', 'active_users']
X = df[features_ml]
y = df['QoS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_sc, y_train)
print("Modèle chargé !")

stream_data = df[features_ml + ['QoS']].tail(300).reset_index(drop=True)
state = {
    'idx': 0,
    'history': [],
    'good': 0,
    'bad': 0
}

def next_point():
    if state['idx'] >= len(stream_data):
        state['idx'] = 0
        state['history'] = []
        state['good'] = 0
        state['bad'] = 0
    row = stream_data.iloc[state['idx']]
    X_live = pd.DataFrame([[
        row['throughput_mbps'], row['latency_ms'],
        row['packet_loss_pct'], row['rsrp_dbm'],
        row['rsrq_db'], row['handover_count'], row['active_users']
    ]], columns=features_ml)
    pred  = int(rf.predict(scaler.transform(X_live))[0])
    proba = float(max(rf.predict_proba(scaler.transform(X_live))[0]))
    if pred == 1: state['good'] += 1
    else:         state['bad']  += 1
    point = {
        'idx':          state['idx'],
        'throughput':   round(float(row['throughput_mbps']), 2),
        'latency':      round(float(row['latency_ms']), 2),
        'packet_loss':  round(float(row['packet_loss_pct']), 3),
        'rsrp':         round(float(row['rsrp_dbm']), 1),
        'pred':         pred,
        'proba':        round(proba * 100, 1),
        'good':         state['good'],
        'bad':          state['bad'],
    }
    state['history'].append(point)
    if len(state['history']) > 60:
        state['history'].pop(0)
    state['idx'] += 1
    return point


# APP FLASK

app = Flask(__name__)

HTML = '''
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Monitoring QoS 4G/5G</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#0A1628; color:white; font-family:'Segoe UI',sans-serif; }

header {
  background:#081020;
  padding:18px 30px;
  display:flex; align-items:center; justify-content:space-between;
  border-bottom:2px solid #1C7293;
}
header h1 { font-size:20px; color:white; letter-spacing:1px; }
#clock { color:#8EAEC7; font-size:14px; }

.kpi-bar {
  display:grid; grid-template-columns:repeat(4,1fr); gap:14px;
  padding:18px 24px 6px;
}
.kpi-card {
  background:#112240;
  border-radius:10px;
  padding:16px 20px;
  border-left:4px solid #1C7293;
  transition: border-color .3s;
}
.kpi-card .label { font-size:12px; color:#8EAEC7; margin-bottom:6px; }
.kpi-card .value { font-size:28px; font-weight:700; }
.kpi-card .unit  { font-size:12px; color:#8EAEC7; }
.kpi-card.alert  { border-color:#E84545; }
.kpi-card.ok     { border-color:#00C97A; }

.charts-grid {
  display:grid; grid-template-columns:repeat(2,1fr); gap:14px;
  padding:6px 24px;
}
.chart-box {
  background:#112240;
  border-radius:10px;
  padding:14px 16px;
}
.chart-box h3 { font-size:13px; color:#8EAEC7; margin-bottom:10px; }

.bottom {
  display:grid; grid-template-columns:2fr 1fr; gap:14px;
  padding:6px 24px 20px;
}
#status-box {
  background:#112240;
  border-radius:10px;
  padding:22px;
  text-align:center;
  border:2px solid #1C7293;
  transition: all .4s;
}
#status-box h2 { font-size:30px; font-weight:800; margin-bottom:10px; }
#status-box p  { font-size:14px; color:#8EAEC7; }

#stats-box {
  background:#112240;
  border-radius:10px;
  padding:22px;
}
#stats-box h3 { font-size:14px; color:#8EAEC7; margin-bottom:16px; text-align:center; }
.stat-row { display:flex; justify-content:space-between; margin-bottom:12px; }
.stat-num { font-size:28px; font-weight:800; }
.stat-lbl { font-size:12px; color:#8EAEC7; }
.bar-track { background:#1A2A3A; border-radius:6px; height:8px; margin-top:6px; }
.bar-fill  { height:8px; border-radius:6px; transition:width .5s; }
</style>
</head>
<body>

<header>
  <h1>📡 MONITORING RÉSEAU 4G/5G — TEMPS RÉEL</h1>
  <span id="clock">--:--:--</span>
</header>

<div class="kpi-bar">
  <div class="kpi-card" id="card-tp">
    <div class="label"> Throughput</div>
    <div class="value" id="val-tp">--</div>
    <div class="unit">Mbps  (seuil: 30)</div>
  </div>
  <div class="kpi-card" id="card-lat">
    <div class="label"> Latence</div>
    <div class="value" id="val-lat">--</div>
    <div class="unit">ms  (seuil: 50)</div>
  </div>
  <div class="kpi-card" id="card-pl">
    <div class="label"> Packet Loss</div>
    <div class="value" id="val-pl">--</div>
    <div class="unit">%  (seuil: 2)</div>
  </div>
  <div class="kpi-card" id="card-rsrp">
    <div class="label"> RSRP Signal</div>
    <div class="value" id="val-rsrp">--</div>
    <div class="unit">dBm  (seuil: -100)</div>
  </div>
</div>

<div class="charts-grid">
  <div class="chart-box">
    <h3>Throughput (Mbps)</h3>
    <canvas id="chart-tp" height="90"></canvas>
  </div>
  <div class="chart-box">
    <h3>Latence (ms)</h3>
    <canvas id="chart-lat" height="90"></canvas>
  </div>
  <div class="chart-box">
    <h3>Packet Loss (%)</h3>
    <canvas id="chart-pl" height="90"></canvas>
  </div>
  <div class="chart-box">
    <h3>RSRP (dBm)</h3>
    <canvas id="chart-rsrp" height="90"></canvas>
  </div>
</div>

<div class="bottom">
  <div id="status-box">
    <h2 id="status-txt">-- QoS --</h2>
    <p id="status-info">En attente de données...</p>
    <p id="status-point" style="margin-top:8px;font-size:13px;"></p>
  </div>
  <div id="stats-box">
    <h3> Statistiques Session</h3>
    <div class="stat-row">
      <div>
        <div class="stat-num" id="cnt-good" style="color:#00C97A">0</div>
        <div class="stat-lbl">Bonne QoS</div>
        <div class="bar-track"><div class="bar-fill" id="bar-good" style="background:#00C97A;width:0%"></div></div>
      </div>
      <div style="text-align:right">
        <div class="stat-num" id="cnt-bad" style="color:#E84545">0</div>
        <div class="stat-lbl">Mauvaise QoS</div>
        <div class="bar-track"><div class="bar-fill" id="bar-bad" style="background:#E84545;width:0%;margin-left:auto"></div></div>
      </div>
    </div>
  </div>
</div>

<script>
const makeChart = (id, color, threshold) => {
  const ctx = document.getElementById(id).getContext('2d');
  const ch = new Chart(ctx, {
    type:'line',
    data:{ labels:[], datasets:[{
      data:[], borderColor:color, backgroundColor:color+'22',
      borderWidth:2, fill:true, tension:0.4, pointRadius:0
    }]},
    options:{
      animation:false, responsive:true,
      plugins:{ legend:{display:false} },
      scales:{
        x:{ display:false },
        y:{ grid:{color:'#1A2A3A'}, ticks:{color:'#8EAEC7', font:{size:10}},
            ...(threshold ? {} : {}) }
      }
    }
  });
  return ch;
};

const charts = {
  tp:   makeChart('chart-tp',   '#00AAFF', 30),
  lat:  makeChart('chart-lat',  '#FF8800', 50),
  pl:   makeChart('chart-pl',   '#E84545', 2),
  rsrp: makeChart('chart-rsrp', '#9B59B6', -100),
};

function addPoint(chart, label, value, maxPts=60) {
  chart.data.labels.push(label);
  chart.data.datasets[0].data.push(value);
  if(chart.data.labels.length > maxPts) {
    chart.data.labels.shift();
    chart.data.datasets[0].data.shift();
  }
  chart.update('none');
}

function updateClock() {
  const now = new Date();
  document.getElementById('clock').textContent =
    now.toLocaleTimeString('fr-FR');
}
setInterval(updateClock, 1000);
updateClock();

function setCard(id, val, ok) {
  const c = document.getElementById('card-'+id);
  c.className = 'kpi-card ' + (ok ? 'ok' : 'alert');
}

async function fetchData() {
  try {
    const r = await fetch('/data');
    const d = await r.json();

    document.getElementById('val-tp').textContent   = d.throughput + ' Mbps';
    document.getElementById('val-lat').textContent  = d.latency + ' ms';
    document.getElementById('val-pl').textContent   = d.packet_loss + ' %';
    document.getElementById('val-rsrp').textContent = d.rsrp + ' dBm';

    setCard('tp',   d.throughput,   d.throughput > 30);
    setCard('lat',  d.latency,      d.latency < 50);
    setCard('pl',   d.packet_loss,  d.packet_loss < 2);
    setCard('rsrp', d.rsrp,         d.rsrp > -100);

    addPoint(charts.tp,   d.idx, d.throughput);
    addPoint(charts.lat,  d.idx, d.latency);
    addPoint(charts.pl,   d.idx, d.packet_loss);
    addPoint(charts.rsrp, d.idx, d.rsrp);

    const box = document.getElementById('status-box');
    const txt = document.getElementById('status-txt');
    if(d.pred === 1) {
      txt.textContent = '  BONNE QoS';
      txt.style.color = '#00C97A';
      box.style.borderColor = '#00C97A';
      box.style.background  = '#0A1A0A';
    } else {
      txt.textContent = '  MAUVAISE QoS';
      txt.style.color = '#E84545';
      box.style.borderColor = '#E84545';
      box.style.background  = '#1A0A0A';
    }
    document.getElementById('status-info').textContent =
      `Confiance ML : ${d.proba}%`;
    document.getElementById('status-point').textContent =
      `Point #${d.idx + 1} / 300`;

    const total = d.good + d.bad;
    const pctG  = total ? Math.round(d.good/total*100) : 0;
    const pctB  = total ? Math.round(d.bad /total*100) : 0;
    document.getElementById('cnt-good').textContent = d.good;
    document.getElementById('cnt-bad').textContent  = d.bad;
    document.getElementById('bar-good').style.width = pctG + '%';
    document.getElementById('bar-bad').style.width  = pctB + '%';

  } catch(e) { console.error(e); }
}

setInterval(fetchData, 500);
fetchData();
</script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/data')
def data():
    return jsonify(next_point())

if __name__ == '__main__':
    print(" App web démarrée !")
    print(" Ouvre ton navigateur : http://localhost:5000")
    app.run(debug=False, port=5000)