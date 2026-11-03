import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# CHARGER DONNÉES + MODÈLE

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


# DONNÉES STREAMING

stream_data = df[features_ml + ['QoS']].tail(200).reset_index(drop=True)

plt.style.use('dark_background')

history_size = 50
idx = [0]

hist = {'throughput': [], 'latency': [], 'packet_loss': [],
        'rsrp': [], 'qos': [], 'time': []}
alert_counts = {'good': 0, 'bad': 0}


# FIGURE

fig = plt.figure(figsize=(16, 9), facecolor='#0A1628')
fig.suptitle('SYSTEME DE MONITORING RESEAU 4G/5G — TEMPS REEL',
             fontsize=15, fontweight='bold', color='white', y=0.98)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4)

ax_tp     = fig.add_subplot(gs[0, :2])
ax_lat    = fig.add_subplot(gs[0, 2:])
ax_pl     = fig.add_subplot(gs[1, :2])
ax_rsrp   = fig.add_subplot(gs[1, 2:])
ax_status = fig.add_subplot(gs[2, :2])
ax_stats  = fig.add_subplot(gs[2, 2:])

for ax in [ax_tp, ax_lat, ax_pl, ax_rsrp, ax_status, ax_stats]:
    ax.set_facecolor('#0D1F35')

def update(frame):
    if idx[0] >= len(stream_data):
        idx[0] = 0
        for k in hist:
            hist[k].clear()
        alert_counts['good'] = 0
        alert_counts['bad']  = 0

    row = stream_data.iloc[idx[0]]

    hist['throughput'].append(row['throughput_mbps'])
    hist['latency'].append(row['latency_ms'])
    hist['packet_loss'].append(row['packet_loss_pct'])
    hist['rsrp'].append(row['rsrp_dbm'])
    hist['time'].append(idx[0])

    X_live = pd.DataFrame([[
        row['throughput_mbps'], row['latency_ms'],
        row['packet_loss_pct'], row['rsrp_dbm'],
        row['rsrq_db'], row['handover_count'], row['active_users']
    ]], columns=features_ml)

    pred  = rf.predict(scaler.transform(X_live))[0]
    proba = rf.predict_proba(scaler.transform(X_live))[0]
    hist['qos'].append(pred)

    if pred == 1:
        alert_counts['good'] += 1
    else:
        alert_counts['bad'] += 1

    for k in ['throughput', 'latency', 'packet_loss', 'rsrp', 'qos', 'time']:
        if len(hist[k]) > history_size:
            hist[k].pop(0)

    x         = range(len(hist['time']))
    qos_color = '#00C97A' if pred == 1 else '#E84545'
    status_txt = 'BONNE QoS' if pred == 1 else 'MAUVAISE QoS'
    status_icon = '[OK]' if pred == 1 else '[ALERTE]'

    # Throughput
    ax_tp.clear()
    ax_tp.set_facecolor('#0D1F35')
    ax_tp.plot(x, hist['throughput'], color='#00AAFF', linewidth=2)
    ax_tp.fill_between(x, hist['throughput'], alpha=0.2, color='#00AAFF')
    ax_tp.axhline(y=30, color='#FF4444', linestyle='--', linewidth=1.2)
    ax_tp.set_title(f'Throughput — {row["throughput_mbps"]:.1f} Mbps', color='white', fontsize=11)
    ax_tp.set_ylabel('Mbps', color='#8EAEC7', fontsize=9)
    ax_tp.tick_params(colors='#8EAEC7')

    # Latence
    ax_lat.clear()
    ax_lat.set_facecolor('#0D1F35')
    ax_lat.plot(x, hist['latency'], color='#FF8800', linewidth=2)
    ax_lat.fill_between(x, hist['latency'], alpha=0.2, color='#FF8800')
    ax_lat.axhline(y=50, color='#FF4444', linestyle='--', linewidth=1.2)
    ax_lat.set_title(f'Latence — {row["latency_ms"]:.1f} ms', color='white', fontsize=11)
    ax_lat.set_ylabel('ms', color='#8EAEC7', fontsize=9)
    ax_lat.tick_params(colors='#8EAEC7')

    # Packet Loss
    ax_pl.clear()
    ax_pl.set_facecolor('#0D1F35')
    ax_pl.plot(x, hist['packet_loss'], color='#E84545', linewidth=2)
    ax_pl.fill_between(x, hist['packet_loss'], alpha=0.2, color='#E84545')
    ax_pl.axhline(y=2, color='#FFAA00', linestyle='--', linewidth=1.2)
    ax_pl.set_title(f'Packet Loss — {row["packet_loss_pct"]:.2f}%', color='white', fontsize=11)
    ax_pl.set_ylabel('%', color='#8EAEC7', fontsize=9)
    ax_pl.tick_params(colors='#8EAEC7')

    # RSRP
    ax_rsrp.clear()
    ax_rsrp.set_facecolor('#0D1F35')
    ax_rsrp.plot(x, hist['rsrp'], color='#9B59B6', linewidth=2)
    ax_rsrp.fill_between(x, hist['rsrp'], alpha=0.2, color='#9B59B6')
    ax_rsrp.axhline(y=-100, color='#FF4444', linestyle='--', linewidth=1.2)
    ax_rsrp.set_title(f'RSRP Signal — {row["rsrp_dbm"]:.1f} dBm', color='white', fontsize=11)
    ax_rsrp.set_ylabel('dBm', color='#8EAEC7', fontsize=9)
    ax_rsrp.tick_params(colors='#8EAEC7')

    # Status ML
    ax_status.clear()
    ax_status.set_facecolor('#1A0A0A' if pred == 0 else '#0A1A0A')
    ax_status.axis('off')
    ax_status.text(0.5, 0.72, f'{status_icon}  {status_txt}',
                   ha='center', va='center', fontsize=22, fontweight='bold',
                   color=qos_color, transform=ax_status.transAxes)
    ax_status.text(0.5, 0.40,
                   f'Confiance ML : {max(proba)*100:.1f}%  |  Point #{idx[0]+1}/200',
                   ha='center', va='center', fontsize=12,
                   color='white', transform=ax_status.transAxes)
    ax_status.text(0.5, 0.14,
                   f'Heure : {datetime.now().strftime("%H:%M:%S")}',
                   ha='center', va='center', fontsize=11,
                   color='#8EAEC7', transform=ax_status.transAxes)

    # Stats
    ax_stats.clear()
    ax_stats.set_facecolor('#0D1F35')
    ax_stats.axis('off')
    total    = alert_counts['good'] + alert_counts['bad']
    pct_good = alert_counts['good'] / total * 100 if total > 0 else 0
    pct_bad  = alert_counts['bad']  / total * 100 if total > 0 else 0

    ax_stats.text(0.5, 0.92, 'Statistiques Session',
                  ha='center', fontsize=12, fontweight='bold',
                  color='white', transform=ax_stats.transAxes)
    ax_stats.text(0.25, 0.62, str(alert_counts['good']),
                  ha='center', fontsize=28, fontweight='bold',
                  color='#00C97A', transform=ax_stats.transAxes)
    ax_stats.text(0.25, 0.38, 'Bonne QoS',
                  ha='center', fontsize=11,
                  color='#00C97A', transform=ax_stats.transAxes)
    ax_stats.text(0.25, 0.16, f'{pct_good:.1f}%',
                  ha='center', fontsize=12,
                  color='#00C97A', transform=ax_stats.transAxes)

    ax_stats.text(0.75, 0.62, str(alert_counts['bad']),
                  ha='center', fontsize=28, fontweight='bold',
                  color='#E84545', transform=ax_stats.transAxes)
    ax_stats.text(0.75, 0.38, 'Mauvaise QoS',
                  ha='center', fontsize=11,
                  color='#E84545', transform=ax_stats.transAxes)
    ax_stats.text(0.75, 0.16, f'{pct_bad:.1f}%',
                  ha='center', fontsize=12,
                  color='#E84545', transform=ax_stats.transAxes)

    idx[0] += 1


# LANCER

print("Monitoring demarre — ferme la fenetre pour arreter")
ani = animation.FuncAnimation(fig, update, interval=400, cache_frame_data=False)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
print("Monitoring termine !")