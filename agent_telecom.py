from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import urllib.request
import json
import warnings
warnings.filterwarnings('ignore')


# CHARGER DONNÉES + MODÈLE

df = pd.read_csv('df_5g_clean.csv')

features_ml = ['throughput_mbps', 'latency_ms', 'packet_loss_pct',
                'rsrp_dbm', 'rsrq_db', 'handover_count', 'active_users']

X = df[features_ml]
y = df['QoS']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_sc, y_train)


# CONTEXTE DU PROJET (injecté dans l'agent)

CONTEXTE_PROJET = f"""
Tu es un assistant expert en ingénierie des réseaux mobiles 4G/5G.
Tu travailles sur un projet d'analyse et prédiction de la QoS.

DONNÉES RÉELLES DU PROJET :
━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Dataset 5G KPI :
   • Total lignes analysées : {len(df)}
   • Bonne QoS  : {df['QoS'].sum()} lignes ({df['QoS'].mean()*100:.1f}%)
   • Mauvaise QoS : {(df['QoS']==0).sum()} lignes ({(1-df['QoS'].mean())*100:.1f}%)

📈 KPIs moyens :
   • Throughput moyen    : {df['throughput_mbps'].mean():.1f} Mbps
   • Latence moyenne     : {df['latency_ms'].mean():.1f} ms
   • Packet Loss moyen   : {df['packet_loss_pct'].mean():.2f}%
   • RSRP moyen          : {df['rsrp_dbm'].mean():.1f} dBm
   • Utilisateurs actifs : {df['active_users'].mean():.0f} en moyenne

📊 Seuils QoS définis :
   • Throughput > 30 Mbps
   • Latence < 50 ms
   • Packet Loss < 2%
   • RSRP > -100 dBm

🤖 Résultats Machine Learning :
   • Logistic Regression : 93.9% accuracy, AUC = 0.980
   • Random Forest       : 100% accuracy, AUC = 1.000
   • Feature la + importante : RSRP (84% d'importance)
   • Cross-validation RF : 100% ± 0.05% (très stable)

🚨 Scénario réel testé (CELL_007) :
   • Throughput : 8.5 Mbps (critique)
   • Latence    : 120 ms (critique)
   • RSRP       : -108 dBm (signal faible)
   • Résultat ML : Dégradation détectée à 99% confiance

🛠️ Technologies utilisées :
   Python, Scikit-learn, Matplotlib, Flask, Machine Learning

RÈGLES DE COMPORTEMENT :
━━━━━━━━━━━━━━━━━━━━━━━
1. Réponds TOUJOURS en français
2. Utilise les vraies données du projet ci-dessus
3. Sois précis et technique mais clair
4. Donne toujours des recommandations concrètes
5. Utilise des emojis pour rendre la réponse lisible
6. Maximum 200 mots par réponse
7. Si on te demande de prédire la QoS → utilise les seuils définis
"""


# INTERFACE HTML

HTML_PAGE = """
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agent IA Télécom 4G/5G</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }

body {
  background: #0A1628;
  color: white;
  font-family: 'Segoe UI', sans-serif;
  height: 100vh;
  display: flex;
  flex-direction: column;
}

header {
  background: #081020;
  padding: 16px 24px;
  border-bottom: 2px solid #1C7293;
  display: flex;
  align-items: center;
  gap: 14px;
}

.agent-avatar {
  width: 46px; height: 46px;
  background: linear-gradient(135deg, #1C7293, #00C97A);
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 22px;
}

header h1 {
  font-size: 18px;
  color: white;
}

header p {
  font-size: 12px;
  color: #8EAEC7;
  margin-top: 2px;
}

.status-dot {
  width: 10px; height: 10px;
  background: #00C97A;
  border-radius: 50%;
  margin-left: auto;
  box-shadow: 0 0 8px #00C97A;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%,100% { opacity:1; }
  50% { opacity:0.4; }
}

.kpi-bar {
  background: #0D1F35;
  padding: 10px 24px;
  display: flex;
  gap: 24px;
  border-bottom: 1px solid #1A2A3A;
  overflow-x: auto;
}

.kpi-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  min-width: 100px;
}

.kpi-item .kpi-val {
  font-size: 16px;
  font-weight: 700;
  color: #00C97A;
}

.kpi-item .kpi-lbl {
  font-size: 10px;
  color: #8EAEC7;
  margin-top: 2px;
}

.quick-questions {
  padding: 12px 20px;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  background: #0A1628;
  border-bottom: 1px solid #1A2A3A;
}

.quick-btn {
  background: #112240;
  border: 1px solid #1C7293;
  color: #8EAEC7;
  padding: 6px 14px;
  border-radius: 20px;
  cursor: pointer;
  font-size: 12px;
  transition: all 0.2s;
}

.quick-btn:hover {
  background: #1C7293;
  color: white;
  transform: translateY(-1px);
}

.chat-container {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.message {
  display: flex;
  gap: 12px;
  max-width: 80%;
  animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
  from { opacity:0; transform:translateY(10px); }
  to   { opacity:1; transform:translateY(0); }
}

.message.user {
  align-self: flex-end;
  flex-direction: row-reverse;
}

.msg-avatar {
  width: 36px; height: 36px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 18px;
  flex-shrink: 0;
}

.msg-avatar.agent-av {
  background: linear-gradient(135deg, #1C7293, #00C97A);
}

.msg-avatar.user-av {
  background: #112240;
  border: 1px solid #1C7293;
}

.msg-bubble {
  background: #112240;
  border: 1px solid #1A2A3A;
  border-radius: 12px;
  padding: 12px 16px;
  font-size: 14px;
  line-height: 1.6;
  color: #E8F4FD;
}

.message.user .msg-bubble {
  background: #0D4F6B;
  border-color: #1C7293;
}

.msg-time {
  font-size: 10px;
  color: #8EAEC7;
  margin-top: 4px;
}

.typing {
  display: flex;
  gap: 5px;
  padding: 12px 16px;
}

.typing span {
  width: 8px; height: 8px;
  background: #1C7293;
  border-radius: 50%;
  animation: bounce 1.2s infinite;
}

.typing span:nth-child(2) { animation-delay: 0.2s; }
.typing span:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
  0%,60%,100% { transform: translateY(0); }
  30%          { transform: translateY(-8px); }
}

.input-area {
  background: #081020;
  border-top: 1px solid #1A2A3A;
  padding: 16px 20px;
  display: flex;
  gap: 12px;
  align-items: center;
}

#user-input {
  flex: 1;
  background: #112240;
  border: 1px solid #1C7293;
  border-radius: 24px;
  padding: 12px 20px;
  color: white;
  font-size: 14px;
  outline: none;
  transition: border-color 0.2s;
}

#user-input:focus {
  border-color: #00C97A;
}

#user-input::placeholder {
  color: #8EAEC7;
}

#send-btn {
  width: 46px; height: 46px;
  background: linear-gradient(135deg, #1C7293, #00C97A);
  border: none;
  border-radius: 50%;
  color: white;
  font-size: 20px;
  cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  transition: transform 0.2s;
}

#send-btn:hover { transform: scale(1.1); }
#send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
</head>
<body>

<header>
  <div class="agent-avatar">📡</div>
  <div>
    <h1>Agent IA Télécom 4G/5G</h1>
    <p>Assistant intelligent — Analyse & Prédiction QoS</p>
  </div>
  <div class="status-dot"></div>
</header>

<div class="kpi-bar">
  <div class="kpi-item">
    <span class="kpi-val">5 000</span>
    <span class="kpi-lbl">Données analysées</span>
  </div>
  <div class="kpi-item">
    <span class="kpi-val" style="color:#00C97A">55.5%</span>
    <span class="kpi-lbl">Bonne QoS</span>
  </div>
  <div class="kpi-item">
    <span class="kpi-val" style="color:#E84545">44.5%</span>
    <span class="kpi-lbl">Mauvaise QoS</span>
  </div>
  <div class="kpi-item">
    <span class="kpi-val" style="color:#00C97A">100%</span>
    <span class="kpi-lbl">RF Accuracy</span>
  </div>
  <div class="kpi-item">
    <span class="kpi-val">93.9%</span>
    <span class="kpi-lbl">LR Accuracy</span>
  </div>
  <div class="kpi-item">
    <span class="kpi-val" style="color:#FF8800">84%</span>
    <span class="kpi-lbl">RSRP Importance</span>
  </div>
</div>

<div class="quick-questions">
  <span style="font-size:12px;color:#8EAEC7;align-self:center;">Questions rapides :</span>
  <button class="quick-btn" onclick="sendQuick('Analyse l état actuel du réseau')">📊 État du réseau</button>
  <button class="quick-btn" onclick="sendQuick('Pourquoi la QoS est mauvaise ?')">❓ Causes QoS</button>
  <button class="quick-btn" onclick="sendQuick('Quels sont les KPIs les plus importants ?')">🔍 KPIs importants</button>
  <button class="quick-btn" onclick="sendQuick('Que recommandes-tu pour améliorer le réseau ?')">💡 Recommandations</button>
  <button class="quick-btn" onclick="sendQuick('Explique moi le RSRP')">📡 C est quoi RSRP</button>
  <button class="quick-btn" onclick="sendQuick('Prédis la QoS si latence est 120ms et throughput 8 Mbps')">🤖 Prédiction QoS</button>
</div>

<div class="chat-container" id="chat">
  <div class="message">
    <div class="msg-avatar agent-av">📡</div>
    <div>
      <div class="msg-bubble">
        👋 Bonjour ! Je suis votre <strong>Agent IA spécialisé en réseaux mobiles 4G/5G</strong>.<br><br>
        Je connais en détail votre projet d'analyse QoS :<br>
        ✅ <strong>5 000 données</strong> analysées<br>
        ✅ Modèle ML avec <strong>100% de précision</strong><br>
        ✅ RSRP identifié comme KPI dominant (<strong>84%</strong>)<br><br>
        Posez-moi n'importe quelle question sur votre réseau ! 🚀
      </div>
      <div class="msg-time">Agent IA • Maintenant</div>
    </div>
  </div>
</div>

<div class="input-area">
  <input type="text" id="user-input"
         placeholder="Posez votre question sur le réseau 4G/5G..."
         onkeypress="if(event.key==='Enter') sendMessage()">
  <button id="send-btn" onclick="sendMessage()">➤</button>
</div>

<script>
function getTime() {
  return new Date().toLocaleTimeString('fr-FR', {hour:'2-digit', minute:'2-digit'});
}

function addMessage(text, isUser) {
  const chat = document.getElementById('chat');
  const div = document.createElement('div');
  div.className = 'message' + (isUser ? ' user' : '');
  div.innerHTML = `
    <div class="msg-avatar ${isUser ? 'user-av' : 'agent-av'}">${isUser ? '👤' : '📡'}</div>
    <div>
      <div class="msg-bubble">${text}</div>
      <div class="msg-time">${isUser ? 'Vous' : 'Agent IA'} • ${getTime()}</div>
    </div>
  `;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function addTyping() {
  const chat = document.getElementById('chat');
  const div = document.createElement('div');
  div.className = 'message';
  div.id = 'typing-indicator';
  div.innerHTML = `
    <div class="msg-avatar agent-av">📡</div>
    <div class="msg-bubble">
      <div class="typing">
        <span></span><span></span><span></span>
      </div>
    </div>
  `;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function removeTyping() {
  const t = document.getElementById('typing-indicator');
  if(t) t.remove();
}

async function sendMessage() {
  const input = document.getElementById('user-input');
  const btn   = document.getElementById('send-btn');
  const text  = input.value.trim();
  if(!text) return;

  addMessage(text, true);
  input.value = '';
  btn.disabled = true;
  addTyping();

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text})
    });
    const data = await res.json();
    removeTyping();
    addMessage(data.response.replace(/\\n/g, '<br>'), false);
  } catch(e) {
    removeTyping();
    addMessage('❌ Erreur de connexion. Vérifiez que le serveur tourne.', false);
  }
  btn.disabled = false;
}

function sendQuick(text) {
  document.getElementById('user-input').value = text;
  sendMessage();
}
</script>
</body>
</html>
"""


# ROUTES FLASK

app = Flask(__name__)
CORS(app)

conversation_history = []

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/chat', methods=['POST'])
def chat():
    global conversation_history
    user_message = request.json.get('message', '')

    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    # Appel API Groq
    try:
        payload = json.dumps({
            "model": "llama-3.3-70b-versatile",
            "max_tokens": 1024,
            "messages": [{"role": "system", "content": CONTEXTE_PROJET}] + conversation_history[-10:]
        }).encode('utf-8')

        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                'Authorization': 'Bearer YOUR_GROQ_API_KEY',
            }
        )

        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            agent_response = result['choices'][0]['message']['content']

    except Exception as e:
        agent_response = f"❌ Erreur API : {str(e)}"

    conversation_history.append({
        "role": "assistant",
        "content": agent_response
    })

    return jsonify({"response": agent_response})

if __name__ == '__main__':
    print(" Agent IA Télécom démarré !")
    print(" Ouvre : http://localhost:5001")
    app.run(debug=False, port=5001)