import os
import requests
import pandas as pd
from dotenv import load_dotenv
from dune_client.client import DuneClient
from dune_client.query import QueryBase
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIG & SETUP ---
load_dotenv()
DUNE_API_KEY = os.getenv("DUNE_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
QUERY_ID = 6225590  # Твой ID запроса LINK holders

if not DUNE_API_KEY:
    raise ValueError("❌ CRITICAL ERROR: DUNE_API_KEY is missing in .env file")

client = DuneClient(DUNE_API_KEY)

# --- 2. DATA EXTRACTION (ETL) ---
print(f"🚀 Starting Whale Hunter pipeline for Query {QUERY_ID}...")
print("⏳ Fetching Top 1000 LINK Holders from Dune Analytics...")

try:
    query = QueryBase(query_id=QUERY_ID)
    response = client.get_latest_result(query)
    
    df = pd.DataFrame(response.result.rows)
    
    df['balance_link'] = pd.to_numeric(df['balance_link'])
    df['holding_days'] = pd.to_numeric(df['holding_days'])
    df['days_since_last_active'] = pd.to_numeric(df['days_since_last_active'])
    df['tx_count'] = pd.to_numeric(df['tx_count'])
    
    print(f"✅ Success! Loaded {len(df)} wallet addresses.")
    
    print("\n📊 --- Data Statistics ---")
    print(f"Median Balance: {df['balance_link'].median():,.2f} LINK")
    print(f"Whale Ceiling:  {df['balance_link'].max():,.2f} LINK")

except Exception as e:
    print(f"❌ API Error or Data Processing Fail: {e}")
    exit()

# --- 3. MACHINE LEARNING (CLUSTERING) ---
print("\n🧠 Running K-Means Clustering...")

features = ['balance_link', 'holding_days', 'tx_count', 'days_since_last_active']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

cluster_analysis = df.groupby('cluster')[features].mean()
print("\n🔎 --- Cluster Profiles (Averages) ---")
print(cluster_analysis)

cluster_stats = df.groupby('cluster')['balance_link'].mean()
RISKY_CLUSTER_ID = cluster_stats.idxmax()
avg_whale_balance = cluster_stats.max()

print(f"\n🔥 TARGET IDENTIFIED: Cluster {RISKY_CLUSTER_ID} is the 'Whale' group.")
print(f"   Avg Balance: {avg_whale_balance:,.0f} LINK")

# --- 4. VISUALIZATION ---
print("\n🎨 Generating visualization plot...")
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df, 
    x='holding_days', 
    y='balance_link', 
    hue='cluster', 
    palette='viridis', 
    s=100, 
    alpha=0.8
)

plt.title(f'LINK Wallets Segmentation (Whales = Cluster {RISKY_CLUSTER_ID})')
plt.xlabel('Days Holding')
plt.ylabel('Balance (LINK)')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.savefig('cluster_visualization.png', dpi=300)
plt.close()
print("✅ Saved chart to 'cluster_visualization.png'")

# --- 5. EXPORT RESULTS ---
godzillas = df[df['cluster'] == RISKY_CLUSTER_ID]['address'].tolist()
pd.DataFrame(godzillas, columns=['address']).to_csv('godzilla_list.csv', index=False)
print(f"💾 Saved {len(godzillas)} target wallets to 'godzilla_list.csv'")

# --- 6. TELEGRAM NOTIFICATION ---
def send_telegram_report(whales_count, top_whale_address, top_whale_balance):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram keys not found in .env - Skipping report.")
        return

    message = (
        f"🤖 <b>Daily Link Marines Report</b>\n\n"
        f"✅ <b>Clustering Completed successfully.</b>\n"
        f"🎯 <b>Strategy:</b> K-Means (Dynamic Identification)\n"
        f"🐋 <b>Whales Detected:</b> {whales_count}\n"
        f"🔥 <b>Biggest Mover:</b> <code>{top_whale_address[:6]}...{top_whale_address[-4:]}</code>\n"
        f"💰 <b>Balance:</b> {top_whale_balance:,.0f} LINK\n\n"
        f"<i>Target list updated in 'godzilla_list.csv'. Watching mode ON.</i>"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    
    try:
        r = requests.post(url, json=payload)
        if r.status_code == 200:
            print("📨 Notification sent to Telegram!")
        else:
            print(f"⚠️ Telegram API Error: {r.text}")
    except Exception as e:
        print(f"⚠️ Network Error sending to TG: {e}")

if len(godzillas) > 0:
    top_whale_idx = df[df['cluster'] == RISKY_CLUSTER_ID]['balance_link'].idxmax()
    top_whale_row = df.loc[top_whale_idx]
    
    send_telegram_report(
        whales_count=len(godzillas),
        top_whale_address=top_whale_row['address'],
        top_whale_balance=top_whale_row['balance_link']
    )
else:
    print("🤷‍♂️ No whales found in current analysis.")
