import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

def main():
    db_path = "db/cadx.db"
    if not os.path.exists(db_path):
        print(f"[ERROR] Database {db_path} not found.")
        return
        
    conn = sqlite3.connect(db_path)
    # Query all metric logs
    query = """
        SELECT epoch, val_loss, kappa, batch_size, ts 
        FROM metrics 
        WHERE epoch IS NOT NULL 
        ORDER BY ts ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("[WARNING] No metrics recorded in the database yet.")
        return
        
    print(f"[INFO] Loaded {len(df)} metrics entries.")
    
    # Sort by epoch and filter duplicates by taking the latest timestamp for each epoch
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.sort_values('ts').groupby('epoch').last().reset_index()
    
    print(df)
    
    # Create logs dir if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Val Loss', color=color)
    line1 = ax1.plot(df['epoch'], df['val_loss'], color=color, marker='o', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Val QWK (Kappa)', color=color)
    line2 = ax2.plot(df['epoch'], df['kappa'], color=color, marker='s', label='Val QWK')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.title('Prostate CADx Training Metrics (Contrastive & Supervised Loss)')
    fig.tight_layout()
    
    output_path = "logs/metrics_curve.png"
    plt.savefig(output_path, dpi=300)
    print(f"[INFO] Plotted training metrics successfully to {output_path}")

if __name__ == "__main__":
    main()
