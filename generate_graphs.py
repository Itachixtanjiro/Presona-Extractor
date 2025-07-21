import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_visualizations():
    """
    Reads benchmark data from CSV and generates comparison graphs.
    """
    csv_file = 'Chunkers/rag_combinations_benchmark.csv'
    output_dir = 'visualizations'

    if not os.path.exists(csv_file):
        print(f"Error: Benchmark file not found at {csv_file}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    df = pd.read_csv(csv_file)

    # --- Data Preparation ---
    # Extract model names and relevant metrics
    model_names = [
        'RAG+Longformer', 
        'RAG+Reformer', 
        'RAG+Quantized'
    ]
    scores = [df['RAG+Longformer Score'][0], df['RAG+Reformer Score'][0], df['RAG+Quantized Score'][0]]
    times = [df['RAG+Longformer Time (s)'][0], df['RAG+Reformer Time (s)'][0], df['RAG+Quantized Time (s)'][0]]

    # --- 1. Performance Score Comparison Chart ---
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, scores, color=['#4c72b0', '#55a868', '#c44e52'])
    plt.ylabel('Performance Score (Higher is Better)')
    plt.title('Model Performance Score Comparison')
    plt.ylim(0, max(scores) * 1.1) 
    
    # Add score labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center')

    score_graph_path = os.path.join(output_dir, 'score_comparison.png')
    plt.savefig(score_graph_path)
    print(f"Saved score comparison graph to {score_graph_path}")
    plt.close()


    # --- 2. Processing Time Comparison Chart ---
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, times, color=['#4c72b0', '#55a868', '#c44e52'])
    plt.ylabel('Processing Time in Seconds (Lower is Better)')
    plt.title('Model Processing Time Comparison')
    plt.ylim(0, max(times) * 1.1)

    # Add time labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}s', va='bottom', ha='center')

    time_graph_path = os.path.join(output_dir, 'time_comparison.png')
    plt.savefig(time_graph_path)
    print(f"Saved time comparison graph to {time_graph_path}")
    plt.close()

if __name__ == '__main__':
    print("Generating graphs from benchmark data...")
    generate_visualizations()
    print("Done.")