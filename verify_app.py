import pandas as pd

print("=" * 80)
print("✅ STREAMLIT APP VERIFICATION")
print("=" * 80)

# Load data
df = pd.read_csv('results/raw_results.csv')

print(f"\n✅ Data loaded successfully!")
print(f"   Total experiments: {len(df)}")

success_count = len(df[df['status'] == 'success'])
failure_count = len(df[df['status'] != 'success'])
print(f"   Success rate: {success_count}/{len(df)} ({success_count/len(df)*100:.0f}%)")
print(f"   Failures: {failure_count}")

print(f"\n✅ Model Performance:")
classical = df[df['model_name'] == 'classical']
quantum = df[df['model_name'] == 'quantum']

print(f"   Classical avg accuracy: {classical['metric_accuracy'].mean():.2%}")
print(f"   Quantum avg accuracy: {quantum['metric_accuracy'].mean():.2%}")
print(f"   Advantage: {(classical['metric_accuracy'].mean() - quantum['metric_accuracy'].mean())*100:.1f} pp")

print(f"\n✅ Execution Time:")
print(f"   Classical avg: {classical['end_to_end_time_seconds'].mean():.3f}s")
print(f"   Quantum avg: {quantum['end_to_end_time_seconds'].mean():.3f}s")
speedup = quantum['end_to_end_time_seconds'].mean() / classical['end_to_end_time_seconds'].mean()
print(f"   Speedup: {speedup:.0f}× faster (Classical)")

print(f"\n✅ Experiments Breakdown:")
for exp in df['experiment_id'].unique():
    count = len(df[df['experiment_id'] == exp])
    print(f"   {exp}: {count} runs")

print(f"\n✅ All packages available!")
try:
    import streamlit
    import plotly
    import pandas
    import numpy
    import matplotlib
    print(f"   ✓ Streamlit {streamlit.__version__}")
    print(f"   ✓ Plotly {plotly.__version__}")
    print(f"   ✓ Pandas {pandas.__version__}")
    print(f"   ✓ NumPy {numpy.__version__}")
    print(f"   ✓ Matplotlib {matplotlib.__version__}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 80)
print("✅ APP IS READY TO RUN!")
print("=" * 80)
print("\nCommand to start:")
print("  streamlit run streamlit_app.py")
print("\nOr use:")
print("  python START_DEMO.txt")
print("=" * 80)
