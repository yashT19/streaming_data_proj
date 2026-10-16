import pandas as pd

df = pd.read_csv('results/raw_results.csv')
exp_a = df[df['experiment_id'] == 'A_performance']
classical = exp_a[exp_a['model_name'] == 'classical']
quantum = exp_a[exp_a['model_name'] == 'quantum']

print('EXPERIMENT A (Main Performance Comparison):')
print('='*60)
print('\nCLASSICAL:')
print(f'  Accuracy:  {classical["metric_accuracy"].mean():.4f} = {classical["metric_accuracy"].mean()*100:.2f}%')
print(f'  Precision: {classical["metric_precision"].mean():.4f} = {classical["metric_precision"].mean()*100:.2f}%')
print(f'  Recall:    {classical["metric_recall"].mean():.4f} = {classical["metric_recall"].mean()*100:.2f}%')
print(f'  F1-Score:  {classical["metric_f1"].mean():.4f}')
print(f'  Time:      {classical["end_to_end_time_seconds"].mean():.6f}s')

print('\nQUANTUM:')
print(f'  Accuracy:  {quantum["metric_accuracy"].mean():.4f} = {quantum["metric_accuracy"].mean()*100:.2f}%')
print(f'  Precision: {quantum["metric_precision"].mean():.4f} = {quantum["metric_precision"].mean()*100:.2f}%')
print(f'  Recall:    {quantum["metric_recall"].mean():.4f} = {quantum["metric_recall"].mean()*100:.2f}%')
print(f'  F1-Score:  {quantum["metric_f1"].mean():.4f}')
print(f'  Time:      {quantum["end_to_end_time_seconds"].mean():.6f}s')

speedup = quantum['end_to_end_time_seconds'].mean() / classical['end_to_end_time_seconds'].mean()
print(f'\nSpeedup: {speedup:.0f}× faster (Classical)')
print('='*60)

print('\nValues for Streamlit/Presentation:')
print(f'Classical Accuracy:   {classical["metric_accuracy"].mean()*100:.2f}%')
print(f'Quantum Accuracy:     {quantum["metric_accuracy"].mean()*100:.2f}%')
print(f'Classical Precision:  {classical["metric_precision"].mean()*100:.2f}%')
print(f'Quantum Precision:    {quantum["metric_precision"].mean()*100:.2f}%')
print(f'Classical Recall:     {classical["metric_recall"].mean()*100:.2f}%')
print(f'Quantum Recall:       {quantum["metric_recall"].mean()*100:.2f}%')
print(f'Classical F1:         {classical["metric_f1"].mean():.4f}')
print(f'Quantum F1:           {quantum["metric_f1"].mean():.4f}')
print(f'Speed Advantage:      {speedup:.0f}×')
