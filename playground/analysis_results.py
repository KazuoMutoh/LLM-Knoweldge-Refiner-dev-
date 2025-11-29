import pandas as pd
import matplotlib.pyplot as plt

dict_table = {}
for i in range(2,11):
    df_i = pd.read_pickle(f'/app/experiments/20251105/try2/iter_{i}/df_score_diff.pkl')

    if i == 2:
        dict_table[0] = {'mean_score':df_i['score (i)'].mean(), 'num. of added triples':df_i['num. of added triples'].sum()}
    
    dict_table[i-1] = {'mean_score':df_i['score (i+1)'].mean(), 'num. of added triples':df_i['num. of added triples'].sum()}

df_table = pd.DataFrame.from_dict(dict_table, orient='index')
df_table.index.name = 'iteration'
df_table.reset_index(inplace=True)  
df_table['cumulative num. of added triples'] = df_table['num. of added triples'].cumsum()

fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
axs[0].plot(df_table['iteration'], df_table['mean_score'], marker='o')
axs[0].set_ylabel('Mean Score', fontsize=16)
axs[0].set_title('Mean Score vs. Iteration', fontsize=16)
axs[0].tick_params(axis='both', labelsize=16)
axs[0].grid() 
axs[1].plot(df_table['iteration'], df_table['cumulative num. of added triples'], marker='o', color='orange')
axs[1].set_xlabel('Iteration', fontsize=16)
axs[1].set_ylabel('Cumulative Number \n of Added Triples', fontsize=16)
axs[1].set_title('Cumulative Number of Added Triples vs. Iteration', fontsize=16)
axs[1].tick_params(axis='both', labelsize=16)
axs[1].grid() 
plt.tight_layout()
plt.savefig('/app/experiments/20251105/try2/analysis_results/mean_score_cumulative_added_triples.png')

list_report_rules_update = []
for i in range(1,11):
    dir_iter = f'/app/experiments/20251105/try2/iter_{i}/'
    df_i = pd.read_csv(dir_iter + 'rules.csv')
    list_report_rules_update.append(df_i['body'].tolist())


md_report = '# Report on Triple Addition Rules\n\n'

md_report += '## Summary Table\n\n'
md_report += df_table.to_markdown(index=False) + '\n\n' 
# add plots
md_report += '## Plots\n\n'
md_report += '### Mean Score and Cumulative Number of Added Triples vs. Iteration\n\n'
md_report += '![Mean Score and Cumulative Number of Added Triples vs. Iteration](mean_score_cumulative_added_triples.png)\n\n'

md_report += '## Triple Addtion Rules\n\n'
for i, rules in enumerate(list_report_rules_update[:-1], start=1):
    md_report += f'### Iteration {i-1} -> {i}\n\n'
    for rule in rules:
        md_report += f'- {rule}\n'
    md_report += '\n'
with open('/app/experiments/20251105/try2/analysis_results/report_rules_update.md', 'w') as f:
    f.write(md_report)
    



