with open('/workspaces/cloud-sre-gym/cloudscalerl/README.md', 'r') as f:
    content = f.read()

# Fix codeblock for the tree
content = content.replace(
    "    └── tasks/          # Grader Logic for Easy/Medium/Hard tasks\n### The \"Physics\" of the Cluster",
    "    └── tasks/          # Grader Logic for Easy/Medium/Hard tasks\n```\n\n### The \"Physics\" of the Cluster"
)

# Fix Tasks table
tasks_table = """Task ID\tDifficulty\tObjective\tGrader Criteria (0.0–1.0)
task1_steady\tEasy\tMaintain SLO during diurnal cycle\t1.0 if Uptime > 99.9%
task2_burst\tMedium\tHandle 10x Flash Sale spike\tWeighted: 60% SLO + 40% Budget
task3_chaos\tHard\tRegional Outage + Carbon Optimization\t50% Recovery Speed + 50% Carbon Save"""

md_tasks_table = """| Task ID | Difficulty | Objective | Grader Criteria (0.0–1.0) |
|---|---|---|---|
| task1_steady | Easy | Maintain SLO during diurnal cycle | 1.0 if Uptime > 99.9% |
| task2_burst | Medium | Handle 10x Flash Sale spike | Weighted: 60% SLO + 40% Budget |
| task3_chaos | Hard | Regional Outage + Carbon Optimization | 50% Recovery Speed + 50% Carbon Save |"""

content = content.replace(tasks_table, md_tasks_table)

# Fix Performance table
perf_table = """Metric\tStatic Rule (Standard K8s)\tCloudScaleRL Agent (LLM)\tImprovement
SLO Adherence\t84.2%\t97.8%\t+13.6%
Avg. Cost/Hour\t$18.40\t**$12.10**\t-34.2%
Carbon Footprint\t420g / kWh\t315g / kWh\t-25%
Success Score\t0.62\t0.89\tSuperior"""

md_perf_table = """| Metric | Static Rule (Standard K8s) | CloudScaleRL Agent (LLM) | Improvement |
|---|---|---|---|
| SLO Adherence | 84.2% | 97.8% | +13.6% |
| Avg. Cost/Hour | $18.40 | **$12.10** | -34.2% |
| Carbon Footprint | 420g / kWh | 315g / kWh | -25% |
| Success Score | 0.62 | 0.89 | Superior |"""

content = content.replace(perf_table, md_perf_table)

# Just in case spaces instead of tabs were used for tables
tasks_table_spaces = """Task ID Difficulty      Objective       Grader Criteria (0.0–1.0)
task1_steady    Easy    Maintain SLO during diurnal cycle       1.0 if Uptime > 99.9%
task2_burst     Medium  Handle 10x Flash Sale spike     Weighted: 60% SLO + 40% Budget
task3_chaos     Hard    Regional Outage + Carbon Optimization   50% Recovery Speed + 50% Carbon Save"""
if tasks_table_spaces in content:
    content = content.replace(tasks_table_spaces, md_tasks_table)

perf_table_spaces = """Metric  Static Rule (Standard K8s)      CloudScaleRL Agent (LLM)        Improvement
SLO Adherence   84.2%   97.8%   +13.6%
Avg. Cost/Hour  $18.40  **$12.10**      -34.2%
Carbon Footprint        420g / kWh      315g / kWh      -25%
Success Score   0.62    0.89    Superior"""
if perf_table_spaces in content:
    content = content.replace(perf_table_spaces, md_perf_table)

with open('/workspaces/cloud-sre-gym/cloudscalerl/README.md', 'w') as f:
    f.write(content)
