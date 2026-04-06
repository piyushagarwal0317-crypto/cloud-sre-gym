import re

with open('/workspaces/cloud-sre-gym/cloudscalerl/README.md', 'r') as f:
    content = f.read()

# Remove the conversational part at the end
content = re.sub(r'Where to put the Statistics part\?.*', '', content, flags=re.DOTALL)

# Add standard markdown headings if missing
content = re.sub(r'^CloudScaleRL 🚀', '# CloudScaleRL 🚀', content)
content = re.sub(r'^\s*🌌 The Problem & Utility \(30%\)', '## 🌌 The Problem & Utility (30%)', content, flags=re.MULTILINE)
content = re.sub(r'^\s*The "Physics" of the Cluster', '### The "Physics" of the Cluster', content, flags=re.MULTILINE)
content = re.sub(r'^\s*🎮 Tasks & Graders \(25%\)', '## 🎮 Tasks & Graders (25%)', content, flags=re.MULTILINE)
content = re.sub(r'^\s*📊 Baseline Statistics & Results', '## 📊 Baseline Statistics & Results', content, flags=re.MULTILINE)
content = re.sub(r'^\s*Performance Comparison', '### Performance Comparison', content, flags=re.MULTILINE)
content = re.sub(r'^\s*🛠️ Environment Design \(20%\)', '## 🛠️ Environment Design (20%)', content, flags=re.MULTILINE)
content = re.sub(r'^\s*Reward Function', '### Reward Function', content, flags=re.MULTILINE)
content = re.sub(r'^\s*🚀 Setup & Deployment', '## 🚀 Setup & Deployment', content, flags=re.MULTILINE)
content = re.sub(r'^\s*Local Execution \(Docker\)', '### Local Execution (Docker)', content, flags=re.MULTILINE)
content = re.sub(r'^\s*Running the Agent', '### Running the Agent', content, flags=re.MULTILINE)
content = re.sub(r'^\s*Spec Validation', '### Spec Validation', content, flags=re.MULTILINE)
content = re.sub(r'^\s*💡 Creativity & Novelty \(10%\)', '## 💡 Creativity & Novelty (10%)', content, flags=re.MULTILINE)

# Fix Math formatting
old_math = """L 
p99
​
 =L 
base+( 
Capacity
RPS
​
 ) 
4"""
new_math = "$$L_{p99} = L_{base} + \\left(\\frac{RPS}{Capacity}\\right)^4$$"
content = content.replace(old_math, new_math)

old_math_2 = """R=(W 
avail
​
 ⋅SLO)−(W 
cost
​
 ⋅Cost)−(W 
stab
​
 ⋅Thrash)"""
new_math_2 = "$$R = (W_{avail} \\cdot SLO) - (W_{cost} \\cdot Cost) - (W_{stab} \\cdot Thrash)$$"
content = content.replace(old_math_2, new_math_2)

# Fix code blocks for bash
content = content.replace("Bash\ndocker build -t cloudscalerl ./server\ndocker run -p 8000:8000 cloudscalerl", "```bash\ndocker build -t cloudscalerl ./server\ndocker run -p 8000:8000 cloudscalerl\n```")
content = content.replace("Bash\nexport OPENAI_API_KEY='your_key_here'\npython client.py --task task2_burst", "```bash\nexport OPENAI_API_KEY='your_key_here'\npython client.py --task task2_burst\n```")
content = content.replace("Bash\nopenenv validate http://localhost:8000", "```bash\nopenenv validate http://localhost:8000\n```")

# Add missing punctuation
content = content.replace("CloudScaleRL is a high-fidelity,", "CloudScaleRL is a high-fidelity")

with open('/workspaces/cloud-sre-gym/cloudscalerl/README.md', 'w') as f:
    f.write(content.strip() + "\n")
