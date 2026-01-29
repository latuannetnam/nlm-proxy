# Example: Using NotebookLM MCP Tool in Default Mode

This example demonstrates the **full streaming experience** with Default function calling mode.

## Setup

1. **Configure Function Calling Mode:**
   - Go to **Admin Panel** → **Settings** → **Models**
   - Select your model (e.g., GPT-4, Claude)
   - Under **Advanced Parameters**, set **Function Calling** to **"Default"**

2. **Enable the Tool:**
   - In your chat, click the **+** button
   - Select **NotebookLM MCP Client**

## Example Conversation

### User:
```
List my NotebookLM notebooks
```

### AI Response (with tool call):
```
📚 Fetching notebooks...
✅ Found 3 notebooks
```

**Tool Output:**
```markdown
## 📚 Your NotebookLM Notebooks

| Title | Notebook ID | Sources |
|-------|-------------|----------|
| Research Papers 2024 | `abc-123-def-456` | 12 |
| Meeting Notes | `xyz-789-ghi-012` | 5 |
| Product Documentation | `mno-345-pqr-678` | 8 |

*Showing 3 notebook(s)*
```

---

### User:
```
Query the Research Papers notebook: What are the key findings from the AI safety papers?
```

### AI Response (with streaming):

**Real-time progress updates appear as the query runs:**

```
🔗 Connecting to MCP server...
📝 Sending query to NotebookLM...

💭 Thinking: Analyzing 12 sources for AI safety content...

💭 Thinking: Identifying key findings across the papers...

💭 Thinking: Cross-referencing common themes in AI safety research...

💭 Thinking: Synthesizing comprehensive answer from sources...

📝 Answer:

Based on the research papers in your notebook, here are the key findings:

1. **Alignment Challenges**: Multiple papers emphasize the difficulty of aligning 
   advanced AI systems with human values, particularly as systems become more capable.

2. **Interpretability Research**: There's growing focus on making AI decision-making 
   processes more transparent and understandable to humans.

3. **Scaling Laws**: Papers discuss how safety properties may or may not scale with 
   model size, suggesting that larger models require different safety approaches.

4. **Verification Methods**: New techniques for formally verifying AI behavior are 
   being developed, though significant challenges remain.

5. **Multi-agent Considerations**: As AI systems interact, emergent behaviors create 
   new safety challenges that weren't present in single-agent scenarios.

The papers generally agree that AI safety requires continued research investment and 
coordination across the AI research community.

✅ Complete - 8 updates, 4 thinking steps

💬 Conversation ID: `conv-456-789-abc` (use for follow-ups)
```

---

### User:
```
Follow up on that - what specific verification methods were mentioned?
```

### AI Response:
```
🔗 Connecting to MCP server...
📝 Sending query to NotebookLM...

💭 Thinking: Searching conversation history for verification method details...

💭 Thinking: Extracting specific techniques from the papers...

📝 Answer:

The papers discuss several verification methods:

1. **Formal Verification**: Using mathematical proofs to verify AI behavior within 
   specified constraints.

2. **Runtime Monitoring**: Real-time systems that check AI outputs against safety 
   policies before deployment.

3. **Red Teaming**: Adversarial testing where human experts attempt to find failure 
   modes in AI systems.

4. **Mechanistic Interpretability**: Techniques for understanding internal 
   representations and computations within neural networks.

5. **Constitutional AI**: Training models with explicit principles and constraints 
   that can be verified.

The papers note that these methods often need to be combined for comprehensive 
safety assurance.

✅ Complete - 5 updates, 2 thinking steps
```

---

### User:
```
Get detailed info about the Research Papers notebook
```

### AI Response:
```
📖 Fetching notebook details...
✅ Done
```

**Tool Output:**
```markdown
## 📖 Notebook: Research Papers 2024

**ID:** `abc-123-def-456`

### 📄 Sources (12)

1. **AI Alignment: Survey and Open Problems** (`src-001`) - *PDF*
2. **Mechanistic Interpretability at Scale** (`src-002`) - *PDF*
3. **Constitutional AI: Harmlessness from AI Feedback** (`src-003`) - *PDF*
4. **Scaling Laws for AI Safety** (`src-004`) - *PDF*
5. **Runtime Monitoring for Neural Networks** (`src-005`) - *PDF*
6. **Red Teaming Language Models** (`src-006`) - *PDF*
7. **Multi-Agent Safety Considerations** (`src-007`) - *PDF*
8. **Formal Verification of ML Systems** (`src-008`) - *PDF*
9. **Adversarial Robustness in Practice** (`src-009`) - *PDF*
10. **AI Safety Benchmarks and Metrics** (`src-010`) - *PDF*
11. **Value Learning from Human Preferences** (`src-011`) - *PDF*
12. **Cooperative AI and Safety** (`src-012`) - *PDF*
```

---

## What You See in Default Mode

### Status Updates (Progress Bar)
- Top of the message shows status indicators
- Updates in real-time as processing happens
- Shows completion state when done

### Thinking Steps (In Chat)
- Each thinking step appears immediately
- Prefixed with 💭 emoji
- Gives insight into AI's reasoning process
- Useful for understanding complex queries

### Answer Streaming
- Answer appears progressively
- Can start reading before completion
- Faster perceived response time

### Conversation Context
- Conversation IDs enable follow-up questions
- AI maintains context across queries
- More natural conversational flow

## Benefits of Default Mode

1. **Transparency**: See what the AI is thinking
2. **Responsiveness**: Updates appear immediately
3. **Context**: Better understanding of reasoning process
4. **Debugging**: Can spot issues in real-time
5. **User Experience**: More engaging and interactive

## Performance Notes

- Initial connection: ~500ms
- Thinking steps: Stream in real-time (0-5 seconds each)
- Final answer: Arrives progressively
- Total time: Typically 10-30 seconds depending on query complexity
