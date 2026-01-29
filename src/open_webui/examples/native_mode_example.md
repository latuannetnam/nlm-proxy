# Example: Using NotebookLM MCP Tool in Native Mode

This example shows what happens in **Native (Agentic) function calling mode**.

## Setup

1. **Configure Function Calling Mode:**
   - Go to **Admin Panel** → **Settings** → **Models**
   - Select your model
   - Under **Advanced Parameters**, set **Function Calling** to **"Native"**

2. **Enable the Tool:**
   - In your chat, click the **+** button
   - Select **NotebookLM MCP Client**

## What Works in Native Mode

✅ **Status Updates** - Progress indicators  
✅ **Final Answers** - Complete responses  
✅ **Error Handling** - User-friendly errors  
✅ **Citations** - Source references  
✅ **Health Checks** - Server connectivity  

## What Doesn't Work

❌ **Thinking Steps** - Get overwritten by completion snapshots  
❌ **Answer Streaming** - Content appears all at once  
❌ **Message Events** - Replaced by native completion  
❌ **Progressive Updates** - Only final result visible  

## Example Conversation

### User:
```
List my NotebookLM notebooks
```

### AI Response:
```
[Tool call in progress...]
```

**Tool Output (appears all at once):**
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
Query the Research Papers notebook: What are the key findings?
```

### AI Response:

**You'll see a warning notification:**
```
⚠️ Native function calling mode detected. Streaming may be limited. 
Switch to Default mode for full real-time experience.
```

**Then the tool processes...**

[No visible thinking steps - they get overwritten]

**Final answer appears all at once:**
```
Based on the research papers in your notebook, here are the key findings:

1. **Alignment Challenges**: Multiple papers emphasize the difficulty...
2. **Interpretability Research**: There's growing focus on...
3. **Scaling Laws**: Papers discuss how safety properties...
4. **Verification Methods**: New techniques for formally verifying...
5. **Multi-agent Considerations**: As AI systems interact...

The papers generally agree that AI safety requires continued research...

💬 Conversation ID: `conv-456-789-abc` (use for follow-ups)
```

---

## Comparison: Default vs Native Mode

### Query Processing Time Visualization

**Default Mode:**
```
Time: 0s ──────> 5s ──────> 10s ──────> 15s ──────> 20s
      Connect    Think #1    Think #2    Think #3    Answer
      [visible]  [visible]   [visible]   [visible]   [visible]
```

**Native Mode:**
```
Time: 0s ──────────────────────────────────────────> 20s
      Connect  [hidden processing]                   Answer
      [visible]                                      [visible]
```

### User Experience

| Aspect | Default Mode | Native Mode |
|--------|-------------|-------------|
| **Feedback** | Continuous updates | Loading spinner only |
| **Transparency** | See reasoning process | Black box |
| **Perceived Speed** | Feels faster (progressive) | Feels slower (wait for all) |
| **Debugging** | Easy to spot issues | Hard to diagnose |
| **Engagement** | Interactive | Passive waiting |

### When to Use Native Mode

Despite limitations, Native mode may be preferable when:

1. **Simple Queries**: Quick lookups where streaming isn't needed
2. **Batch Processing**: Running multiple queries programmatically
3. **Model Requirements**: Your model works better with native calling
4. **Reduced Latency**: Native mode has slightly lower overhead
5. **System Tools**: Using built-in Open WebUI agentic tools

## Technical Explanation

### Why Native Mode Breaks Streaming

In Native mode:

1. **Model Controls Tool Calls**: The LLM directly invokes tools using its native API
2. **Completion Snapshots**: Server sends repeated full-content snapshots via `chat:completion` events
3. **Content Replacement**: Client replaces entire message with each snapshot
4. **Event Overwriting**: Tool-emitted `message` events get replaced by completion snapshots
5. **Result**: Thinking steps flicker and disappear

### Architecture Difference

**Default Mode Flow:**
```
User → Open WebUI → Tool → Progress Events → UI Updates (preserved)
                                    ↓
                            Final Result → UI
```

**Native Mode Flow:**
```
User → Open WebUI → Model (Native) → Tool → Progress Events → UI (overwritten)
                              ↓                                      ↓
                        Completion Snapshots ──────────> UI (replaces everything)
```

## Recommendations

### For Best Experience
- **Use Default Mode** when working with NotebookLM queries
- Long-running queries benefit most from real-time updates
- Thinking steps provide valuable context

### For Compatibility
- Native mode still works, just with limited streaming
- All core functionality remains available
- Consider it for specific model requirements

### Configuration Tips

You can set mode **per-chat** without changing admin settings:

1. Click **Chat Controls** in the chat interface
2. Go to **Advanced Params**
3. Set `function_calling` to `"default"` or `"native"`

This overrides the global setting for that conversation only.

## Example: Mode Detection

The tool automatically detects which mode you're using:

**Default Mode:**
```
[Smooth streaming experience with all features]
```

**Native Mode:**
```
⚠️ Native function calling mode detected. Streaming may be limited.
Switch to Default mode for full real-time experience.

[Tool proceeds with limited streaming]
```

## Troubleshooting Native Mode

### "Why don't I see thinking steps?"

**Explanation**: Native mode's completion snapshots overwrite progress events.

**Solution**: Switch to Default mode for thinking steps.

### "Tool still works, just no streaming?"

**Confirmation**: Yes! The tool is fully functional in Native mode.

**Trade-off**: You get the final answer, just not the intermediate steps.

### "Can I force streaming in Native mode?"

**Answer**: No - it's an architectural limitation of how Native mode works.

**Alternative**: Use status events only (they work in both modes).

## Summary

- ✅ Native mode **works** but with **limited streaming**
- ✅ Best for **simple queries** and **quick lookups**
- ⚠️ **Switch to Default mode** for full streaming experience
- 📊 Native mode has **slightly lower latency**
- 🎯 Choose mode based on your **use case** and **model requirements**
