# retico-agent-VQA

## Overview
Point&Ask is a prototype project that demonstrates real-time visual question answering capabilities. Users can point their camera at objects and ask spoken questions, receiving comprehensive answers with conversational follow-up support. The system integrates live camera feeds with speech recognition through a unified multimodal processing pipeline, utilizing intelligent agents for dynamic external knowledge retrieval and tool access.

## Features:
- **Real-time Camera Input**: Continuous processing of live camera feeds
- **Speech Recognition**: Whisper ASR for natural voice interaction
- **Rich Visual Understanding**: End-to-end scene analysis without traditional CV preprocessing
- **Conversational Context**: Multi-turn dialogue with context maintenance
- **External Knowledge Access**: Agent-based tools for real-time information retrieval
- **Multimodal Integration**: Unified processing through Qwen2.5-VL

### Example Interactions
- Object identification: *"What is this plant?"*
- Current information: *"What is the price of this car today?"*
- Contextual analysis: *"What do you see in this scene?"*
- Follow-up questions: Maintains context across 2-3 related questions

## Installation:
### Prerequisites
- Python 3.10+
- smolAgents 1.19.x
### Set up
**Clone the repository:**

```bash
git clone https://github.com/AnhBui1108/retico-agent-VQA.git
cd retico-agent-VQA
```
You also need to add to your Python path the retico-vision, retico-core module you can download from the Retico GitHub repository.

**Configure API access:** Edit runner_project.py and add your API key 

```bash
model = OpenAIServerModel(
    model_id= "qwen/qwen2.5-vl-32b-instruct:free",
    api_base = "https://openrouter.ai/api/v1",
    stream = False,
    api_key="YOUR_API_KEY")
```
