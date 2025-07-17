import os, sys, requests

import retico_core

os.environ['GOOGLE_APPLICATION_CREDENTIALS']='/Users/avocado/.gcloud/keys/protean-speech.json'


os.environ['RETICO'] = 'retico_core'
os.environ['VISION'] = 'retico-vision'
os.environ['WS']='retico-whisperasr'
os.environ['AG']='smolAgent'
os.environ['AGM'] ="project"
os.environ['TTS'] = "retico-speechbraintts"


sys.path.append(os.environ['RETICO'])
sys.path.append(os.environ['WS'])
sys.path.append(os.environ['AG'])
sys.path.append(os.environ['VISION'])
sys.path.append(os.environ['AGM'])
sys.path.append(os.environ['TTS'])
sys.path.append("..")


from retico_core import *
from retico_vision.vision import WebcamModule
from retico_core.text import SpeechRecognitionIU


from smolAgent.smolAgents2 import SmolAgentsModule
from retico_whisperasr.whisperasr import WhisperASRModule
from retico_core.debug import DebugModule
from retico_core.audio import MicrophoneModule, SpeakerModule
from QweenVL_agentModule import VLAgentModule
from SpeechCamModule import CombinedIU, CombinedIuModule
from retico_speechbraintts.speechbraintts import SpeechBrainTTSModule
from retico_googletts.googletts import GoogleTTSModule

import smolagents 

from smolagents import CodeAgent, DuckDuckGoSearchTool
from smolagents.models import OpenAIServerModel
from smolagents import Tool


# Create Agent
prompt = """
You are a helpful AI assistant designed to process user requests efficiently and accurately.


CRITICAL:
1. Task Assessment: Evaluate if you have sufficient information to complete the user's request
2. Immediate Response: If you can complete the task, provide your response in final_answer() immediately
3. Clarification: If information is missing or unclear, ask specific clarifying questions in final_answer()
4. Step limit: maximum of three (03) processing step
5. Image Processing: Fully utilize the model's visual capabilities before generating any code solutions
6. Number Conversion: Convert all numerical values to words (e.g., "5 cars" becomes "five cars") 

EXAMPLE FORMAT:
<code>
final_answer("I can see a pair of tortoiseshell glasses and TWO tube of lip gloss on a wooden surface.")
</code>
"""


class GoogleSearchTool(Tool):
    name = "google_search"
    description = "Search the web using Google Custom Search API"
    inputs = {
        "query": {
            "type": "string", 
            "description": "The search query to execute"
        }
    }
    output_type = "string"
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        print("Initializing GoogleSearchTool...")
        self.search = GoogleSearchAPIWrapper(
            google_api_key="YOUR-GOOGLE-API-KEY", # Insert your google API key here
            google_cse_id="YOUR-CSE-ID" # Insert your CSE ID here
        )

    
    def forward(self, query: str) -> str:
        """Search the web for the given query"""
        try:
            results = self.search.run(query)
            return results
        except Exception as e:
            return f"Search failed: {str(e)}"       

model = OpenAIServerModel(
    model_id= "qwen/qwen2.5-vl-32b-instruct:free",
    api_base = "https://openrouter.ai/api/v1",
    stream = False,
    api_key="YOUR_API_KEY")  # Insert your API Key here

agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
    instructions = prompt,
   
)

#Pipeline
mic = MicrophoneModule(chunk_size = 320, rate = 16000)
asr = WhisperASRModule()
cam = WebcamModule()
debug = DebugModule()

cb = CombinedIuModule()
dm = VLAgentModule(agent)
tts = SpeechBrainTTSModule()
speaker = SpeakerModule(rate=22050)


tts.setup()


mic.subscribe(asr)
asr.subscribe(cb)
cam.subscribe(cb)
cb.subscribe(dm)
dm.subscribe(tts)
tts.subscribe(speaker)
dm.subscribe(debug)
asr.subscribe(debug)


mic.run()
asr.run()
cam.run()
# yolo.run()
cb.run()
dm.run()
# tts.run()
# speaker.run()
debug.run()

input()

asr.stop()
mic.stop()
cam.stop()
cb.stop()
dm.stop()
tts.stop()
speaker.stop()

debug.stop()





        
    
    