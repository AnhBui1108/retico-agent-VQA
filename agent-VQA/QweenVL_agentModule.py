import threading
import logging
import re
from typing import List

from smolagents import CodeAgent
from SpeechCamModule import CombinedIU
import retico_core
from retico_core import abstract
from retico_core.text import SpeechRecognitionIU, TextIU




class VLAgentModule(retico_core.abstract.AbstractModule):
    """
    SmolAgents module for Retico framework
    Acts as NLU + Dialog Manager + NLG using CodeAgent
    """
    
    def __init__(self, agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent
        self.processing = False
        self.lock = threading.Lock()
        self.logger = logging.getLogger(self.name())
        self.conversation_history = []
    
    @staticmethod
    def name() -> str:
        return "Visual - Language Agents Module"
    
    @staticmethod
    def description() -> str:
        return "Agent module for real time vision Q&A "
    
    @staticmethod
    def input_ius() -> List[type]:  
        return [CombinedIU]
    
    @staticmethod
    def output_iu():
        return TextIU
    
    def process_update(self, um) -> None:
        """Process incoming combined iu updates"""
        if self.processing:
            return
        
        for iu, ut in um:
            if (ut == abstract.UpdateType.COMMIT and isinstance(iu, CombinedIU)):
                user_text = iu.text
                image = iu.image if hasattr(iu, 'image') else None
                objects = iu.objects if hasattr(iu, 'objects') else []

                print(f"=======PROCESSING COMBINED IU=======")
                print(f"Text: {user_text}")
                print(f"Objects detected: {len(objects) if objects else 0}")
                print(f"Image available: {image is not None}")

       
        # Start processing in separate thread
        threading.Thread(
            target=self._process_with_agent, 
            args=(iu,), 
            daemon=True
        ).start()


    def _process_with_agent(self, iu) -> None:
        """Process user input with SmolAgent and send response"""
        with self.lock:
            
            self.processing = True
            user_input = iu.text
            camera_input = [iu.image]
            print(f"=====AGENT INPUT IMAGE CHECKPOINT====", camera_input)
         

            # Build context prompt
            if self.conversation_history:
                recent_context = "\n".join(self.conversation_history[-8:])  # Last 8 exchanges
                context_prompt = f"""Recent conversation:{recent_context}
                                Current user input: {user_input}"""
            else:
                context_prompt = user_input
            
            # Get agent response
            
            raw_response = self.agent.run(context_prompt, images = camera_input)
            print(f"=====USER INPUT=====", user_input)

            clean_response = self._clean_response(raw_response)
            self.logger.info(f"Response: '{clean_response}'")
            
            # Update conversation history
            camera_note = "[with image]" if camera_input and camera_input[-1] is not None else ""
            self.conversation_history.extend([
                f"User: {user_input}{camera_note}"
                f"System: {clean_response}"
            ]) 
                
            # Keep only last 16 exchanges (8 user + 8 assistant)
            if len(self.conversation_history) > 16:
                self.conversation_history = self.conversation_history[-16:]
            
                
            # Send response word by word
            self._send_response(iu, clean_response)  
            self.processing = False
    

    def _clean_response(self, response) -> str:
        """Clean agent response for speech output"""
        if not response:
            return "I didn't understand that."
        
        text = str(response)
        
        # Handle dictionary responses
        if isinstance(response, dict):
            items = [f"{k}: {v}" for k, v in response.items()]
            return f"I have collected your information: {', '.join(items)}"
        
        # Extract from final_answer() or print() calls
        patterns = [
            r'final_answer\s*\(["\']([^"\']*)["\']',
            r'print\s*\(["\']([^"\']*)["\']'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        # Clean raw text (keep essential punctuation)
        clean = text.strip()
        return clean if clean else "I'm processing that."
    

    
    def _send_response(self, iu, text: str):
        """Send response as incremental TextIUs word by word"""
        
        words = text.split()
        if not words:
            return
        
        output_ius = []
        
        # Send each word as ADD message
        for word in words:
            output_iu = self.create_iu(iu)
            output_iu.text = word
            output_iu.payload =word
            output_ius.append(output_iu)
            
            
            add_msg = retico_core.UpdateMessage.from_iu(
                output_iu, 
                retico_core.UpdateType.ADD
            )
            self.append(add_msg)

        
        # Send final COMMIT message
        if output_ius:
                last_iu = output_ius[-1]
                commit_msg = retico_core.UpdateMessage.from_iu(
                    last_iu,
                    retico_core.UpdateType.COMMIT
                )
                self.append(commit_msg)
            
        
    
    def setup(self) -> None:
        super().setup()
        self.logger.info("SmolAgents module ready")
    

    def shutdown(self) -> None:
        super().shutdown()
        self.processing = False
        self.logger.info("SmolAgents module shutdown")