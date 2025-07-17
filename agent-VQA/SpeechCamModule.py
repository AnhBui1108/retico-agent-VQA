
import retico_core
from retico_core import *

from retico_vision.vision import ImageIU
from retico_core.text import SpeechRecognitionIU
# from retico_yolov11.yolov11 import Yolov11

# from smolAgent.smolAgents2 import SmolAgentsModule
# from retico_whisperasr.whisperasr import WhisperASRModule
# from retico_core.debug import DebugModule
# from retico_core.audio import MicrophoneModule 
# from project.agentModule import AgentsModule
# import PIL

class CombinedIU(retico_core.IncrementalUnit):
    """
    This module take 2 input IUs: SpeechRecogitionIU and DetectedObjectIU, combine it into one combinedIU

    Attributes:
        creator (AbstractModule): the module that created this IU
        previous_iu (CombinedIU): A link to the IU created before the current one
        grounded_iu (SpeechRecogitionIU): A link to the the SpeechRecogitionIU that COMMIT triggered this one 
        created_at (float): the UNIX timestamp of the moment the IU is created
        text (str) : a full user utterance
        image (bytes[]): the image of this IU

    """
    def __init__(
            self,
            creator,
            grounded_iu=None,
            image=None,
            text=None,
            previous_iu = None,
            iuid=0,
            **kwargs
            ):
        super().__init__(creator=creator,iuid = iuid)
        self.previous_iu= previous_iu
        self.image = image
        self.text = text
        self.grounded_iu =grounded_iu
        

    @staticmethod
    def type():
        return "CombinedIU"

###CombinedModule - Module to combined SpeechRecogintionIU and DetecedObjectIU

class CombinedIuModule(retico_core.AbstractModule):
    """
    A Module that produce CombinedIU containing full user utterance from SpeechRecogitionIU and DetectedObjectIU
    """

    def __init__(self, **kwargs):
        super().__init__( **kwargs)
        self.text_buffered=""
        self.latest_image = None
        self.lastest_combined_iu = None
        self.lastest_vision_iu = None
        self.lastest_speech_iu = None
        self.emit = False

    @staticmethod
    def name():
        return "CombinedIu Moduel"
    

    @staticmethod
    def description():
        return " A producing model that combine SpeechRecogitioinIU and ImageIU"
    

    @staticmethod
    def input_ius():
        return [SpeechRecognitionIU, ImageIU]
    
    @staticmethod
    def output_iu():
        return CombinedIU
    

    def process_update(self, um):
        user_text=[]
        self.emit == False
        for iu, ut in um:
            #Speech Recogition update
            if (isinstance(iu, SpeechRecognitionIU) and ut==UpdateType.COMMIT):
                user_text.append(iu.text.strip())
                self.emit= True
                self.last_SpeechIU = iu
            
            #vison update:
            if (isinstance(iu, ImageIU) and ut==UpdateType.ADD):
                self.latest_image = iu.image
                self.lastest_vision_iu = iu
                    
        
        if self.emit == True:
            self.text_buffered =" ".join(user_text)
          
        
            combined_iu = self.create_iu()
            combined_iu.previou_iu = self.lastest_combined_iu
            combined_iu.text            = self.text_buffered
            combined_iu.image           = self.latest_image
         
            combined_iu.grounded_iu = self.lastest_speech_iu
            

            um = UpdateMessage()
            um.add_iu(combined_iu, UpdateType.ADD)
            um.add_iu(combined_iu, UpdateType.COMMIT)
          
            print(f"=====TEXT======", combined_iu.text)
            

            #update the text_buffered and previu
            self.text_buffered = ""
            self.lastest_combined_iu = combined_iu
            self.emit= False
            return um
        return None


