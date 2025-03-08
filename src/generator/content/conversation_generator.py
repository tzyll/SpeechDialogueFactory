from utils.llm import LLM
from typing import Optional, List, Literal
import json
import logging
from data.dialogue import Conversation, Dialogue

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_TEMPLATE = """
You are a dialogue generator for creating natural, realistic conversations. Your task is to generate a complete conversation in JSON format based on the provided metadata and script outline. The resulting dialogue should feel authentic and true to the characters' personalities, their situation, and the narrative flow.

**INPUT FORMAT:**  
You will receive:  
1. **Scenario (JSON)** containing dialogue type, temporal context, spatial context, cultural background, language, and a custom prompts.
1. **Metadata (JSON)** containing setting, character details, personalities, relationship dynamics, and context for the scene.  
2. **Script outline** with scene description and narrative flow (including key points and events to cover).

**OUTPUT REQUIREMENTS:**  
Generate a **single JSON object** with a `"utterances"` array. Each element in the array represents a single turn in the conversation. Include the following fields in each turn:  
- **speaker_id**: Either "role_1" or "role_2" to match the role in the metadata
- **speaker_name**: The character's name (as defined in metadata)
- **text**: The character's spoken dialogue  
- **emotion**: A brief description of the character's emotional state (e.g., "curious", "slightly nervous", "enthusiastic")  
- **speech_rate**: One of `["slow", "medium", "fast"]`  
  - Use "slow" for moments of careful thought, emphasis, or complexity.  
  - Use "medium" for normal-paced conversation.  
  - Use "fast" for excitement, urgency, or nervousness.  
- **pause_after**: One of `["short", "medium", "long"]`  
  - Reflect the natural flow of speech. For instance, after a big reveal or a thoughtful statement, consider a "long" pause. After a quick, casual remark, a "short" pause might suffice.
- **tts_prompt**: A concise natural language prompt describing ONLY how the text should be spoken by a text-to-speech model, focusing exclusively on paralinguistic features like tone, pitch, rhythm, and vocal qualities. DO NOT reference the content of what is being said.

**LANGUAGE CONSISTENCY:**
You must generate the dialogue in the language specified in the metadata. If the metadata indicates "Chinese", all dialogue should be in Chinese. If it indicates "English", all dialogue should be in English. Ensure that speech patterns, idioms, and expressions are appropriate for the specified language and cultural context.

**DIALOGUE GUIDELINES:**  
1. **Natural Speech Patterns:**  
   - Incorporate small hesitations, filler words (e.g., "um", "uh", "well...") where appropriate.  
   - Use contractions ("I'm", "don't", "we're") to sound more natural.  
   - Allow for occasional incomplete thoughts or self-corrections.  
   - Avoid overly polished or robotic phrasing—conversations should sound like real people talking.
   - Make sure the dialogue is in the language specified in the metadata, while respecting cultural backgrounds.

2. **Character Consistency:**  
   - Reflect each character's personality traits, background, and relationship to others (as provided in metadata).  
   - Keep each character's language style consistent. For example, if a character is warm and friendly, their dialogue should often include informal niceties, humor, or supportive remarks.  
   - If the character is more formal or reserved, their speech might contain more careful phrasing and fewer slang terms.

3. **Conversation Flow:**  
   - Follow the narrative structure from the script outline, ensuring all key story beats are covered.  
   - Introduce and transition between topics smoothly. Characters should respond naturally to each other's prompts, questions, and statements.  
   - Include moments of genuine small talk or personal connection before and after hitting key plot points, when appropriate.  
   - Ensure realistic turn-taking: characters may react to what was just said, ask clarifying questions, or acknowledge previous statements before moving on.

4. **Emotional Nuance:**  
   - Emotional descriptions in the "emotion" field should feel authentic. For example, if the character is excited about a new opportunity, "hopeful" or "enthusiastic" might fit; if they're caught off guard, "surprised" or "unsure" might be suitable.  
   - Vary emotional states as the conversation progresses, reflecting changes in mood and context.

5. **Logical Progression & Key Points:**  
   - Ensure the dialogue feels like a coherent exchange, not just disconnected lines.  
   - Cover all necessary plot elements from the script outline, but do so naturally (avoid abruptly forcing in key points without any conversational lead-in).

6. **Technical Adherence:**  
   - Only output valid JSON.  
   - Use the exact strings for `speech_rate` and `pause_after` as specified.  
   - Match the number of turns specified in the metadata (if provided).  
   - Keep the conversation self-contained and consistent.
   - Ensure speaker_id correctly matches the role in metadata (role_1 or role_2).

**TTS PROMPT GUIDELINES:**
For the tts_prompt field:
- IMPORTANT: Focus EXCLUSIVELY on paralinguistic features (HOW something is said), NOT on the content (WHAT is said)
- DO NOT reference the specific content, topic, or subject matter of the utterance
- DO include:
  - Voice characteristics
  - Emotional tone
  - Speaking style
  - Speaker age/gender
  - Pace variations
  - Volume patterns
  - Pitch patterns
- Keep it concise (1-2 short sentences)
- Example of GOOD tts_prompt: "Elderly male speaking slowly with a gentle, warm tone. Slightly quavering voice with soft volume."
- Example of BAD tts_prompt: "Man explaining his concerns about education with a worried tone, emphasizing the importance of finding a job."

**EXAMPLE OUTPUT FORMAT:**
```json
{
  "utterances": [
    {
      "speaker_id": "role_2",
      "speaker_name": "Sarah",
      "text": "Hey Mike! Um... could I get my usual coffee please?",
      "emotion": "friendly",
      "speech_rate": "medium",
      "pause_after": "short",
      "tts_prompt": "Young female voice, friendly and casual tone with slight hesitation. Normal pitch with a hint of cheerfulness."
    },
    {
      "speaker_id": "role_1",
      "speaker_name": "Mike",
      "text": "Sure thing! I was actually just thinking of trying a new blend. Wanna be my taste-tester?",
      "emotion": "enthusiastic",
      "speech_rate": "medium",
      "pause_after": "medium",
      "tts_prompt": "Middle-aged male voice speaking warmly with gradually increasing enthusiasm. Slightly rising pitch at the end."
    }
  ]
}
```
"""

USER_PROMPT_TEMPLATE = """
# Scenario
```json
{scenario}
```

# Metadata
```json
{metadata}
```

# Script
{script}
"""


class DialogueGenerator:
    def __init__(self, llm: LLM):
        self.llm = llm

    def _construct_prompt(self, dialogues):
        messages = []
        for dialogue in dialogues:
            message = [
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
                {
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(
                        scenario=dialogue.scenario.to_json(pretty=True),
                        metadata=dialogue.metadata.to_json(pretty=True),
                        script=dialogue.script,
                    ),
                },
            ]
            messages.append(message)
        return messages

    def _fill_back(self, outputs, dialogues):
        remaining_dialogues = []
        for i in outputs["success_indices"]:
            utterances = outputs["responses"][i]
            dialogues[i].conversation = Conversation.model_validate(utterances)
            remaining_dialogues.append(dialogues[i])
        return remaining_dialogues

    def generate_dialogue(
        self,
        dialogues: List[Dialogue],
        gen_params={},
    ):
        prompt = self._construct_prompt(dialogues)
        logger.info(f"Generating {len(prompt)} conversations...")
        outputs = self.llm.generate(prompt, Conversation, **gen_params)
        remaining_dialogues = self._fill_back(outputs, dialogues)
        logger.info(f"Received {len(remaining_dialogues)} conversations from LLM.")
        return dialogues
