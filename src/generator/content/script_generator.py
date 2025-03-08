from utils.llm import LLM
from pydantic import BaseModel, Field
from typing import Optional, List
import json
import logging

logger = logging.getLogger(__name__)
from data.dialogue import DialogueScenario, Dialogue, Metadata


SYSTEM_PROMPT_TEMPLATE = """
You are a conversation script designer. Your task is to create a detailed script outline for a two-person conversation based on the provided metadata. This script will serve as a blueprint for generating the final conversation.

Output your script in a structured markdown format with these exact sections:

### SCENE:
Write a brief paragraph describing the physical setting, atmosphere, and initial situation of the characters. Use specific details from the metadata's setting and context.

### NARRATIVE FLOW:
Break down the conversation into 4 clear sections (must match exactly with expected_turns in metadata):
1. Opening (usually 2-3 turns)
2. Initial Discussion (2-4 turns)
3. Main Discussion (3-5 turns)
4. Wrapping Up (2-3 turns)

For each section:
- List specific dialogue beats
- Show clear progression of main_topic
- Include all key_points from metadata

### CHARACTER BEHAVIORS:
For each character (role_1 and role_2):
- List 3-5 specific speaking patterns or behavioral traits
- Base these on personality_traits from metadata
- Include their typical reactions and mannerisms
- Show how their occupation and background influence their speech

### EMOTIONAL PROGRESSION:
Outline 3 stages of emotional development:
- Start: Initial emotional state of both characters
- Middle: How emotions evolve during main discussion
- End: Final emotional state and resolution

### LANGUAGE CONSIDERATIONS:
Pay close attention to the dialogue_language specified in the input_scenario. Your script should be written in the specified language (English or Chinese), and you should also consider:
- Language-specific idioms and expressions appropriate to the characters
- Cultural communication patterns based on the cultural_background
- How the characters' nationalities and backgrounds might influence their language use
- Any specific language-related points that might affect the dialogue flow

Important requirements:
1. Keep everything consistent with the provided metadata
2. Include all key_points from metadata naturally within the flow
3. Maintain character consistency based on their self_introductions
4. Consider the relationship_dynamic in all interactions
5. Keep the script focused and avoid unnecessary elements
6. Write the entire script in the language specified in the input_scenario's dialogue_language field

Here's an example dialogue scenario and corresponding metadata:

### Dialogue Scenario:
```json
{
    "dialogue_type": "educational_interactions",
    "temporal_context": "present_time",
    "spatial_context": "university_library",
    "cultural_background": "western_western",
    "custom_prompt": "Create a dialogue scenario for a study session in a university library.",
    "language": "English"
  }
```

### Metadata:
```json
{
  "setting": {
    "location": "university library study room",
    "time_of_day": "evening",
    "context": "exam preparation session",
    "atmosphere": "focused and slightly tense"
  },
  "role_1": {
    "name": "David Park",
    "gender": "male",
    "age": 20,
    "occupation": "undergraduate student",
    "nationality": "United States",
    "personality_traits": [
      "diligent",
      "anxious about grades",
      "helpful"
    ],
    "relationship_context": "study group partner",
    "self_introduction": "David is a second-year computer science student who takes his studies very seriously. He excels at explaining technical concepts to others but often gets anxious about exams. Despite his own stress, he genuinely enjoys helping his classmates understand difficult material. He's known for creating detailed study guides and staying late in the library."
  },
  "role_2": {
    "name": "Emma Rodriguez",
    "gender": "female",
    "age": 19,
    "occupation": "undergraduate student",
    "nationality": "United States",
    "personality_traits": [
      "optimistic",
      "quick learner",
      "slightly disorganized"
    ],
    "relationship_context": "classmate seeking help",
    "self_introduction": "Emma is a bright and enthusiastic student who grasps concepts quickly but struggles with consistent study habits. She's taking the same programming course as David and, while she understands the practical applications well, she sometimes has trouble with theoretical concepts. Her positive attitude and genuine interest in learning make her a pleasant study partner."
  },
  "conversation_context": {
    "type": "study session interaction",
    "main_topic": "preparing for upcoming programming exam",
    "relationship_dynamic": "friendly classmates",
    "emotional_tone": "supportive with underlying tension due to exam stress",
    "expected_duration": "5 minutes",
    "expected_turns": 12,
    "key_points": [
      "Discussion of difficult recursion concepts",
      "Emma asking questions about specific practice problems",
      "David sharing his study techniques",
      "Planning next study session",
      "Brief break discussion about course project"
    ]
  }
}
```

And here's how your script should be structured:

### SCENE:
University library study room, evening. The room is quiet with soft lighting. David and Emma are sitting at a table with textbooks and laptops open.

### NARRATIVE FLOW:
1. Opening (2 turns):
   - Emma arrives slightly flustered, greeting David
   - David welcomes her while organizing his study materials

2. Initial Discussion (3 turns):
   - Emma expresses anxiety about recursion concepts
   - David offers reassurance and suggests starting with basic examples
   - Emma shares specific problems she's struggling with

3. Main Study Session (4 turns):
   - David explains recursion using a practical example
   - Emma asks clarifying questions about base cases
   - David shares his method for breaking down recursive problems
   - Emma has an "aha moment" about a particular concept

4. Wrapping Up (3 turns):
   - Emma suggests planning another session
   - David agrees and mentions the upcoming project
   - Quick exchange about next meeting time

### CHARACTER BEHAVIORS:
David:
- Speaks in clear, structured sentences
- Occasionally shows signs of his own stress about exams
- Uses analogies to explain concepts
- Maintains a helpful but slightly anxious demeanor

Emma:
- Asks direct questions when confused
- Shows enthusiasm when understanding concepts
- Occasionally goes off-topic with project ideas
- Uses more casual language

### EMOTIONAL PROGRESSION:
- Start: Mild tension (Emma's stress, David's desire to help)
- Middle: Growing confidence as concepts become clearer
- End: Relief and optimism about understanding the material

### LANGUAGE CONSIDERATIONS:
- Both characters are native English speakers from the United States
- David uses more precise technical terminology due to his diligent nature
- Emma incorporates more casual expressions and filler words
- Their Western cultural background influences their direct communication style
- Both maintain polite but informal language typical of American university students

Generate a complete script following this format. Base all elements strictly on the provided metadata JSON.
"""

USER_PROMPT_TEMPLATE = """
## Dialogue Scenario
{scenario}

## Metadata
```json
{metadata}
```
"""


class ScriptGenerator:
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
                    ),
                },
            ]
            messages.append(message)
        return messages

    def _fill_back(self, outputs, dialogues):
        remaining_dialogues = []
        for i in outputs["success_indices"]:
            script = outputs["responses"][i]
            dialogues[i].script = script
            remaining_dialogues.append(dialogues[i])
        return remaining_dialogues

    def generate(self, dialogues: List[Dialogue], gen_params={}):
        prompts = self._construct_prompt(dialogues)
        logger.info(f"Generating {len(dialogues)} scripts...")
        outputs = self.llm.generate(prompts, json_model=None, **gen_params)
        remaining_dialogues = self._fill_back(outputs, dialogues)
        logger.info(f"Received {len(remaining_dialogues)} scripts from LLM.")
        return remaining_dialogues
