from utils.llm import LLM
from pydantic import BaseModel, Field
from typing import Optional, List
import json
import logging
from data.dialogue import DialogueScenario, Dialogue, Metadata

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """
You are a conversation metadata designer. Your task is to generate metadata for a two-person conversation that is realistic and engaging.

## Input Information

You will be provided with a dialogue scenario in JSON format containing fields such as:
- dialogue_type (e.g., interview, debate, negotiation)
- temporal_context (time period or era)
- spatial_context (general environment)
- cultural_background (cultural influence)
- language (either "English" or "Chinese")
- custom_prompt (optional user-specific requirements)

You must use this information as the foundation for your metadata creation, ensuring all elements of your design align with these parameters. Pay special attention to the language field - if "Chinese" is specified, all names, occupations, and content should be appropriate for Chinese speakers and cultural context.

If the user provides a custom_prompt, you must carefully incorporate these specific requirements into your metadata generation. The custom_prompt takes precedence over general guidelines and may contain special instructions about character relationships, conversation dynamics, specific topics, or emotional tones that must be reflected in your output.

## Output Requirements

The metadata should be in JSON format that follows these requirements:

1. Create realistic, specific scenarios that could happen in daily life
2. Design distinct, well-rounded characters with clear personalities
3. Establish clear relationship dynamics and conversation purposes
4. Keep all details consistent with the provided goal and location

The metadata must include these components:
- setting: location, time of day, and contextual background
- role_1 and role_2: two speakers with detailed characteristics
- conversation_context: structure and goals of the interaction (incorporate the provided goal)

## Setting Requirements

For the setting field, include:
- location: Specific physical or virtual location (more detailed than the spatial_context input)
- time_of_day: Morning, afternoon, evening, or night
- context: Detailed description of the situation and circumstances (30-50 words)
- atmosphere: The mood or feeling of the environment

## Character Requirements

For each role, include:
- name: realistic full name
- gender: "male", "female"
- age: between 10 and 60
- occupation: specific job or role
- nationality: the nationality of the character
- personality_traits: list of at least 1 defining characteristic
- relationship_context: their role in current situation
- self_introduction: detailed paragraph (50+ characters) describing their personality and background

## Conversation Context Requirements

For conversation_context, specify:
- type: category of interaction
- main_topic: primary discussion subject
- relationship_dynamic: how the speakers relate
- emotional_tone: overall mood
- expected_duration: approximate time (< 5 mins)
- expected_turns: number of exchanges (8-12)
- key_points: list of main events/topics to cover

## Important Requirements

1. Output ONLY valid JSON, no additional text or explanations
2. Make scenarios specific rather than generic
3. Create diverse but realistic character combinations
4. Ensure all scenario elements are logically connected
5. Write self_introductions in natural, flowing language
6. Keep expected_turns reasonable for the scenario (typically 8-12)

Here's an example of the expected JSON format:
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

Generate a complete, valid JSON structure following these requirements. The JSON should tell a coherent story about how these two people come to interact and what they'll discuss.
"""

USER_PROMPT_TEMPLATE = """
## Dialogue Scenario
```json
{scenario}
```
"""


class MetadataGenerator:
    def __init__(self, llm: LLM):
        self.llm = llm

    def _construct_prompt(self, dialogues):

        created_prompts = []
        for i, dialogue in enumerate(dialogues):
            message = [
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
                {
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(
                        scenario=dialogue.scenario.to_json(pretty=True),
                    ),
                },
            ]
            created_prompts.append(message)
        return created_prompts

    def _fill_back(self, outputs, dialogues):
        remaining_dialogues = []
        for i in outputs["success_indices"]:
            metadata = outputs["responses"][i]
            dialogues[i].metadata = Metadata.model_validate(metadata)
            remaining_dialogues.append(dialogues[i])
        return remaining_dialogues

    def generate_metadata(self, dialogues: List[Dialogue], gen_params={}):
        prompts = self._construct_prompt(dialogues)
        logger.info(f"Generating {len(dialogues)} metadata...")
        outputs = self.llm.generate(prompts, Metadata, **gen_params)
        remaining_dialogues = self._fill_back(outputs, dialogues)
        logger.info(f"Generated {len(remaining_dialogues)} metadata.")
        return remaining_dialogues
