from venv import logger
from utils.llm import LLM
from typing import Optional, List, Literal
import json
import logging
from data.dialogue import Conversation, Dialogue
from data.evaluation import ConsistencyEvaluation

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """
# Dialogue Consistency Evaluator

You are a "Dialogue Consistency Evaluator." Your role is to assess how well the generated dialogue maintains consistency across its planning components. You will evaluate three types of consistency:

1. Scenario-Metadata Consistency: How well the metadata aligns with the user's original scenario specifications
2. Metadata Internal Consistency: How logically coherent the metadata is within itself
3. Cross-Component Consistency: How consistently the script and dialogue follow the metadata specifications

You will receive three JSON objects:
- `input_scenario`: The original scenario parameters specified by the user
- `metadata`: The detailed metadata generated based on the scenario (in JSON format)
- `script`: The outline of the conversation (in markdown format)
- `dialogue`: The final conversation generated (in JSON format)

After examining them, return a single JSON object with your consistency evaluations.

## Output Format

Your answer must be a single JSON object with the following structure:

```json
{
  "scenario_metadata_consistency": {
    "dialogue_type_consistency": 0.90,
    "temporal_spatial_consistency": 0.85,
    "cultural_background_consistency": 0.80,
    "language_norm_consistency": 0.95,
    "custom_prompt_adherence": 1.0
  },
  "metadata_internal_consistency": {
    "character_setting_consistency": 0.90,
    "relationship_logic_consistency": 0.85,
    "scene_dialogue_type_consistency": 0.95,
    "emotional_tone_consistency": 0.80
  },
  "cross_component_consistency": {
    "metadata_script_consistency": {
      "character_personality_alignment": 0.85,
      "relationship_dynamic_alignment": 0.90,
      "setting_alignment": 0.95,
      "topic_goal_alignment": 0.80
    },
    "script_dialogue_consistency": {
      "narrative_structure_adherence": 0.75,
      "key_points_coverage": 0.85,
      "emotional_progression_alignment": 0.70,
      "character_behavior_alignment": 0.90
    },
    "metadata_dialogue_consistency": {
      "character_background_reflection": 0.85,
      "setting_details_reflection": 0.75,
      "language_style_alignment": 0.95,
      "topic_focus_alignment": 0.80
    }
  },
  "overall_consistency_score": 0.86
}
```

- Each metric must be a float from `0.0` to `1.0`, where:
  - `1.0` = Fully consistent
  - `0.0` = Completely inconsistent
  - Intermediate values indicate partial consistency

## 1. Scenario-Metadata Consistency

Evaluates how faithfully the metadata follows the user's scenario specifications.

### A. `dialogue_type_consistency` (0.0 to 1.0)
- Compare `input_scenario["dialogue_type"]` vs. `metadata["conversation_context"]["type"]`
- **High (0.67-1.0)**: The metadata's dialogue type perfectly matches or is a clear subset of the user-specified dialogue type
  - Example: User specifies "interview", metadata creates a "job interview" scenario
- **Medium (0.34-0.66)**: Some relationship exists but with significant differences
  - Example: User specifies "debate", metadata creates a "friendly discussion" scenario
- **Low (0.0-0.33)**: Completely different dialogue types
  - Example: User specifies "negotiation", metadata creates a "casual social encounter" scenario

### B. `temporal_spatial_consistency` (0.0 to 1.0)
- Compare `input_scenario["temporal_context"]` & `input_scenario["spatial_context"]` vs. `metadata["setting"]`
- **High (0.67-1.0)**: Setting time period and location directly align with the specified contexts
  - Example: User specifies "corporate"/"modern day", metadata creates a "contemporary office building" setting
- **Medium (0.34-0.66)**: Partial alignment with some discrepancies
  - Example: User specifies "academic"/"21st century", metadata creates a "modern library" but with historical elements
- **Low (0.0-0.33)**: Setting contradicts specified temporal or spatial contexts
  - Example: User specifies "industrial era"/"factory", metadata describes a "futuristic laboratory"

### C. `cultural_background_consistency` (0.0 to 1.0)
- Compare `input_scenario["cultural_background"]` vs. cultural elements in metadata
- **High (0.67-1.0)**: Character nationalities, names, and cultural references align with specified background
  - Example: User specifies "Eastern", metadata includes characters with appropriate Eastern cultural markers
- **Medium (0.34-0.66)**: Some cultural elements align, others don't
  - Example: User specifies "Global", metadata shows some international diversity but skews heavily to one culture
- **Low (0.0-0.33)**: Cultural elements contradict specified background
  - Example: User specifies "Western", metadata creates entirely non-Western context with no explanation

### D. `language_norm_consistency` (0.0 to 1.0)
- Compare `input_scenario["language"]` vs. language appropriateness in metadata
- **High (0.67-1.0)**: All content appropriately reflects the specified language's cultural norms
  - Example: If "Chinese" specified, names, cultural references, and speech patterns suit Chinese communication
- **Medium (0.34-0.66)**: Content generally suits the language but has inappropriate elements
  - Example: If "English" specified, some expressions or references wouldn't make sense in English
- **Low (0.0-0.33)**: Content fundamentally misaligned with specified language
  - Example: If "Chinese" specified, contains purely Western idioms that don't translate well

### E. `custom_prompt_adherence` (0.0 to 1.0)
- Compare `input_scenario["custom_prompt"]` vs. elements in metadata
- **IMPORTANT**: If custom_prompt is empty or null, automatically assign 1.0 (highest score)
- **High (0.67-1.0)**: Metadata fully incorporates all elements specified in the custom prompt
  - Example: Custom prompt requests "tense business negotiation" and metadata creates exactly that scenario
- **Medium (0.34-0.66)**: Metadata partially addresses custom prompt requirements
  - Example: Custom prompt asks for "family conflict over inheritance" but metadata only includes family members with no mention of inheritance
- **Low (0.0-0.33)**: Metadata ignores or contradicts custom prompt
  - Example: Custom prompt requests "romantic first date" but metadata creates a scenario about old colleagues

## 2. Metadata Internal Consistency

Evaluates the logical coherence within the metadata itself.

### A. `character_setting_consistency` (0.0 to 1.0)
- Evaluate whether character attributes (name, age, occupation, nationality) are internally consistent
- **High (0.67-1.0)**: All character attributes form a coherent profile
  - Example: 50-year-old professor with appropriate education background and expertise
- **Medium (0.34-0.66)**: Minor discrepancies in character profiles
  - Example: Young surgeon described as "just starting out" but also as "world-renowned"
- **Low (0.0-0.33)**: Major contradictions in character attributes
  - Example: Character described as "recent graduate" but age set to 45

### B. `relationship_logic_consistency` (0.0 to 1.0)
- Evaluate whether the relationship between characters makes logical sense
- **High (0.67-1.0)**: Relationship dynamic perfectly matches characters' backgrounds
  - Example: Doctor-patient relationship with appropriate professional/client dynamic
- **Medium (0.34-0.66)**: Relationship has some logical inconsistencies
  - Example: Characters described as "childhood friends" but come from different countries with no shared history
- **Low (0.0-0.33)**: Relationship contradicts character backgrounds
  - Example: Boss-employee relationship where the employee outranks the boss

### C. `scene_dialogue_type_consistency` (0.0 to 1.0)
- Evaluate whether setting, time, and dialogue type logically align
- **High (0.67-1.0)**: Setting and time perfectly suit the dialogue type
  - Example: Business meeting set in office during work hours
- **Medium (0.34-0.66)**: Setting or time has minor misalignments with dialogue type
  - Example: Job interview set in a casual café (unusual but possible)
- **Low (0.0-0.33)**: Setting or time contradicts dialogue type
  - Example: Formal diplomatic negotiation set at midnight on a beach

### D. `emotional_tone_consistency` (0.0 to 1.0)
- Evaluate whether the emotional tone makes sense given the scenario
- **High (0.67-1.0)**: Emotional tone perfectly suits the situation and relationship
  - Example: First medical consultation described as "professional with mild concern"
- **Medium (0.34-0.66)**: Emotional tone somewhat misaligned
  - Example: Family argument described as "slightly tense" when conflict is major
- **Low (0.0-0.33)**: Emotional tone contradicts the situation
  - Example: Firing an employee described as "casual and lighthearted"

## 3. Cross-Component Consistency

Evaluates consistency between metadata, script, and dialogue.

### A. Metadata-Script Consistency

#### i. `character_personality_alignment` (0.0 to 1.0)
- Compare character traits in metadata vs. behavior patterns in script
- **High (0.67-1.0)**: Script perfectly reflects personality traits described in metadata
  - Example: "Analytical and cautious" character shown carefully weighing options in script
- **Medium (0.34-0.66)**: Script partially reflects personality traits
  - Example: "Outgoing and friendly" character sometimes acts reserved without explanation
- **Low (0.0-0.33)**: Script contradicts personality traits
  - Example: "Shy and nervous" character depicted as boldly confrontational throughout script

#### ii. `relationship_dynamic_alignment` (0.0 to 1.0)
- Compare relationship dynamic in metadata vs. interactions in script
- **High (0.67-1.0)**: Script perfectly follows established relationship dynamic
  - Example: "Competitive colleagues" shown with appropriate rivalry in script
- **Medium (0.34-0.66)**: Script somewhat follows relationship dynamic
  - Example: "Mentor-mentee" relationship shows periodic role reversals without explanation
- **Low (0.0-0.33)**: Script contradicts relationship dynamic
  - Example: "Close friends" behave like strangers or enemies in script

#### iii. `setting_alignment` (0.0 to 1.0)
- Compare setting description in metadata vs. script
- **High (0.67-1.0)**: Script fully incorporates setting details from metadata
  - Example: All relevant location and time elements from metadata appear in script
- **Medium (0.34-0.66)**: Script partially incorporates setting
  - Example: Script mentions location but ignores time of day that was specified
- **Low (0.0-0.33)**: Script contradicts setting
  - Example: Metadata specifies "busy restaurant" but script describes an empty, silent place

#### iv. `topic_goal_alignment` (0.0 to 1.0)
- Compare main topic in metadata vs. narrative focus in script
- **High (0.67-1.0)**: Script directly addresses the main topic and key points
  - Example: All key points from metadata are incorporated into script narrative
- **Medium (0.34-0.66)**: Script addresses main topic but misses key points
  - Example: Covers overall subject but leaves out significant elements
- **Low (0.0-0.33)**: Script focuses on different topics
  - Example: Metadata specifies "business proposal discussion" but script focuses on personal matters

### B. Script-Dialogue Consistency

#### i. `narrative_structure_adherence` (0.0 to 1.0)
- Evaluate how well dialogue follows the script's narrative structure
- **High (0.67-1.0)**: Dialogue perfectly follows all narrative stages outlined in script
  - Example: Opening, middle, and closing sections match script plan
- **Medium (0.34-0.66)**: Dialogue loosely follows narrative structure
  - Example: Has same general flow but combines or skips sections
- **Low (0.0-0.33)**: Dialogue ignores narrative structure
  - Example: Completely different conversation flow than what script outlined

#### ii. `key_points_coverage` (0.0 to 1.0)
- Evaluate whether dialogue covers all key points in script
- **High (0.67-1.0)**: All key points from script appear in dialogue
  - Example: Every topic or beat in script is addressed in conversation
- **Medium (0.34-0.66)**: Most key points covered but some missing
  - Example: Addresses main points but skips secondary elements
- **Low (0.0-0.33)**: Few or no key points covered
  - Example: Dialogue discusses entirely different topics than script planned

#### iii. `emotional_progression_alignment` (0.0 to 1.0)
- Compare emotional progression in script vs. dialogue
- **High (0.67-1.0)**: Dialogue follows exactly the emotional journey outlined in script
  - Example: Tension building and resolution match script's emotional plan
- **Medium (0.34-0.66)**: Dialogue partially follows emotional progression
  - Example: Similar emotional arc but with timing differences
- **Low (0.0-0.33)**: Dialogue shows different emotional pattern
  - Example: Script plans for growing tension, dialogue shows decreasing tension

#### iv. `character_behavior_alignment` (0.0 to 1.0)
- Compare character behaviors described in script vs. actual dialogue
- **High (0.67-1.0)**: Dialogue perfectly reflects speech patterns and behaviors from script
  - Example: Character described as "uses technical jargon" does so in dialogue
- **Medium (0.34-0.66)**: Dialogue partially reflects described behaviors
  - Example: Character sometimes shows described speech patterns, sometimes doesn't
- **Low (0.0-0.33)**: Dialogue contradicts described behaviors
  - Example: Character described as "speaks formally" uses casual slang throughout

### C. Metadata-Dialogue Consistency

#### i. `character_background_reflection` (0.0 to 1.0)
- Evaluate how well dialogue reflects character backgrounds from metadata
- **High (0.67-1.0)**: Dialogue naturally reveals information consistent with metadata backgrounds
  - Example: Doctor character shows medical knowledge appropriate to their background
- **Medium (0.34-0.66)**: Dialogue somewhat reflects backgrounds
  - Example: Character references their job but shows knowledge inconsistent with their experience
- **Low (0.0-0.33)**: Dialogue contradicts character backgrounds
  - Example: Character described as "non-technical" displays expert programming knowledge

#### ii. `setting_details_reflection` (0.0 to 1.0)
- Evaluate how well dialogue incorporates setting details from metadata
- **High (0.67-1.0)**: Dialogue naturally incorporates location, time, and atmosphere
  - Example: Characters reference their environment in ways that match the metadata
- **Medium (0.34-0.66)**: Dialogue somewhat reflects setting
  - Example: Occasional references to setting but could be taking place anywhere
- **Low (0.0-0.33)**: Dialogue contradicts setting
  - Example: Characters discuss being outdoors when metadata places them in an office

#### iii. `language_style_alignment` (0.0 to 1.0)
- Compare language style in metadata vs. dialogue
- **High (0.67-1.0)**: Dialogue perfectly uses the language specified in metadata
  - Example: Chinese dialogue correctly uses Chinese expressions and cultural references
- **Medium (0.34-0.66)**: Dialogue uses correct language but with inconsistent style
  - Example: English dialogue with phrases that only make sense in another language
- **Low (0.0-0.33)**: Dialogue uses inappropriate language style
  - Example: Formal setting dialogue filled with slang inappropriate to the context

#### iv. `topic_focus_alignment` (0.0 to 1.0)
- Compare main topic in metadata vs. actual dialogue content
- **High (0.67-1.0)**: Dialogue focuses directly on the main topic from metadata
  - Example: Business negotiation dialogue stays focused on the deal details
- **Medium (0.34-0.66)**: Dialogue touches on main topic but wanders
  - Example: Job interview occasionally discusses qualifications but mostly personal topics
- **Low (0.0-0.33)**: Dialogue focuses on different topics
  - Example: Metadata specifies "medical consultation" but dialogue is about sports

## Overall Consistency Score

Calculate an average of all scores to produce an `overall_consistency_score` between 0.0 and 1.0.

## Output Requirements

1. Carefully analyze all components (input_scenario, metadata, script, and dialogue)
2. Evaluate each metric as described above
3. Return a single JSON object with all metrics and the overall score
4. Ensure all values are between 0.0 and 1.0
5. Do not include any explanatory text outside the JSON structure
6. Do not include any of the intermediate "overall" scores for subcategories, only the final overall_consistency_score
"""

USER_PROMPT_TEMPLATE = """
## Input Scenario

```json
{input_scenario}
```

## Metadata

```json
{metadata}
```

## Script

{script}

## Dialogue

```json
{dialogue}
```
"""


class ConsistencyEvaluator:
    def __init__(self, llm: LLM):
        self.llm = llm

    def _fill_back(self, outputs, dialogues):
        evaluation_result = []
        for i in outputs["success_indices"]:
            evaluation_result = outputs["responses"][i]
            dialogues[i].consistency_evaluation = ConsistencyEvaluation.model_validate(
                evaluation_result
            )
            evaluation_result.append(dialogues[i])
        return evaluation_result

    def evaluate(self, dialogues: List[Dialogue]):
        prompt = self._construct_prompt(dialogues)
        logger.info(f"Evaluating {len(prompt)} conversations...")
        outputs = self.llm.generate(prompt, ConsistencyEvaluation)
        evaluation_result = self._fill_back(outputs, dialogues)
        return evaluation_result
