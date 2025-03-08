from utils.llm import LLM
from pydantic import BaseModel, Field
from typing import Optional, List
import logging
from data.dialogue import DialogueScenario, Dialogue

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_TEMPLATE = """
# Dialogue Scene Seed Generator

You are a specialized dialogue scenario seed creator. Your task is to generate diverse, high-level scenario parameters that will serve as inputs for a more detailed dialogue metadata generator.

## Your Objective

Create varied and realistic scenario seeds that can be expanded into rich conversation contexts. These seeds should provide broad directional guidance while leaving room for creative development in the next stage of generation.

## Output Format

Generate a JSON object with the following structure:
```json
{
  "dialogue_type": "",
  "temporal_context": "",
  "spatial_context": "",
  "cultural_background": ""
}
```

## User Input Adaptation

The user may provide specific requirements or contextual information in their prompt. When this happens:

1. Carefully analyze any user-provided context, themes, or constraints
2. Prioritize user specifications over general diversity guidelines
3. Create scenario seeds that directly align with the user's expressed needs
4. Maintain coherence between all parameters while incorporating user requirements
5. If the user requests specific dialogue types, contexts, or cultural backgrounds, ensure your output reflects these exactly

## Generation Guidelines

1. Create realistic combinations that could plausibly exist in the real world
2. Ensure diversity across all parameters when generating multiple seeds
3. Allow for interesting juxtapositions that might lead to compelling interactions
4. Keep values concise but descriptive enough to guide the next generation phase
5. Avoid repetitive patterns or overly similar combinations when creating multiple seeds

Remember, these seeds are the starting point for a more detailed dialogue scenario generation process. They should provide clear direction while leaving room for creative expansion in subsequent stages.
"""

USER_PROMPT_TEMPLATE = """
## Dialogue ID:
{dialogue_id}

## Dialogue Language: 
{dialogue_language}

## Custom Prompt:
{custom_prompt}
"""


class ScenarioGenerator:
    def __init__(self, llm):
        self.llm = llm

    def _construct_prompt(self, num_scenarios, dialogue_languages, custom_prompts):
        if custom_prompts is not None and dialogue_languages is not None:
            assert len(custom_prompts) == len(dialogue_languages) == num_scenarios

        created_prompts = []
        for i in range(num_scenarios):
            message = [
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
                {
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(
                        dialogue_id=i,
                        dialogue_language=(
                            dialogue_languages[i]
                            if dialogue_languages is not None
                            else self.default_language
                        ),
                        custom_prompt=(
                            custom_prompts[i] if custom_prompts is not None else "N/A"
                        ),
                    ),
                },
            ]
            created_prompts.append(message)
        return created_prompts

    def generate_scenario_with_user_input(
        self,
        num_scenarios,
        dialogue_languages: List[str] = None,
        custom_prompts: List[str] = None,
        gen_params={},
    ):
        """
        Generate a diverse dialogue scenario seed based on high-level parameters.
        """
        prompts = self._construct_prompt(
            num_scenarios, dialogue_languages, custom_prompts
        )

        logger.info(f"Generating {num_scenarios} scenarios...")

        scenarios = self.llm.generate(prompts, DialogueScenario, **gen_params)[
            "responses"
        ]

        for i in range(len(scenarios)):
            scenarios[i]["custom_prompt"] = custom_prompts[i]
            scenarios[i]["dialogue_language"] = dialogue_languages[i]

        # Deduplicate exactly same scenarios, but we need to make sure the order is kept
        scenarios = list({str(scenario): scenario for scenario in scenarios}.values())

        logger.info(f"Received {len(scenarios)} scenarios from LLM.")

        # Pack scenarios into a predefined dialogue object

        dialogues = []
        for scenario in scenarios:
            dialogues.append(Dialogue(scenario=scenario))
        return dialogues

    def generate_scenario_batched_auto(
        self, num_scenarios, dialogue_language=None, gen_params={}
    ):
        """
        Generate a diverse dialogue scenario seed based on high-level parameters.
        """
        prompts = self._construct_prompt(
            num_scenarios, [dialogue_language] * num_scenarios, None
        )

        logger.info(f"Generating {num_scenarios} scenarios...")

        scenarios = self.llm.generate(prompts, DialogueScenario, **gen_params)[
            "responses"
        ]

        for i in range(len(scenarios)):
            scenarios[i]["dialogue_language"] = dialogue_language
            scenarios[i]["custom_prompt"] = ""

        # Deduplicate exactly same scenarios
        scenarios = list({str(scenario): scenario for scenario in scenarios}.values())

        logger.info(f"Generated {len(scenarios)} scenarios.")

        dialogues = []
        for scenario in scenarios:
            dialogues.append(Dialogue(scenario=scenario))
        return dialogues
