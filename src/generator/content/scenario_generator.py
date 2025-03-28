from utils.base_classes import SDFModule
from utils.llm import LLM
from pydantic import BaseModel, Field
from typing import Optional, List
import logging
from data_classes.dialogue import DialogueScenario, Dialogue

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

SYSTEM_PROMPT_TEMPLATE_CN = """
# 对话场景种子生成器

您是一位专业的对话场景种子创建者。您的任务是生成多样化、高层次的场景参数，这些参数将作为更详细对话元数据生成器的输入。

## 您的目标

创建多样化且真实的场景种子，这些种子可以扩展为丰富的对话背景。这些种子应提供广泛的方向性指导，同时为下一阶段的创意发展留出空间。

## 输出格式

生成具有以下结构的JSON对象：
```json
{
  "dialogue_type": "",
  "temporal_context": "",
  "spatial_context": "",
  "cultural_background": ""
}
```

## 用户输入适配

用户可能在其提示中提供特定要求或上下文信息。当这种情况发生时：
1. 仔细分析任何用户提供的上下文、主题或约束
2. 优先考虑用户规格，而不是一般的多样性指南
3. 创建直接与用户表达的需求相一致的场景种子
4. 在包含用户要求的同时，保持所有参数之间的一致性
5. 如果用户请求特定的对话类型、上下文或文化背景，请确保您的输出准确反映这些内容

## 生成指南
1. 创建现实的组合，这些组合在现实世界中可能存在
2. 在生成多个种子时确保所有参数的多样性
3. 允许有趣的并置，这可能会带来引人入胜的互动
4. 保持值简洁，但足够描述性，以指导下一阶段的生成
5. 在创建多个种子时，避免重复的模式或过于相似的组合
6. 确保生成的场景种子内容为中文，但是JSON的key仍然为英文

记住，这些种子是更详细的对话场景生成过程的起点。它们应提供明确的方向，同时为后续阶段的创意扩展留出空间。
"""

USER_PROMPT_TEMPLATE_CN = """
## 对话ID:
{dialogue_id}
## 对话语言:
{dialogue_language}
## 自定义提示:
{custom_prompt}
"""

@SDFModule.set_role("generator")
class ScenarioGenerator(SDFModule):
    def __init__(self, args, llm: LLM=None):
        self.llm = llm
        self.default_language = args.default_language

    def _construct_prompt(self, num_scenarios, dialogue_languages, custom_prompts):
        if custom_prompts is not None and dialogue_languages is not None:
            assert len(custom_prompts) == len(dialogue_languages) == num_scenarios

        created_prompts = []
        for i in range(num_scenarios):
            dialogue_langue = dialogue_languages[i] if dialogue_languages is not None else self.default_language
            # Chinese or English
            SPROMPT = SYSTEM_PROMPT_TEMPLATE_CN if dialogue_langue == "Chinese" else SYSTEM_PROMPT_TEMPLATE
            UPROMPT = USER_PROMPT_TEMPLATE_CN if dialogue_langue == "Chinese" else USER_PROMPT_TEMPLATE

            message = [
                {"role": "system", "content": SPROMPT},
                {
                    "role": "user",
                    "content": UPROMPT.format(
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

    def generate(
        self,
        num_dialogues,
        dialogue_languages: List[str] = None,
        custom_prompts: List[str] = None,
        gen_params={"temperature": 1.7},
    ):
        """
        Generate a diverse dialogue scenario seed based on high-level parameters.
        """
        prompts = self._construct_prompt(
            num_dialogues, dialogue_languages, custom_prompts
        )

        logger.info(f"Generating {num_dialogues} scenarios...")

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
            dialogues.append(Dialogue(scenario=DialogueScenario.model_validate(scenario)))
        return dialogues
