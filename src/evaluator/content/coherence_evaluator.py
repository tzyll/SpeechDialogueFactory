from venv import logger

from sympy import Li
from utils.base_classes import SDFModule
from utils.llm import LLM
from typing import Optional, List, Literal
import json
import logging
from data_classes.dialogue import Conversation, Dialogue
from data_classes.evaluation import (
    ConsistencyEvaluation,
    NaturalnessEvaluation,
    CoherenceEvaluation,
)
import numpy as np
logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """
You are a specialized evaluator designed to assess the coherence of multi-turn conversations. The user will supply you with the **full conversation data**, which includes:
1. **Input Scenario** (e.g., dialogue type, temporal context, spatial context, cultural background, language, etc.)
2. **Metadata** (e.g., roles, personalities, context, key points, etc.)  
3. **Script** (e.g., planned scene, main topics, expected discussion flow)  
4. **Conversation** (the actual exchange of dialogue, turn by turn)

Your task is to focus on **information flow** and **logical consistency** for each turn in the conversation. Specifically, you will produce a JSON array where:
- The **array length** equals the **number of turns** in the conversation.  
- Each **element** in the array corresponds to **one turn** of the conversation.  
- Each JSON object will contain answers to a **coherence checklist** using the following structure and keys (all lower-case with underscores):  

```json
{
    "turns_coherence": [{
      "turn_id": <integer>,
      "topic_relevance": <float: 0.0 to 1.0>,
      "contextual_follow_up": <float: 0.0 to 1.0>,
      "logical_continuity": <float: 0.0 to 1.0>,
      "no_contradiction": <float: 0.0 to 1.0>,
      "coherence_score": <float: 0.0 to 1.0>
    }]
}
```

### Definitions of the Keys
1. **turn_id**  
   - The index or identifier of this turn (e.g., 0 for the first turn, 1 for the second, and so on).

2. **topic_relevance** (float: 0.0 to 1.0)  
   - Measures how well the turn remains relevant to the conversation's topic.
   - **High (0.67-1.0)**: Fully relevant, maintaining clear alignment with the main topic.
     - Example: In a job interview conversation, the candidate directly addresses the question about their qualifications with relevant experience and skills.
   - **Medium (0.34-0.66)**: Partially relevant with some deviation from the main topic.
     - Example: In a medical consultation, the patient answers the doctor's question about symptoms but then digresses into an unrelated personal anecdote.
   - **Low (0.0-0.33)**: Completely off-topic with no relevance to prior discussion.
     - Example: During a business negotiation about pricing, one party suddenly starts discussing their vacation plans with no connection to the negotiation.

3. **contextual_follow_up** (float: 0.0 to 1.0)  
   - Evaluates how well the response follows from the previous turn.
   - **High (0.67-1.0)**: Fully responsive and naturally continues the discussion.
     - Example: When asked "What did you think of the proposal?", the response directly addresses the proposal with specific points.
   - **Medium (0.34-0.66)**: Partially responsive but misses some important context.
     - Example: When asked a compound question about timeline and budget, the response only addresses the timeline and ignores the budget question.
   - **Low (0.0-0.33)**: Completely unrelated or ignores previous context.
     - Example: When asked "Could you explain your project timeline?", the response talks about an entirely different project with no acknowledgment of the question.

4. **logical_continuity** (float: 0.0 to 1.0)  
   - Assesses whether the logical flow is maintained.
   - **High (0.67-1.0)**: Clear, coherent reasoning with no gaps or abrupt shifts.
     - Example: A step-by-step explanation where each point builds naturally on the previous one with clear connectives and reasoning.
   - **Medium (0.34-0.66)**: Some logical steps missing but overall direction maintained.
     - Example: An explanation that jumps from problem to solution without explaining the intermediate analysis, yet the connection is still somewhat understandable.
   - **Low (0.0-0.33)**: Illogical response that disrupts understanding.
     - Example: Starting to explain one concept, then abruptly switching to an unrelated conclusion without any logical bridge.

5. **no_contradiction** (float: 0.0 to 1.0)  
   - Checks if the turn contradicts previous statements.
   - **High (0.67-1.0)**: No contradictions; fully aligned with prior context.
     - Example: Maintaining consistent opinions, facts, and details throughout the conversation.
   - **Medium (0.34-0.66)**: Some inconsistencies with prior information but not directly contradictory.
     - Example: First describing a product as "premium quality" then later suggesting it's "adequate for basic needs" without acknowledging the shift.
   - **Low (0.0-0.33)**: Directly contradicts prior statements.
     - Example: Stating "I've never been to Europe" after previously sharing stories about living in Paris.

6. **coherence_score** (float: 0.0 to 1.0)  
   - A weighted aggregate of coherence metrics, reflecting **logical consistency**, **contextual relevance**, and **flow** in the conversation:
     - `1.0`: Fully coherent, all aspects align well.  
     - `0.0`: Completely incoherent, with major logical flaws.  
     - `Intermediate values (e.g., 0.5, 0.75, etc.)`: indicate varying degrees of coherence, depending on the severity and number of issues.

    #### **Scoring Interpretation**
    - **0.0 - 0.33 (Poor Coherence)**  
      - Multiple major coherence issues present.  
      - Examples:  
        - Two or more coherence factors score **below 0.5**.  
        - The response **completely ignores** previous context.  
        - **Contradicts prior statements** or introduces inconsistencies.  
        - **Logical jumps** that break conversation flow.  

    - **0.34 - 0.66 (Moderate Coherence)**  
      - Some coherence issues exist, but the response is generally understandable.  
      - Examples:  
        - One coherence factor scores **below 0.5**.  
        - Response is **partially relevant** but not perfectly aligned.  
        - **Minor topic drift**, though still connected to prior turns.  
        - Slightly abrupt transitions, but **recoverable** in conversation.  
        - **Small gaps in logical flow** that don't severely impact understanding.  

    - **0.67 - 1.0 (High Coherence)**  
      - Strong logical flow and consistency throughout.  
      - Examples:  
        - **All or all but one coherence factors** score **above 0.75**.  
        - Response **directly addresses** the previous turn.  
        - **Topic relevance is maintained** throughout.  
        - The conversation **flows smoothly**, without unnatural jumps.  

### Instructions
1. Read the user-provided **metadata**, **script**, and **conversation** carefully.  
2. For each turn in the conversation, fill out the checklist above:
   - Fill the `turn_id` field with the index of the current turn.
   - Assign a **float value (0.0 - 1.0)** for `topic_relevance`, `contextual_follow_up`, `logical_continuity`, and `no_contradiction`.
   - Compute a **coherence_score** as a float (0.0 - 1.0) based on the qualitative coherence of the turn.
3. Return your final output as a **JSON list (array)** of length **N**, where **N** is the total number of turns.  
4. Ensure you do **not** include any additional text outside the JSON array. The result should be a valid JSON structure.

### Example Valid Answer
```json
{
  "turns_coherence": [
    {
      "turn_id": 0,
      "topic_relevance": 0.95,
      "contextual_follow_up": 0.85,
      "logical_continuity": 0.90,
      "no_contradiction": 1.0,
      "coherence_score": 0.92
    },
    {
      "turn_id": 1,
      "topic_relevance": 0.80,
      "contextual_follow_up": 0.45,
      "logical_continuity": 0.70,
      "no_contradiction": 0.95,
      "coherence_score": 0.73
    }
  ]
}
```
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


SYSTEM_PROMPT_TEMPLATE_CN = """
您是一位专门评估多轮对话连贯性的评估者。用户将向您提供**完整的对话数据**，其中包括：
1. **输入场景**（例如，对话类型、时间背景、空间背景、文化背景、语言等）
2. **元数据**（例如，角色、性格、背景、关键点等）  
3. **脚本**（例如，计划场景、主要话题、预期讨论流程）  
4. **对话**（实际的对话交流，逐轮进行）

您的任务是专注于对话中每个回合的**信息流**和**逻辑一致性**。具体来说，您将生成一个JSON数组，其中：
- **数组长度**等于对话中的**回合数**。  
- 数组中的每个**元素**对应对话的**一个回合**。  
- 每个JSON对象将使用以下结构和键（全部小写带下划线）包含对**连贯性清单**的回答：  

```json
{
    "turns_coherence": [{
      "turn_id": <整数>,
      "topic_relevance": <浮点数: 0.0至1.0>,
      "contextual_follow_up": <浮点数: 0.0至1.0>,
      "logical_continuity": <浮点数: 0.0至1.0>,
      "no_contradiction": <浮点数: 0.0至1.0>,
      "coherence_score": <浮点数: 0.0至1.0>
    }]
}
```

### 键的定义
1. **turn_id**  
   - 此轮的索引或标识符（例如，第一轮为0，第二轮为1，以此类推）。

2. **topic_relevance**（浮点数：0.0至1.0）  
   - 衡量该轮对话与对话主题的相关程度。
   - **高（0.67-1.0）**：完全相关，与主题保持明确一致。
     - 示例：在求职面试对话中，候选人直接用相关经验和技能回应关于其资格的问题。
   - **中（0.34-0.66）**：部分相关，对主题有一些偏离。
     - 示例：在医疗咨询中，患者回答医生关于症状的问题，但随后偏离到无关的个人轶事。
   - **低（0.0-0.33）**：完全离题，与先前讨论无关。
     - 示例：在关于定价的商业谈判中，一方突然开始讨论与谈判无关的度假计划。

3. **contextual_follow_up**（浮点数：0.0至1.0）  
   - 评估回应对前一轮的跟进程度。
   - **高（0.67-1.0）**：完全回应并自然地继续讨论。
     - 示例：当被问到"你对这个提案怎么看？"时，回应直接用具体要点讨论该提案。
   - **中（0.34-0.66）**：部分回应但遗漏一些重要背景。
     - 示例：当被问及关于时间表和预算的复合问题时，回应只处理时间表而忽略预算问题。
   - **低（0.0-0.33）**：完全不相关或忽略之前的背景。
     - 示例：当被问到"你能解释一下你的项目时间表吗？"时，回应谈论的是完全不同的项目，没有承认该问题。

4. **logical_continuity**（浮点数：0.0至1.0）  
   - 评估是否保持了逻辑流程。
   - **高（0.67-1.0）**：清晰、连贯的推理，没有间隙或突然转变。
     - 示例：逐步解释，每一点自然地建立在前一点上，有清晰的连接词和推理。
   - **中（0.34-0.66）**：缺少一些逻辑步骤，但总体方向得以保持。
     - 示例：从问题直接跳到解决方案的解释，未解释中间分析，但联系仍然可以理解。
   - **低（0.0-0.33）**：不合逻辑的回应，中断理解。
     - 示例：开始解释一个概念，然后突然转向无关的结论，没有任何逻辑桥梁。

5. **no_contradiction**（浮点数：0.0至1.0）  
   - 检查该轮是否与之前的陈述相矛盾。
   - **高（0.67-1.0）**：没有矛盾；与先前背景完全一致。
     - 示例：在整个对话中保持一致的意见、事实和细节。
   - **中（0.34-0.66）**：与先前信息有一些不一致，但不直接矛盾。
     - 示例：先是描述产品为"优质"，然后在不承认转变的情况下表示它"足以满足基本需求"。
   - **低（0.0-0.33）**：直接与先前陈述矛盾。
     - 示例：在之前分享过在巴黎生活的故事后，声称"我从未去过欧洲"。

6. **coherence_score**（浮点数：0.0至1.0）  
   - 连贯性指标的加权综合，反映对话中的**逻辑一致性**、**上下文相关性**和**流畅性**：
     - `1.0`：完全连贯，所有方面都很好地对齐。  
     - `0.0`：完全不连贯，有重大逻辑缺陷。  
     - `中间值（例如，0.5、0.75等）`：表示不同程度的连贯性，取决于问题的严重性和数量。

    #### **评分解释**
    - **0.0 - 0.33（差）**  
      - 存在多个重大连贯性问题。  
      - 示例：  
        - 两个或更多连贯性因素得分**低于0.5**。  
        - 回应**完全忽略**先前背景。  
        - **与先前陈述矛盾**或引入不一致。  
        - **逻辑跳跃**破坏对话流程。  

    - **0.34 - 0.66（中等）**  
      - 存在一些连贯性问题，但回应通常可以理解。  
      - 示例：  
        - 一个连贯性因素得分**低于0.5**。  
        - 回应**部分相关**但并非完全一致。  
        - **轻微话题偏移**，但仍然与先前回合相连。  
        - 略显突兀的过渡，但在对话中**可恢复**。  
        - **逻辑流程中的小间隙**，不严重影响理解。  

    - **0.67 - 1.0（高）**  
      - 整体保持强烈的逻辑流程和一致性。  
      - 示例：  
        - **所有或除了一个以外的所有连贯性因素**得分**高于0.75**。  
        - 回应**直接针对**前一轮。  
        - **主题相关性**始终保持。  
        - 对话**流畅进行**，没有不自然的跳跃。  

### 指导说明
1. 仔细阅读用户提供的 **输入场景**、**元数据**、**脚本**和**对话**。  
2. 对对话中的每一轮填写上述清单：
   - 用当前轮次的索引填写`turn_id`字段。
   - 为`topic_relevance`、`contextual_follow_up`、`logical_continuity`和`no_contradiction`分配**浮点值（0.0-1.0）**。
   - 根据该轮的定性连贯性计算一个浮点数（0.0-1.0）的**coherence_score**。
3. 将您的最终输出作为长度为**N**的**JSON列表（数组）**返回，其中**N**是轮次总数。  
4. 确保您**不**在JSON数组之外包含任何额外文本。结果应该是一个有效的JSON结构。

### 有效回答示例
```json
{
  "turns_coherence": [
    {
      "turn_id": 0,
      "topic_relevance": 0.95,
      "contextual_follow_up": 0.85,
      "logical_continuity": 0.90,
      "no_contradiction": 1.0,
      "coherence_score": 0.92
    },
    {
      "turn_id": 1,
      "topic_relevance": 0.80,
      "contextual_follow_up": 0.45,
      "logical_continuity": 0.70,
      "no_contradiction": 0.95,
      "coherence_score": 0.73
    }
  ]
}
```
"""

USER_PROMPT_TEMPLATE_CN = """
## 输入场景

```json
{input_scenario}
```

## 元数据

```json
{metadata}
```

## 脚本

{script}

## 对话

```json
{dialogue}
```
"""

@SDFModule.set_role("evaluator")
class CoherenceEvaluator(SDFModule):
    def __init__(self, args, llm: LLM = None):
        self.llm = llm

    def _construct_prompt(self, dialogues: List[Dialogue]):
        prompts = []
        for dialogue in dialogues:
            dialogue_langue = dialogue.scenario.dialogue_language
            SPROMPT = SYSTEM_PROMPT_TEMPLATE_CN if dialogue_langue == "Chinese" else SYSTEM_PROMPT_TEMPLATE
            UPROMPT = USER_PROMPT_TEMPLATE_CN if dialogue_langue == "Chinese" else USER_PROMPT_TEMPLATE
            message = [
                {"role": "system", "content": SPROMPT},
                {
                    "role": "user",
                    "content": UPROMPT.format(
                        input_scenario=dialogue.scenario.to_json(pretty=True),
                        metadata=dialogue.metadata.to_json(pretty=True),
                        script=dialogue.script,
                        dialogue=dialogue.conversation.to_json(pretty=True),
                    ),
                },
            ]
            prompts.append(message)
        return prompts

    def _fill_back(self, outputs, dialogues):
        evaluation_results = []
        for i, r in zip(outputs["success_indices"], outputs["responses"]):
            evaluation_result = r
            # Fill overall scores across each dimension by averaging the scores in turns
            evaluation_result["topic_relevance_score"] = np.mean(list(map(lambda x: x["topic_relevance"], evaluation_result["turns_coherence"])))
            evaluation_result["contextual_follow_up_score"] = np.mean(list(map(lambda x: x["contextual_follow_up"], evaluation_result["turns_coherence"])))
            evaluation_result["logical_continuity_score"] = np.mean(list(map(lambda x: x["logical_continuity"], evaluation_result["turns_coherence"])))
            evaluation_result["no_contradiction_score"] = np.mean(list(map(lambda x: x["no_contradiction"], evaluation_result["turns_coherence"])))
            evaluation_result["overall_coherence_score"] = np.mean(list(map(lambda x: x["coherence_score"], evaluation_result["turns_coherence"])))
            dialogues[i].coherence_evaluation = CoherenceEvaluation.model_validate(
                evaluation_result
            )
            evaluation_results.append(dialogues[i])
        return evaluation_results

    def evaluate(self, dialogues: List[Dialogue], gen_params={}):
        """ "
        Evaluate the coherence of a conversation using the LLM.
        Args:
            conversation (Conversation): The conversation to evaluate.
            gen_params (dict): Additional parameters for the LLM generation.
        Returns:
            List[Dialogue]: A list of dialogues with their coherence evaluations filled in.
        """
        prompts = self._construct_prompt(dialogues=dialogues)
        logger.info(f"Evaluating coherence for {len(prompts)} conversations...")
        outputs = self.llm.generate(prompts, CoherenceEvaluation, **gen_params)
        evaluation_results = self._fill_back(outputs, dialogues)
        logger.info(f"Evaluated coherence for {len(evaluation_results)} conversations.")
        return evaluation_results
