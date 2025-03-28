from venv import logger
from utils.base_classes import SDFModule
from utils.llm import LLM
from typing import Optional, List, Literal
import json
import logging
from data_classes.dialogue import Conversation, Dialogue
from data_classes.evaluation import (
    ConsistencyEvaluation,
    CoherenceEvaluation,
    NaturalnessEvaluation,
)
import numpy as np

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """
You are a specialized evaluator designed to assess the **Naturalness** of multi-turn conversations. The user will provide you with **complete conversation data**, which includes:
1. **Input Scenario** (e.g., dialogue type, temporal context, spatial context, cultural background, language, etc.)
2. **Metadata** (e.g., roles, personalities, context, key points, etc.)
3. **Script** (e.g., planned scene, main topics, expected discussion flow)
4. **Conversation** (the actual exchange of dialogue, turn by turn)

Your primary goal is to rate how **natural** each turn sounds if spoken aloud—focusing on style, emotional tone, and overall realistic delivery. You will return a **JSON array** (list) where:

- The length of the array **equals the number of turns** in the conversation.  
- Each **element** in this array corresponds to exactly **one turn** of the conversation.  
- Each element is a JSON object containing the following fields (all lower-case with underscores):

```json
{
    "turns_naturalness": [ {
        "turn_id": <integer>,
        "oral_style": <float: 0.0 to 1.0>,
        "length_and_flow": <float: 0.0 to 1.0>,
        "emotion_appropriateness": <float: 0.0 to 1.0>,
        "text_emotion_consistency": <float: 0.0 to 1.0>,
        "contextual_vocabulary_style": <float: 0.0 to 1.0>,
        "naturalness_score": <float: 0.0 to 1.0>
    }]
}
```

### Definitions of the Keys

1. **turn_id**  
   - The index (e.g., 0 for the first speaker turn, 1 for the second, etc.).

2. **oral_style** (float: 0.0 to 1.0)  
   - **Compare**: The degree to which the speaker's language resembles natural spoken dialogue.
   - **High (0.67-1.0)**: Fully conversational with natural speech patterns.
     - Example: "Hey, so I was thinking about what you said earlier... you know, about the project deadline? I'm not sure we can make it by Friday."
   - **Medium (0.34-0.66)**: Somewhat conversational but with occasional unnatural phrasing.
     - Example: "I've been contemplating your earlier statement regarding the project deadline. It appears Friday may present challenges for completion."
   - **Low (0.0-0.33)**: Highly formal, robotic, or written-text style inappropriate for speech.
     - Example: "Upon careful consideration of the aforementioned deadline as previously discussed in our prior correspondence, it has become apparent that the temporal constraints are insufficient for the completion of the assigned tasks."

3. **length_and_flow** (float: 0.0 to 1.0)  
   - **Compare**: Whether the utterance length and structure are suitable for natural spoken conversation.
   - **High (0.67-1.0)**: Well-paced with natural pauses and comfortable length.
     - Example: A brief answer to a simple question, or a longer but well-structured response to a complex question with natural breaks.
   - **Medium (0.34-0.66)**: Somewhat unbalanced length or slightly awkward flow.
     - Example: An unnecessarily lengthy response to a simple question, or a response with awkward transitions between thoughts.
   - **Low (0.0-0.33)**: Severely inappropriate length or disjointed flow.
     - Example: A three-word answer to a complex question requiring explanation, or a five-minute monologue without breaks in casual conversation.

4. **emotion_appropriateness** (float: 0.0 to 1.0)  
   - **Compare**: Whether the expressed or implied emotion fits the conversational context.
   - **High (0.67-1.0)**: Emotion perfectly matches the situation and relationship dynamic.
     - Example: Expressing concern when a friend shares a problem, or excitement when receiving good news.
   - **Medium (0.34-0.66)**: Emotion somewhat misaligned with the context.
     - Example: Responding with mild amusement to a serious work problem, or being overly formal in a casual friendship setting.
   - **Low (0.0-0.33)**: Emotion completely inappropriate for the context.
     - Example: Responding cheerfully to news of a tragedy, or expressing anger during a congratulatory moment.

5. **text_emotion_consistency** (float: 0.0 to 1.0)  
   - **Compare**: The consistency between the actual wording and the stated or inferred emotion.
   - **High (0.67-1.0)**: Words and phrasing perfectly reflect the emotion being conveyed.
     - Example: Using "I'm thrilled" when the emotion is excitement, or including hesitations when nervous.
   - **Medium (0.34-0.66)**: Some disconnect between words and emotion.
     - Example: Stating "I'm happy for you" in a flat, unenthusiastic way, or using formal language during an intimate moment.
   - **Low (0.0-0.33)**: Words directly contradict the stated emotion.
     - Example: Saying "I'm not upset at all" while using aggressive language, or claiming to be "excited" while using pessimistic phrasing.

6. **contextual_vocabulary_style** (float: 0.0 to 1.0)  
   - **Compare**: Whether the speaker's vocabulary and expressions are appropriate for their background, setting, or era.
   - **High (0.67-1.0)**: Vocabulary perfectly matches the speaker's character, background, and setting.
     - Example: A doctor using appropriate medical terminology, or a teenager using contemporary slang.
   - **Medium (0.34-0.66)**: Vocabulary somewhat misaligned with speaker's background.
     - Example: A lawyer occasionally using terms incorrectly, or a character from the 1800s using some modern expressions.
   - **Low (0.0-0.33)**: Vocabulary completely inappropriate for the character or setting.
     - Example: A child using advanced technical jargon, or a historical character using current internet slang.

7. **naturalness_score** (float: 0.0 to 1.0)  
   - A weighted aggregate of the above metrics, representing the overall naturalness of the turn.  
   - **1.0** indicates fully natural, **0.0** indicates entirely unnatural, with intermediate values for varying degrees of fluency.
   
   #### Score = 0.0 - 0.33 (Poor Naturalness)
   - Multiple major naturalness issues present
   - Examples:
     - Two or more boolean checks return false
     - Speech patterns are highly robotic or artificial
     - Vocabulary completely mismatches speaker's background
     - Emotions are severely inconsistent with context
     - Turn length is extremely inappropriate (too long/short)
     - Uses overly formal/academic language in casual settings
     - Multiple instances of unnatural phrasing

   #### Score = 0.34 - 0.66 (Moderate Naturalness)
   - Minor naturalness issues exist, but speech is generally understandable
   - Examples:
     - One boolean check returns false
     - Slightly formal but still conversational
     - Minor vocabulary mismatches
     - Emotions slightly off but not jarring
     - Turn length slightly longer/shorter than ideal
     - Occasional awkward phrasing
     - Small inconsistencies in speaking style

   #### Score = 0.67 - 1.0 (High Naturalness)
   - Flows naturally and authentically throughout
   - Examples:
     - All or all but one boolean checks return true
     - Perfectly conversational tone
     - Vocabulary matches speaker's background
     - Emotions align perfectly with context
     - Appropriate turn length and pacing
     - Natural variations in speaking style
     - Authentic-sounding dialogue

### **Instructions**

1. **Analyze** the user's input.  
2. **For each turn** in the conversation:
   - Assign a **float value (0.0 - 1.0)** for `oral_style`, `length_and_flow`, `emotion_appropriateness`, `text_emotion_consistency`, and `contextual_vocabulary_style`.  
   - Compute a **naturalness_score** as a float (0.0 - 1.0) based on the qualitative coherence of the turn.  
3. **Output** your final assessment as a **JSON list**, containing **N** JSON objects (where **N** is the number of turns).  
4. Ensure you provide **no additional text** beyond the JSON array.

### **Example Valid Answer**
```json
{
  "turns_naturalness": [
    {
      "turn_id": 0,
      "oral_style": 0.95,
      "length_and_flow": 0.85,
      "emotion_appropriateness": 0.90,
      "text_emotion_consistency": 1.0,
      "contextual_vocabulary_style": 0.92,
      "naturalness_score": 0.92
    },
    {
      "turn_id": 1,
      "oral_style": 0.80,
      "length_and_flow": 0.45,
      "emotion_appropriateness": 0.70,
      "text_emotion_consistency": 0.95,
      "contextual_vocabulary_style": 0.88,
      "naturalness_score": 0.76
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
您是一位专门评估多轮对话**自然度**的评估者。用户将向您提供**完整的对话数据**，其中包括：
1. **输入场景**（例如，对话类型、时间背景、空间背景、文化背景、语言等）
2. **元数据**（例如，角色、性格、背景、关键点等）
3. **脚本**（例如，计划场景、主要话题、预期讨论流程）
4. **对话**（实际的对话交流，逐轮进行）

您的主要目标是评价每个回合如果被说出来会听起来有多**自然**——专注于风格、情感基调和整体真实感。您将返回一个**JSON数组**（列表），其中：

- 数组的长度**等于对话中的回合数**。  
- 此数组中的每个**元素**对应对话的**一个回合**。  
- 每个元素是一个包含以下字段的JSON对象（全部小写带下划线）：

```json
{
    "turns_naturalness": [ {
        "turn_id": <整数>,
        "oral_style": <浮点数: 0.0至1.0>,
        "length_and_flow": <浮点数: 0.0至1.0>,
        "emotion_appropriateness": <浮点数: 0.0至1.0>,
        "text_emotion_consistency": <浮点数: 0.0至1.0>,
        "contextual_vocabulary_style": <浮点数: 0.0至1.0>,
        "naturalness_score": <浮点数: 0.0至1.0>
    }]
}
```

### 键的定义

1. **turn_id**  
   - 索引（例如，第一个说话轮次为0，第二个为1，以此类推）。

2. **oral_style**（浮点数：0.0至1.0）  
   - **比较**：说话者的语言与自然口语对话的相似程度。
   - **高（0.67-1.0）**：完全对话化，具有自然的口语模式。
     - 示例："嘿，我在想你之前说的那个事情...你知道，关于项目截止日期的？我不确定我们能否在周五前完成。"
   - **中（0.34-0.66）**：有些对话化但偶尔有不自然的表述。
     - 示例："我一直在思考你之前关于项目截止日期的陈述。看起来周五可能对完成项目有挑战。"
   - **低（0.0-0.33）**：高度正式、机械或书面文本风格，不适合口语。
     - 示例："经过对我们先前通信中讨论的上述截止日期的仔细考虑，已经变得明显的是，时间限制对于完成指定任务是不充分的。"

3. **length_and_flow**（浮点数：0.0至1.0）  
   - **比较**：话语长度和结构是否适合自然口语对话。
   - **高（0.67-1.0）**：节奏良好，有自然停顿和舒适的长度。
     - 示例：对简单问题的简短回答，或对复杂问题的较长但结构良好的回应，有自然的间断。
   - **中（0.34-0.66）**：长度有些不平衡或流程略显尴尬。
     - 示例：对简单问题不必要的冗长回答，或回答中思路之间的过渡尴尬。
   - **低（0.0-0.33）**：长度严重不适当或断断续续的流程。
     - 示例：对需要解释的复杂问题只用三个字回答，或在随意对话中没有停顿的五分钟独白。

4. **emotion_appropriateness**（浮点数：0.0至1.0）  
   - **比较**：表达或暗示的情感是否符合对话背景。
   - **高（0.67-1.0）**：情感完全匹配情境和关系动态。
     - 示例：当朋友分享问题时表达关切，或收到好消息时表达兴奋。
   - **中（0.34-0.66）**：情感与背景有些不一致。
     - 示例：对严肃的工作问题表示轻微的娱乐，或在随意的友谊场合过于正式。
   - **低（0.0-0.33）**：情感完全不适合背景。
     - 示例：对悲剧消息愉快地回应，或在祝贺时刻表达愤怒。

5. **text_emotion_consistency**（浮点数：0.0至1.0）  
   - **比较**：实际用词与所述或推断的情感之间的一致性。
   - **高（0.67-1.0）**：词语和措辞完美反映所传达的情感。
     - 示例：当情感是兴奋时使用"我很激动"，或在紧张时包含犹豫。
   - **中（0.34-0.66）**：词语和情感之间有些脱节。
     - 示例：以平淡、不热情的方式说"我为你高兴"，或在亲密时刻使用正式语言。
   - **低（0.0-0.33）**：词语直接与所述情感矛盾。
     - 示例：说"我一点都不生气"而使用攻击性语言，或声称"很兴奋"却使用悲观措辞。

6. **contextual_vocabulary_style**（浮点数：0.0至1.0）  
   - **比较**：说话者的词汇和表达是否适合其背景、环境或时代。
   - **高（0.67-1.0）**：词汇完全匹配说话者的性格、背景和环境。
     - 示例：医生使用适当的医学术语，或青少年使用现代俚语。
   - **中（0.34-0.66）**：词汇与说话者的背景有些不一致。
     - 示例：律师偶尔错误使用术语，或19世纪的角色使用一些现代表达。
   - **低（0.0-0.33）**：词汇对角色或环境完全不适当。
     - 示例：儿童使用高级技术术语，或历史角色使用当前网络俚语。

7. **naturalness_score**（浮点数：0.0至1.0）  
   - 上述指标的加权综合，代表回合的整体自然度。  
   - **1.0**表示完全自然，**0.0**表示完全不自然，中间值表示不同程度的流畅度。
   
   #### 分数 = 0.0 - 0.33（差）
   - 存在多个重大自然度问题
   - 示例：
     - 言语模式高度机械或人工
     - 词汇与说话者背景完全不匹配
     - 情感与背景严重不一致
     - 回合长度极不适当（太长/太短）
     - 在随意场合使用过于正式/学术的语言
     - 多处不自然的措辞

   #### 分数 = 0.34 - 0.66（中等）
   - 存在轻微的自然度问题，但言语通常可以理解
   - 示例：
     - 略微正式但仍然对话化
     - 轻微的词汇不匹配
     - 情感略有偏差但不刺耳
     - 回合长度略长/短于理想
     - 偶尔有尴尬的措辞
     - 说话风格中的小不一致

   #### 分数 = 0.67 - 1.0（高）
   - 整体流畅自然真实
   - 示例：
     - 完美的对话语调
     - 词汇与说话者背景匹配
     - 情感完全符合背景
     - 适当的回合长度和节奏
     - 说话风格的自然变化
     - 听起来真实的对话

### **指导说明**

1. **分析**用户的输入。  
2. **对于对话中的每个回合**：
   - 为`oral_style`、`length_and_flow`、`emotion_appropriateness`、`text_emotion_consistency`和`contextual_vocabulary_style`分配**浮点值（0.0-1.0）**。  
   - 根据回合的定性连贯性计算一个浮点数（0.0-1.0）的**naturalness_score**。  
3. **输出**您的最终评估作为一个**JSON列表**，包含**N**个JSON对象（其中**N**是回合数）。  
4. 确保您在JSON数组之外**不提供额外文本**。

### **有效回答示例**
```json
{
  "turns_naturalness": [
    {
      "turn_id": 0,
      "oral_style": 0.95,
      "length_and_flow": 0.85,
      "emotion_appropriateness": 0.90,
      "text_emotion_consistency": 1.0,
      "contextual_vocabulary_style": 0.92,
      "naturalness_score": 0.92
    },
    {
      "turn_id": 1,
      "oral_style": 0.80,
      "length_and_flow": 0.45,
      "emotion_appropriateness": 0.70,
      "text_emotion_consistency": 0.95,
      "contextual_vocabulary_style": 0.88,
      "naturalness_score": 0.76
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
class NaturalnessEvaluator(SDFModule):
    def __init__(self, args, llm: LLM=None):
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
            evaluation_result["oral_style_score"] = np.mean(list(map(lambda x: x["oral_style"], evaluation_result["turns_naturalness"])))
            evaluation_result["length_and_flow_score"] = np.mean(list(map(lambda x: x["length_and_flow"], evaluation_result["turns_naturalness"])))
            evaluation_result["emotion_appropriateness_score"] = np.mean(list(map(lambda x: x["emotion_appropriateness"], evaluation_result["turns_naturalness"])))
            evaluation_result["text_emotion_consistency_score"] = np.mean(list(map(lambda x: x["text_emotion_consistency"], evaluation_result["turns_naturalness"])))
            evaluation_result["contextual_vocabulary_style_score"] = np.mean(list(map(lambda x: x["contextual_vocabulary_style"], evaluation_result["turns_naturalness"])))
            evaluation_result["overall_naturalness_score"] = np.mean(list(map(lambda x: x["naturalness_score"], evaluation_result["turns_naturalness"])))
            dialogues[i].naturalness_evaluation = NaturalnessEvaluation.model_validate(
                evaluation_result
            )
            evaluation_results.append(dialogues[i])
        return evaluation_results

    def evaluate(self, dialogues: List[Dialogue], gen_params={}):
        """ "
        Evaluate the naturalness of the conversation in the dialogues.
        Args:
            dialogues (List[Dialogue]): List of Dialogue objects to evaluate.
            gen_params (dict): Additional parameters for the LLM generation.
        Returns:
            List[Dialogue]: List of Dialogue objects with naturalness evaluation results filled in.
        """
        prompts = self._construct_prompt(dialogues)
        logger.info(f"Evaluating naturalness for {len(prompts)} conversations...")
        outputs = self.llm.generate(prompts, NaturalnessEvaluation, **gen_params)
        evaluation_results = self._fill_back(outputs, dialogues)
        logger.info(
            f"Evaluated naturalness for {len(evaluation_results)} conversations."
        )
        return evaluation_results
