from utils.base_classes import SDFModule
from utils.llm import LLM
from pydantic import BaseModel, Field
from typing import Optional, List
import json
import logging
from data_classes.dialogue import DialogueScenario, Dialogue, Metadata

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

SYSTEM_PROMPT_TEMPLATE_CN = """
您是一位对话元数据设计师。您的任务是为一个双人对话生成真实且引人入胜的元数据。

## 输入信息

您将获得一个JSON格式的对话场景，其中包含以下字段：
- dialogue_type（例如，面试，辩论，谈判）
- temporal_context（时间段或时代）
- spatial_context（一般环境）
- cultural_background（文化影响）
- language（"English"或"Chinese"）
- custom_prompt（可选的用户特定需求）

您必须使用这些信息作为元数据创建的基础，确保您设计的所有元素与这些参数一致。特别注意language字段 - 如果指定为"Chinese"，所有名称、职业和内容都应适合中文使用者和文化背景。

如果用户提供了custom_prompt，您必须仔细将这些特定要求融入到您的元数据生成中。custom_prompt优先于一般指南，可能包含有关角色关系、对话动态、特定主题或情感基调的特殊指示，这些都必须反映在您的输出中。

## 输出要求

元数据应采用符合以下要求的JSON格式：

1. 创建现实的、具体的日常生活场景
2. 设计鲜明、立体的角色，具有明确的个性
3. 建立清晰的关系动态和对话目的
4. 保持所有细节与提供的目标和位置一致

元数据必须包括以下组件：
- setting：位置、一天中的时间和背景上下文
- role_1和role_2：两位具有详细特征的说话者
- conversation_context：互动的结构和目标（融入提供的目标）

## 环境要求

对于setting字段，包括：
- location：具体的物理或虚拟位置（比spatial_context输入更详细）
- time_of_day：早上、下午、傍晚或夜晚
- context：情况和环境的详细描述（30-50个词）
- atmosphere：环境的氛围或感觉

## 角色要求

对于每个角色，包括：
- name：真实的全名
- gender："male"或"female"
- age：10至60岁之间
- occupation：具体的工作或角色
- nationality：角色的国籍
- personality_traits：至少1个定义特征的列表
- relationship_context：他们在当前情况中的角色
- self_introduction：详细的段落（50+字符），描述他们的个性和背景

## 对话上下文要求

对于conversation_context，指定：
- type：互动的类别
- main_topic：主要讨论主题
- relationship_dynamic：说话者如何相互关联
- emotional_tone：整体情绪
- expected_duration：大约时间（<5分钟）
- expected_turns：交流次数（8-12次）
- key_points：要涵盖的主要事件/主题列表

## 重要要求

1. 仅输出有效的JSON，无额外文本或解释
2. 使场景具体而非笼统
3. 创建多样但现实的角色组合
4. 确保所有场景元素逻辑上相互关联
5. 用自然流畅的语言撰写self_introductions
6. 使expected_turns对场景合理（通常为8-12次）

以下是预期JSON格式的示例：
```json
{
  "setting": {
    "location": "大学图书馆自习室",
    "time_of_day": "傍晚",
    "context": "期末考试复习时段",
    "atmosphere": "专注且略带紧张"
  },
  "role_1": {
    "name": "李明",
    "gender": "male",
    "age": 20,
    "occupation": "大学本科生",
    "nationality": "中国",
    "personality_traits": [
      "勤奋",
      "对成绩焦虑",
      "乐于助人"
    ],
    "relationship_context": "学习小组伙伴",
    "self_introduction": "李明是一名计算机科学专业的大二学生，他非常重视自己的学业。他擅长向他人解释技术概念，但经常对考试感到焦虑。尽管有自己的压力，他真诚地喜欢帮助同学理解困难的材料。他以创建详细的学习笔记和在图书馆待到很晚而闻名。"
  },
  "role_2": {
    "name": "王芳",
    "gender": "female",
    "age": 19,
    "occupation": "大学本科生",
    "nationality": "中国",
    "personality_traits": [
      "乐观",
      "学习快",
      "略微缺乏条理"
    ],
    "relationship_context": "寻求帮助的同班同学",
    "self_introduction": "王芳是一个聪明而热情的学生，她能快速掌握概念，但在保持一致的学习习惯方面有困难。她和李明上同一门编程课，虽然她对实际应用理解得很好，但有时在理论概念上遇到困难。她积极的态度和对学习的真诚兴趣使她成为一个愉快的学习伙伴。"
  },
  "conversation_context": {
    "type": "学习讨论",
    "main_topic": "准备即将到来的编程考试",
    "relationship_dynamic": "友好的同学关系",
    "emotional_tone": "互相支持，但因考试压力有潜在紧张",
    "expected_duration": "5分钟",
    "expected_turns": 12,
    "key_points": [
      "讨论困难的递归概念",
      "王芳询问关于特定练习问题",
      "李明分享他的学习技巧",
      "计划下一次学习会议",
      "简短讨论课程项目"
    ]
  }
}
```

生成一个完整、有效的JSON结构，遵循这些要求。JSON应该讲述一个关于这两个人如何互动以及他们将讨论什么的连贯故事。
"""


USER_PROMPT_TEMPLATE_CN = """
## 对话场景
```json
{scenario}
```
"""


@SDFModule.set_role("generator")
class MetadataGenerator(SDFModule):
    def __init__(self, args, llm: LLM=None):
        self.llm = llm

    def _construct_prompt(self, dialogues):
        for i, dialogue in enumerate(dialogues):
            dialogue_langue = dialogue.scenario.dialogue_language
            SPROMPT = SYSTEM_PROMPT_TEMPLATE_CN if dialogue_langue == "Chinese" else SYSTEM_PROMPT_TEMPLATE
            UPROMPT = USER_PROMPT_TEMPLATE_CN if dialogue_langue == "Chinese" else USER_PROMPT_TEMPLATE
            created_prompts = []
            message = [
                {"role": "system", "content": SPROMPT},
                {
                    "role": "user",
                    "content": UPROMPT.format(
                        scenario=dialogue.scenario.to_json(pretty=True),
                    ),
                },
            ]
            created_prompts.append(message)
        return created_prompts

    def _fill_back(self, outputs, dialogues):
        remaining_dialogues = []
        for i, r in zip(outputs["success_indices"], outputs["responses"]):
            metadata = r
            dialogues[i].metadata = Metadata.model_validate(metadata)
            remaining_dialogues.append(dialogues[i])
        return remaining_dialogues

    def generate(self, dialogues: List[Dialogue], gen_params={}):
        prompts = self._construct_prompt(dialogues)
        logger.info(f"Generating {len(dialogues)} metadata...")
        outputs = self.llm.generate(prompts, Metadata, **gen_params)
        remaining_dialogues = self._fill_back(outputs, dialogues)
        logger.info(f"Generated {len(remaining_dialogues)} metadata.")
        return remaining_dialogues
