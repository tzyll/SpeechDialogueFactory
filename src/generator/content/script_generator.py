from utils.base_classes import SDFModule
from utils.llm import LLM
from pydantic import BaseModel, Field
from typing import Optional, List
import json
import logging

logger = logging.getLogger(__name__)
from data_classes.dialogue import DialogueScenario, Dialogue, Metadata


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


SYSTEM_PROMPT_TEMPLATE_CN = """
您是一位对话脚本设计师。您的任务是根据提供的元数据创建一个详细的双人对话脚本大纲。这个脚本将作为生成最终对话的蓝图。

请按照以下精确的部分来输出您的脚本，采用结构化的markdown格式：

### 场景：
写一个简短的段落描述物理环境、氛围和角色的初始情况。使用元数据中设置和上下文的具体细节。

### 叙事流程：
将对话分成4个清晰的部分（必须与元数据中的expected_turns完全匹配）：
1. 开场（通常2-3个回合）
2. 初步讨论（2-4个回合）
3. 主要讨论（3-5个回合）
4. 结束（2-3个回合）

对于每个部分：
- 列出具体的对话节点
- 显示main_topic的清晰进展
- 包含元数据中的所有key_points

### 角色行为：
对于每个角色（role_1和role_2）：
- 列出3-5个具体的说话模式或行为特征
- 基于元数据中的personality_traits
- 包括他们典型的反应和举止
- 展示他们的职业和背景如何影响他们的言语

### 情感发展：
概述3个情感发展阶段：
- 开始：两个角色的初始情感状态
- 中间：情感如何在主要讨论中演变
- 结束：最终情感状态和解决方案

重要要求：
1. 保持一切与提供的元数据一致
2. 在流程中自然地包含元数据中的所有key_points
3. 根据角色的self_introductions保持角色一致性
4. 在所有互动中考虑relationship_dynamic
5. 保持脚本集中，避免不必要的元素

以下是对话场景和相应元数据的示例：

### 对话场景：
```json
{
    "dialogue_type": "教育互动",
    "temporal_context": "现代",
    "spatial_context": "大学图书馆",
    "cultural_background": "东方文化背景",
    "custom_prompt": "创建一个大学图书馆学习讨论的对话场景。",
    "language": "Chinese"
  }
```

### 元数据：
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

以下是您的脚本应该如何构建：

### 场景：
大学图书馆自习室，傍晚时分。房间安静，灯光柔和。李明和王芳坐在一张桌子旁，面前摊开着教科书和笔记本电脑。

### 叙事流程：
1. 开场（2个回合）：
   - 王芳略显慌乱地到达，向李明打招呼
   - 李明一边整理学习资料一边欢迎她

2. 初步讨论（3个回合）：
   - 王芳表达对递归概念的焦虑
   - 李明提供安慰并建议从基础例子开始
   - 王芳分享她正在struggle的特定问题

3. 主要学习讨论（4个回合）：
   - 李明使用实际例子解释递归
   - 王芳就基本情况提出澄清问题
   - 李明分享他分解递归问题的方法
   - 王芳对特定概念有了"顿悟"的感觉

4. 结束（3个回合）：
   - 王芳建议计划另一次学习会议
   - 李明同意并提到即将到来的项目
   - 关于下次会面时间的简短交流

### 角色行为：
李明：
- 说话使用清晰、有条理的句子
- 偶尔表现出自己对考试的压力
- 使用类比来解释概念
- 保持乐于助人但略带焦虑的态度

王芳：
- 当困惑时直接提问
- 理解概念时表现出热情
- 偶尔会因项目想法而离题
- 使用更加随意的语言

### 情感发展：
- 开始：轻微紧张（王芳的压力，李明想要帮助的愿望）
- 中间：随着概念变得更清晰，信心增长
- 结束：对理解材料感到轻松和乐观

### 语言考虑：
- 两个角色都是中国的母语为中文的人
- 李明因其勤奋的性格使用更精确的技术术语
- 王芳使用更多日常表达和填充词
- 他们的中国文化背景影响了他们含蓄而礼貌的交流风格
- 两人都保持典型的中国大学生之间尊重但不过分正式的语言风格

根据这个格式生成一个完整的脚本。严格基于提供的元数据JSON的所有元素。
"""

USER_PROMPT_TEMPLATE_CN = """
## 对话场景
{scenario}

## 元数据
```json
{metadata}
```
"""


@SDFModule.set_role("generator")
class ScriptGenerator(SDFModule):
    def __init__(self, args, llm: LLM):
        self.llm = llm

    def _construct_prompt(self, dialogues: List[Dialogue]):
        messages = []
        for dialogue in dialogues:
            dialogue_langue = dialogue.scenario.dialogue_language
            SPROMPT = SYSTEM_PROMPT_TEMPLATE_CN if dialogue_langue == "Chinese" else SYSTEM_PROMPT_TEMPLATE
            UPROMPT = USER_PROMPT_TEMPLATE_CN if dialogue_langue == "Chinese" else USER_PROMPT_TEMPLATE
            message = [
                {"role": "system", "content": SPROMPT},
                {
                    "role": "user",
                    "content": UPROMPT.format(
                        scenario=dialogue.scenario.to_json(pretty=True),
                        metadata=dialogue.metadata.to_json(pretty=True),
                    ),
                },
            ]
            messages.append(message)
        return messages

    def _fill_back(self, outputs, dialogues):
        remaining_dialogues = []
        for i, r in zip(outputs["success_indices"], outputs["responses"]):
            script = r
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
