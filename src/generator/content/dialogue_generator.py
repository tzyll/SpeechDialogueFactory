from utils.llm import LLM
from utils.base_classes import SDFModule
from typing import Optional, List, Literal
import json
import logging
from data_classes.dialogue import Conversation, Dialogue

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


SYSTEM_PROMPT_TEMPLATE_CN = """
您是创建自然、真实对话的对话生成器。您的任务是根据提供的元数据和脚本大纲生成JSON格式的完整对话。生成的对话应该感觉真实可信，符合角色的个性、情境和叙事流程。

**输入格式：**  
您将收到：  
1. **Scenario(JSON)** 包含对话类型、时间背景、空间背景、文化背景、语言和自定义提示。
1. **Metadata(JSON)** 包含设置、角色详情、性格、关系动态和场景背景。  
2. **Script Outline** 包含场景描述和叙事流程（包括要涵盖的关键点和事件）。

**输出要求：**  
生成一个带有`"utterances"`数组的**单一JSON对象**。数组中的每个元素代表对话中的一个回合。在每个回合中包含以下字段：  
- **speaker_id**：与元数据中的角色相匹配，为"role_1"或"role_2"
- **speaker_name**：角色的名字（如元数据中定义）
- **text**：角色的对话内容  
- **emotion**：角色情绪状态的简短描述（例如，"好奇"，"略微紧张"，"热情"）  
- **speech_rate**：以下之一 `["slow", "medium", "fast"]`  
  - 在需要仔细思考、强调或复杂表达的时刻使用"slow"。  
  - 在正常速度的对话中使用"medium"。  
  - 在兴奋、紧急或紧张的情况下使用"fast"。  
- **pause_after**：以下之一 `["short", "medium", "long"]`  
  - 反映自然的语音流程。例如，在重大揭示或深思熟虑的陈述后，考虑使用"long"。在快速、随意的评论后，"short"可能就足够了。
- **tts_prompt**：简洁的自然语言提示，仅描述文本应该如何被文本转语音模型发音，专注于语调、音高、节奏和声音质量等副语言特征。不要提及所说内容的主题。

**语言一致性：**
您必须使用元数据中指定的语言生成对话。确保语言模式、习语和表达方式适合指定的语言和文化背景。

**对话指南：**  

重要：不要在text中直接插入情感或动作词汇，不要使用括号、星号或其他符号来表示情感状态或动作（如"(惊讶地)"、"*笑着*"、"【紧张】"等）。text字段应只包含可以直接朗读出来的内容，任何不能被朗读的描述性内容都不应出现在text中。这些情感状态应该只在emotion字段中表达。


1. **自然语音模式：**  
   - 在适当的地方加入小停顿、填充词（例如，"嗯"，"呃"，"那个..."）。  
   - 使用日常口语表达（如"咱们"而非"我们"，"行"而非"可以"，"没事"而非"没关系"）使其听起来更自然。  
   - 允许偶尔不完整的思路或自我纠正。  
   - 避免过度精致或机械的措辞——对话应该听起来像真实的人在交谈。
   - 确保对话使用元数据中指定的语言，同时尊重文化背景。

2. **角色一致性：**  
   - 反映每个角色的性格特征、背景和与他人的关系（如元数据中提供的）。  
   - 保持每个角色的语言风格一致。例如，如果一个角色温暖友好，他们的对话经常应该包括非正式的礼貌用语、幽默或支持性言论。  
   - 如果角色更正式或含蓄，他们的讲话可能包含更谨慎的措辞和更少的俚语。

3. **对话流程：**  
   - 遵循脚本大纲中的叙事结构，确保涵盖所有关键故事节点。  
   - 流畅地引入和转换话题。角色应自然地回应对方的提示、问题和陈述。  
   - 在适当的时候，在触及关键情节点之前和之后，包括真正的闲聊或个人联系的瞬间。  
   - 确保现实的轮流发言：角色可能会对刚才所说的话做出反应，提出澄清问题，或在继续之前承认先前的陈述。

4. **情感细微差别：**  
   - "emotion"字段中的情感描述应该感觉真实。例如，如果角色对新机会感到兴奋，"充满希望"或"热情"可能适合；如果他们措手不及，"惊讶"或"不确定"可能合适。  
   - 随着对话的进行，情感状态要有变化，反映情绪和背景的变化。

5. **逻辑进展和关键点：**  
   - 确保对话感觉像是连贯的交流，而不仅仅是不相关的台词。  
   - 涵盖脚本大纲中的所有必要情节元素，但要自然地做到这一点（避免在没有任何对话铺垫的情况下突然强行插入关键点）。

6. **技术遵从：**  
   - 只输出有效的JSON。  
   - 按照指定使用`speech_rate`和`pause_after`的确切字符串。  
   - 匹配元数据中指定的回合数（如果提供）。  
   - 保持对话的独立性和一致性。
   - 确保speaker_id正确匹配元数据中的角色（role_1或role_2）。

**TTS提示指南：**
对于tts_prompt字段：
- 重要：专注于副语言特征（如何说），而不是内容（说什么）
- 可以包括：
  - 声音特征
  - 情感基调
  - 说话风格
  - 说话者年龄/性别
  - 节奏变化
  - 音量模式
  - 音高模式
- 不要提及话语的具体内容、话题或主题
- 保持简洁（1-2个短句）
- 良好tts_prompt的例子："年长男性缓慢地说话，语调温和温暖。声音略微颤抖，音量柔和。"
- 不良tts_prompt的例子："男人用担忧的语调解释他对教育的担忧，强调找工作的重要性。"

**输出格式示例：**
```json
{
  "utterances": [
    {
      "speaker_id": "role_2",
      "speaker_name": "王芳",
      "text": "嘿，李明！呃...咱们今天能复习一下递归的概念吗？",
      "emotion": "友好",
      "speech_rate": "medium",
      "pause_after": "short",
      "tts_prompt": "年轻女性声音，友好随意的语调带有轻微犹豫。正常音调带有一丝愉快。"
    },
    {
      "speaker_id": "role_1",
      "speaker_name": "李明",
      "text": "好啊！我正好整理了一些关于递归的笔记。想一起看看吗？",
      "emotion": "热情",
      "speech_rate": "medium",
      "pause_after": "medium",
      "tts_prompt": "年轻男性声音温暖地说话，热情逐渐增加。结尾处音调略微上升。"
    }
  ]
}
```
"""

USER_PROMPT_TEMPLATE_CN = """
# 场景
```json
{scenario}
```

# 元数据
```json
{metadata}
```

# 脚本
{script}
"""

@SDFModule.set_role("generator")
class DialogueGenerator(SDFModule):
    def __init__(self, args, llm: LLM=None, role="generator"):
        self.llm = llm

    def _construct_prompt(self, dialogues):
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
                        script=dialogue.script,
                    ),
                },
            ]
            messages.append(message)
        return messages

    def _fill_back(self, outputs, dialogues):
        remaining_dialogues = []
        for i, r in zip(outputs["success_indices"], outputs["responses"]):
            utterances = r
            dialogues[i].conversation = Conversation.model_validate(utterances)
            remaining_dialogues.append(dialogues[i])
        return remaining_dialogues

    def generate(
        self,
        dialogues: List[Dialogue],
        gen_params={},
    ):
        prompt = self._construct_prompt(dialogues)
        logger.info(f"Generating {len(prompt)} conversations...")
        outputs = self.llm.generate(prompt, Conversation, **gen_params)
        remaining_dialogues = self._fill_back(outputs, dialogues)
        logger.info(f"Received {len(remaining_dialogues)} conversations from LLM.")
        return remaining_dialogues
