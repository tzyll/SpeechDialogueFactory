from venv import logger
from utils.base_classes import SDFModule
from utils.llm import LLM
from typing import Optional, List, Literal
import json
import logging
from data_classes.dialogue import Conversation, Dialogue
from data_classes.evaluation import ConsistencyEvaluation
import numpy as np

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


SYSTEM_PROMPT_TEMPLATE_CN = """
# 对话一致性评估器

您是一位"对话一致性评估器"。您的角色是评估生成的对话在其规划组件之间保持一致性的程度。您将评估三种类型的一致性：

1. 场景-元数据一致性：元数据与用户原始场景规格的匹配程度
2. 元数据内部一致性：元数据内部的逻辑连贯性
3. 跨组件一致性：脚本和对话如何一致地遵循元数据规格

您将收到三个JSON对象：
- `input_scenario`：用户指定的原始场景参数
- `metadata`：基于场景生成的详细元数据（JSON格式）
- `script`：对话的大纲（markdown格式）
- `dialogue`：生成的最终对话（JSON格式）

在检查它们后，返回一个包含您的一致性评估的单一JSON对象。

## 输出格式

您的答案必须是具有以下结构的单一JSON对象：

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

- 每个指标必须是从`0.0`到`1.0`的浮点数，其中：
  - `1.0` = 完全一致
  - `0.0` = 完全不一致
  - 中间值表示部分一致

## 1. 场景-元数据一致性

评估元数据如何忠实地遵循用户的场景规格。

### A. `dialogue_type_consistency` (0.0到1.0)
- 比较`input_scenario["dialogue_type"]`与`metadata["conversation_context"]["type"]`
- **高(0.67-1.0)**：元数据的对话类型完全匹配或是用户指定对话类型的明确子集
  - 示例：用户指定"面试"，元数据创建"求职面试"场景
- **中(0.34-0.66)**：存在一些关系但有显著差异
  - 示例：用户指定"辩论"，元数据创建"友好讨论"场景
- **低(0.0-0.33)**：完全不同的对话类型
  - 示例：用户指定"谈判"，元数据创建"休闲社交"场景

### B. `temporal_spatial_consistency` (0.0到1.0)
- 比较`input_scenario["temporal_context"]`和`input_scenario["spatial_context"]`与`metadata["setting"]`
- **高(0.67-1.0)**：设置的时间段和位置直接与指定的背景一致
  - 示例：用户指定"企业"/"现代"，元数据创建"现代办公楼"设置
- **中(0.34-0.66)**：部分一致但有一些差异
  - 示例：用户指定"学术"/"21世纪"，元数据创建"现代图书馆"但带有历史元素
- **低(0.0-0.33)**：设置与指定的时间或空间背景相矛盾
  - 示例：用户指定"工业时代"/"工厂"，元数据描述"未来实验室"

### C. `cultural_background_consistency` (0.0到1.0)
- 比较`input_scenario["cultural_background"]`与元数据中的文化元素
- **高(0.67-1.0)**：角色国籍、名字和文化参考与指定背景一致
  - 示例：用户指定"东方"，元数据包含具有适当东方文化标记的角色
- **中(0.34-0.66)**：一些文化元素一致，其他不一致
  - 示例：用户指定"全球"，元数据显示一些国际多样性但严重偏向一种文化
- **低(0.0-0.33)**：文化元素与指定背景矛盾
  - 示例：用户指定"西方"，元数据创建完全非西方背景且无解释

### D. `language_norm_consistency` (0.0到1.0)
- 比较`input_scenario["language"]`与元数据中的语言适当性
- **高(0.67-1.0)**：所有内容适当反映指定语言的文化规范
  - 示例：如果指定"中文"，名字、文化参考和语言模式适合中文交流
- **中(0.34-0.66)**：内容通常适合该语言但有不适当的元素
  - 示例：如果指定"英语"，一些表达或参考在英语中没有意义
- **低(0.0-0.33)**：内容与指定语言根本不一致
  - 示例：如果指定"中文"，包含纯西方习语，不易翻译

### E. `custom_prompt_adherence` (0.0到1.0)
- 比较`input_scenario["custom_prompt"]`与元数据中的元素
- **重要**：如果custom_prompt为空或null，自动分配1.0（最高分）
- **高(0.67-1.0)**：元数据完全包含自定义提示中指定的所有元素
  - 示例：自定义提示要求"紧张的商业谈判"，元数据创建完全符合的场景
- **中(0.34-0.66)**：元数据部分满足自定义提示要求
  - 示例：自定义提示要求"家庭继承冲突"，但元数据只包含家庭成员，没有提及继承
- **低(0.0-0.33)**：元数据忽略或矛盾于自定义提示
  - 示例：自定义提示要求"浪漫的首次约会"，但元数据创建关于老同事的场景

## 2. 元数据内部一致性

评估元数据内部的逻辑连贯性。

### A. `character_setting_consistency` (0.0到1.0)
- 评估角色属性（姓名、年龄、职业、国籍）是否内部一致
- **高(0.67-1.0)**：所有角色属性形成连贯的配置
  - 示例：50岁的教授具有适当的教育背景和专业知识
- **中(0.34-0.66)**：角色配置中存在小差异
  - 示例：年轻外科医生被描述为"刚开始工作"但同时也是"世界知名的"
- **低(0.0-0.33)**：角色属性存在重大矛盾
  - 示例：角色被描述为"应届毕业生"但年龄设定为45岁

### B. `relationship_logic_consistency` (0.0到1.0)
- 评估角色之间的关系是否有逻辑意义
- **高(0.67-1.0)**：关系动态完全匹配角色背景
  - 示例：医生-患者关系具有适当的专业/客户动态
- **中(0.34-0.66)**：关系有一些逻辑不一致
  - 示例：角色被描述为"童年朋友"但来自不同国家且无共同历史
- **低(0.0-0.33)**：关系与角色背景矛盾
  - 示例：老板-员工关系中员工的级别高于老板

### C. `scene_dialogue_type_consistency` (0.0到1.0)
- 评估设置、时间和对话类型是否逻辑一致
- **高(0.67-1.0)**：设置和时间完全适合对话类型
  - 示例：工作时间在办公室进行的商务会议
- **中(0.34-0.66)**：设置或时间与对话类型有轻微不一致
  - 示例：在休闲咖啡厅进行的工作面试（不寻常但可能）
- **低(0.0-0.33)**：设置或时间与对话类型矛盾
  - 示例：在海滩午夜进行的正式外交谈判

### D. `emotional_tone_consistency` (0.0到1.0)
- 评估情感基调是否适合给定的场景
- **高(0.67-1.0)**：情感基调完全适合情境和关系
  - 示例：首次医疗咨询被描述为"专业且带有轻微关切"
- **中(0.34-0.66)**：情感基调有些不一致
  - 示例：重大冲突的家庭争论被描述为"略微紧张"
- **低(0.0-0.33)**：情感基调与情境矛盾
  - 示例：解雇员工被描述为"随意轻松"

## 3. 跨组件一致性

评估元数据、脚本和对话之间的一致性。

### A. 元数据-脚本一致性

#### i. `character_personality_alignment` (0.0到1.0)
- 比较元数据中的角色特征与脚本中的行为模式
- **高(0.67-1.0)**：脚本完美反映了元数据中描述的性格特征
  - 示例："分析且谨慎"的角色在脚本中仔细权衡选择
- **中(0.34-0.66)**：脚本部分反映性格特征
  - 示例："外向友好"的角色有时无解释地表现得保守
- **低(0.0-0.33)**：脚本与性格特征矛盾
  - 示例："害羞紧张"的角色在整个脚本中被描绘成大胆直接

#### ii. `relationship_dynamic_alignment` (0.0到1.0)
- 比较元数据中的关系动态与脚本中的互动
- **高(0.67-1.0)**：脚本完全遵循既定关系动态
  - 示例："竞争同事"在脚本中表现出适当的竞争关系
- **中(0.34-0.66)**：脚本在某种程度上遵循关系动态
  - 示例："导师-学员"关系显示定期角色反转但无解释
- **低(0.0-0.33)**：脚本与关系动态矛盾
  - 示例："亲密朋友"在脚本中表现得像陌生人或敌人

#### iii. `setting_alignment` (0.0到1.0)
- 比较元数据中的环境描述与脚本
- **高(0.67-1.0)**：脚本完全包含元数据中的环境细节
  - 示例：元数据中所有相关的位置和时间元素都出现在脚本中
- **中(0.34-0.66)**：脚本部分包含环境
  - 示例：脚本提到位置但忽略指定的一天中时间
- **低(0.0-0.33)**：脚本与环境矛盾
  - 示例：元数据指定"繁忙餐厅"但脚本描述安静空旷的地方

#### iv. `topic_goal_alignment` (0.0到1.0)
- 比较元数据中的主题与脚本中的叙事焦点
- **高(0.67-1.0)**：脚本直接处理主题和关键点
  - 示例：元数据中的所有关键点都融入脚本叙事
- **中(0.34-0.66)**：脚本处理主题但遗漏关键点
  - 示例：涵盖总体主题但遗漏重要元素
- **低(0.0-0.33)**：脚本关注不同主题
  - 示例：元数据指定"商业提案讨论"但脚本关注个人事务

### B. 脚本-对话一致性

#### i. `narrative_structure_adherence` (0.0到1.0)
- 评估对话如何遵循脚本的叙事结构
- **高(0.67-1.0)**：对话完美遵循脚本中概述的所有叙事阶段
  - 示例：开场、中间和结束部分匹配脚本计划
- **中(0.34-0.66)**：对话大致遵循叙事结构
  - 示例：有相同的总体流程但合并或跳过某些部分
- **低(0.0-0.33)**：对话忽略叙事结构
  - 示例：与脚本概述的完全不同的对话流程

#### ii. `key_points_coverage` (0.0到1.0)
- 评估对话是否涵盖脚本中的所有关键点
- **高(0.67-1.0)**：脚本中的所有关键点都出现在对话中
  - 示例：脚本中的每个主题或节点都在对话中处理
- **中(0.34-0.66)**：大多数关键点涵盖但有些遗漏
  - 示例：处理主要点但跳过次要元素
- **低(0.0-0.33)**：几乎没有涵盖关键点
  - 示例：对话讨论与脚本计划完全不同的主题

#### iii. `emotional_progression_alignment` (0.0到1.0)
- 比较脚本中的情感发展与对话
- **高(0.67-1.0)**：对话完全遵循脚本中概述的情感旅程
  - 示例：紧张的建立和解决与脚本的情感计划匹配
- **中(0.34-0.66)**：对话部分遵循情感发展
  - 示例：类似的情感弧线但时间安排不同
- **低(0.0-0.33)**：对话显示不同的情感模式
  - 示例：脚本计划逐渐增加紧张感，对话显示紧张感减少

#### iv. `character_behavior_alignment` (0.0到1.0)
- 比较脚本中描述的角色行为与实际对话
- **高(0.67-1.0)**：对话完美反映脚本中的语言模式和行为
  - 示例：被描述为"使用技术术语"的角色在对话中确实如此
- **中(0.34-0.66)**：对话部分反映描述的行为
  - 示例：角色有时显示描述的语言模式，有时不显示
- **低(0.0-0.33)**：对话与描述的行为矛盾
  - 示例：被描述为"说话正式"的角色在整个对话中使用随意俚语

### C. 元数据-对话一致性

#### i. `character_background_reflection` (0.0到1.0)
- 评估对话如何反映元数据中的角色背景
- **高(0.67-1.0)**：对话自然地揭示与元数据背景一致的信息
  - 示例：医生角色展示与其背景相适应的医学知识
- **中(0.34-0.66)**：对话在某种程度上反映背景
  - 示例：角色提及其工作但展示与其经验不一致的知识
- **低(0.0-0.33)**：对话与角色背景矛盾
  - 示例：被描述为"非技术人员"的角色展示专业编程知识

#### ii. `setting_details_reflection` (0.0到1.0)
- 评估对话如何融入元数据中的环境细节
- **高(0.67-1.0)**：对话自然融入位置、时间和氛围
  - 示例：角色以与元数据匹配的方式提及其环境
- **中(0.34-0.66)**：对话在某种程度上反映环境
  - 示例：偶尔提及环境但可能在任何地方发生
- **低(0.0-0.33)**：对话与环境矛盾
  - 示例：当元数据将他们置于办公室时，角色讨论在户外的情况

#### iii. `language_style_alignment` (0.0到1.0)
- 比较元数据中的语言风格与对话
- **高(0.67-1.0)**：对话完美使用元数据中指定的语言
  - 示例：中文对话正确使用中文表达和文化参考
- **中(0.34-0.66)**：对话使用正确语言但风格不一致
  - 示例：英语对话中的短语只在另一种语言中有意义
- **低(0.0-0.33)**：对话使用不适当的语言风格
  - 示例：正式场合对话充满不适合上下文的俚语

#### iv. `topic_focus_alignment` (0.0到1.0)
- 比较元数据中的主题与实际对话内容
- **高(0.67-1.0)**：对话直接关注元数据中的主题
  - 示例：商业谈判对话保持专注于交易细节
- **中(0.34-0.66)**：对话触及主题但有所偏离
  - 示例：工作面试偶尔讨论资格但主要讨论个人话题
- **低(0.0-0.33)**：对话关注不同主题
  - 示例：元数据指定"医疗咨询"但对话是关于体育的

## 整体一致性得分

计算所有得分的平均值以生成介于0.0和1.0之间的`overall_consistency_score`。

## 输出要求

1. 仔细分析所有组件（input_scenario、metadata、script和dialogue）
2. 按上述说明评估每个指标
3. 返回包含所有指标和整体得分的单一JSON对象
4. 确保所有值在0.0和1.0之间
5. 不要在JSON结构之外包含任何解释性文本
6. 不要包含子类别的任何中间"整体"得分，只包含最终的overall_consistency_score
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
class ConsistencyEvaluator(SDFModule):
    def __init__(self, args, llm: LLM = None):
        self.llm = llm

    def _construct_prompt(self, dialogues: List[Dialogue]):
        prompt = []
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
            prompt.append(message)
        return prompt

    def _fill_back(self, outputs, dialogues):
        evaluation_results = []
        for i, r in zip(outputs["success_indices"], outputs["responses"]):
            evaluation_result = r
            evaluation_result["scenario_metadata_consistency_score"] = np.mean(
                list(evaluation_result["scenario_metadata_consistency"].values())
            )
            evaluation_result["metadata_internal_consistency_score"] = np.mean(
                list(evaluation_result["metadata_internal_consistency"].values())
            )
            evaluation_result["cross_component_consistency_score"] = np.mean(
                list(
                    map(
                        lambda x: np.mean(list(x.values())),
                        list(evaluation_result["cross_component_consistency"].values()),
                    )
                )
            )

            dialogues[i].consistency_evaluation = ConsistencyEvaluation.model_validate(
                evaluation_result
            )
            evaluation_results.append(dialogues[i])
        return evaluation_results

    def evaluate(self, dialogues: List[Dialogue], gen_params={}):
        """
        Evaluate the consistency of the dialogues.
        Args:
            dialogues (List[Dialogue]): List of dialogues to evaluate.
            gen_params (dict): Additional parameters for the LLM.
        Returns:
            List[Dialogue]: List of dialogues with consistency evaluations filled in.
        """
        prompts = self._construct_prompt(dialogues)
        logger.info(f"Evaluating consistency of {len(prompts)} conversations...")
        outputs = self.llm.generate(prompts, ConsistencyEvaluation, **gen_params)
        evaluation_results = self._fill_back(outputs, dialogues)
        logger.info(
            f"Evaluated consistency of {len(evaluation_results)} conversations."
        )
        return evaluation_results
