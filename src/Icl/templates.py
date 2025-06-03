cmre_icl_template = [
'''接下来你需要完成隐喻成分抽取任务。任务定义为，抽取给定隐喻句中的本体（比喻中原始的、实际的对象或概念）、喻体（比喻中描述本体的对象或概念）、喻体动作（比喻中描述行为或动作的部分）、喻体属性（比喻中描述特征或属性的部分）、喻体部件（比喻中表示整体的一部分的对象或概念）。句中可能存在多组比喻，请抽取所有可能的比喻对。比喻对存在以下四种情况：
[本体, 喻体]，[本体, 喻体动作]，[本体, 喻体属性]，[本体, 喻体部件]
为便于你进一步了解任务详情，提供以下示例作为参考：

{example}
请确保你的输出格式与示例保持一致！现在，请你基于以上示例，对以下的句子完成隐喻成分的抽取：\n"{input}"\n正确输出：\n'''
]

csr_icl_template = ['''接下来你需要完成隐喻成分抽取任务。任务目标如下：
1. 抽取给定隐喻句的本体。本体是被比喻的事物，一般为名词或代词。
2. 抽取给定隐喻句的喻体。喻体是基于本体延伸出的概念或事物，一般为名词。
3. 确保本体和喻体的精简性。需保证本体和喻体的共同属性，而不是简单的去除所有的修饰词。
这儿有一些示例供你参考:

{example}
请确保你的输出格式与示例保持一致！准备好了吗？让我们在以下句子中应用起来吧！\n"{input}"\n你的正确输出：\n''']

lcc_icl_template = ['''Please identify and extract metaphorical components from metaphorical sentences. These components include: Tenor – the actual object or concept being described; Vehicle – the object or concept used to describe the tenor; Vehicle Action – the part of the metaphor that conveys an action or behavior; Vehicle Attribute – the part of the metaphor that conveys a characteristic or quality. There are three types of metaphorical mappings:
[[Tenor], [Vehicle]], [[Tenor], [Vehicle Action]], [[Tenor], [Vehicle Attribute]]. Below are some examples for reference:

{example}
Please ensure that your output strictly follows the same format as shown in the examples. Now, given the metaphorical sentence: "{input}" \nWhat is your correct output?\n''']

cmre_template = [
'''接下来你需要完成隐喻成分抽取任务。任务定义为，抽取给定隐喻句中的本体（比喻中原始的、实际的对象或概念）、喻体（比喻中描述本体的对象或概念）、喻体动作（比喻中描述行为或动作的部分）、喻体属性（比喻中描述特征或属性的部分）、喻体部件（比喻中表示整体的一部分的对象或概念）。句中可能存在多组比喻，请抽取所有可能的比喻对。比喻对存在以下四种情况：
[本体, 喻体]，[本体, 喻体动作]，[本体, 喻体属性]，[本体, 喻体部件]
现在，请对以下的句子完成隐喻成分的抽取：\n"{input}"\n正确输出：\n'''
]

csr_template = ['''接下来你需要完成隐喻成分抽取任务。任务目标如下： 
1. 抽取给定隐喻句的本体。本体是被比喻的事物，一般为名词或代词。
2. 抽取给定隐喻句的喻体。喻体是基于本体延伸出的概念或事物，一般为名词。
3. 确保本体和喻体的精简性。需保证本体和喻体的共同属性，而不是简单的去除所有的修饰词。
准备好了吗？让我们在以下句子中应用起来吧！\n"{input}"\n你的正确输出：\n''']

lcc_template = ['''Please identify and extract metaphorical components from metaphorical sentences. These components include: Tenor – the actual object or concept being described; Vehicle – the object or concept used to describe the tenor; Vehicle Action – the part of the metaphor that conveys an action or behavior; Vehicle Attribute – the part of the metaphor that conveys a characteristic or quality. There are three types of metaphorical mappings:
[[Tenor], [Vehicle]], [[Tenor], [Vehicle Action]], [[Tenor], [Vehicle Attribute]]. 
Now, given the metaphorical sentence: "{input}" \nWhat is your correct output?\n''']

csr_ficl_template = '''请你识别并抽取隐喻句中的本体（被比喻的事物）和喻体（基于本体延伸出的概念或事物）。以下是一些供你参考的示例：
示例1：
输入："我们只好呼朋唤友，像一群老鼠绕来绕去。"
输出："[我们], [老鼠]"
示例2：
输入："日记，我们的心灵空间，心语的倾诉对象，我们可以在这里留下我们那无声的哭，辛酸的泪，就像一个知音，一个肝胆相照的朋友。"
输出："[日记], [知音, 朋友]"
示例3：
输入："东面的白堤和西南的苏堤就像两条绿带子，漂在西湖上。"
输出："[白堤, 苏堤], [带子]"
请确保本体和喻体的精简性，同时保证输出与示例的输出格式一致！现在，给定隐喻句输入："{input}"\n你的输出是？\n
'''

cmre_ficl_template = '''请你识别并抽取隐喻句中隐喻成分（本体-比喻中实际的对象或概念、喻体-比喻中描述本体的对象或概念、喻体动作-比喻中描述行为或动作的部分、喻体属性-比喻中描述特征或属性的部分、喻体部件-比喻中表示整体的一部分的对象或概念）。比喻对存在四种情况：
[本体, 喻体]，[本体, 喻体动作]，[本体, 喻体属性]，[本体, 喻体部件]，以下是一些供你参考的示例：
示例1：
输入："将深色的窗帘吹得像旗子一般飘舞起来。"
输出："[深色的窗帘, 旗子]"
示例2：
输入："油价发烧，经济咳嗽。"
输出："[油价, 发烧], [经济, 咳嗽]"
示例3：
输入："银行首席投资策略师认为，牛市虽然已老，但还没有致命的创伤。"
输出："[牛市, 老], [牛市, 致命的创伤]"
请保证输出与示例的输出格式一致！现在，给定隐喻句输入："{input}"\n你的输出是？\n
'''

lcc_ficl_template = '''Please identify and extract metaphorical components from metaphorical sentences. These components include: Tenor – the actual object or concept being described; Vehicle – the object or concept used to describe the tenor; Vehicle Action – the part of the metaphor that conveys an action or behavior; Vehicle Attribute – the part of the metaphor that conveys a characteristic or quality. There are three types of metaphorical mappings:
[Tenor, Vehicle], [Tenor, Vehicle Action], [Tenor, Vehicle Attribute]. Below are some examples for reference:
Example 1:
Input: Then provide drive by comments about "stooges" and " gun cult ."
Output: [gun, cult]
Example 2:
Input: The stock market is a roller coaster.
Output: [stock market, roller coaster]
Example 3:
Input: Government declined by 900 payroll jobs and the private sector added 4,100.
Output: [Government, declined]
Example 4: I think the last time we strayed was when one of the guys went on a income tax rant .
Output: [income tax, rant]
Please ensure that your output strictly follows the same format as shown in the examples. Now, given the metaphorical sentence: "{input}" \nWhat is your output?\n
'''

csr_no_sft_template = '''请你识别并抽取隐喻句中的本体（被比喻的事物）和喻体（基于本体延伸出的概念或事物）。以下是供你参考的示例：

{example}
请确保本体和喻体的精简性，同时保证输出与示例的输出格式一致！现在，给定隐喻句输入："{input}"\n你的输出是？\n
'''

cmre_no_sft_template = '''请你识别并抽取隐喻句中隐喻成分（本体-比喻中实际的对象或概念、喻体-比喻中描述本体的对象或概念、喻体动作-比喻中描述行为或动作的部分、喻体属性-比喻中描述特征或属性的部分、喻体部件-比喻中表示整体的一部分的对象或概念）。比喻对存在四种情况：
[本体, 喻体]，[本体, 喻体动作]，[本体, 喻体属性]，[本体, 喻体部件]，以下是供你参考的示例：

{example}
请保证输出与示例的输出格式一致！现在，给定隐喻句输入："{input}"\n你的输出是？\n
'''

fixed_examples_csr = '''示例1：
输入："我们只好呼朋唤友，像一群老鼠绕来绕去。"
输出："[我们], [老鼠]"
示例2：
输入："日记，我们的心灵空间，心语的倾诉对象，我们可以在这里留下我们那无声的哭，辛酸的泪，就像一个知音，一个肝胆相照的朋友。"
输出："[日记], [知音, 朋友]"
示例3：
输入："东面的白堤和西南的苏堤就像两条绿带子，漂在西湖上。"
输出："[白堤, 苏堤], [带子]"
示例4：
输入："然而壶口瀑布那一方的太阳却黯然失色，像一个很大的蛋黄悬在高空。"
输出："[太阳], [蛋黄]"
'''

fixed_examples_cmre = '''示例1：
输入："将深色的窗帘吹得像旗子一般飘舞起来。"
输出："[深色的窗帘, 旗子]"
示例2：
输入："油价发烧，经济咳嗽。"
输出："[油价, 发烧], [经济, 咳嗽]"
示例3：
输入："银行首席投资策略师认为，牛市虽然已老，但还没有致命的创伤。"
输出："[牛市, 老], [牛市, 致命的创伤]"
示例4：
输入："她都快被暖化了"
输出："[她, 暖化]"
'''

fixed_examples_lcc = '''Example 1:
Input: Then provide drive by comments about "stooges" and " gun cult ."
Output: [gun], [cult]
Example 2:
Input: The stock market is a roller coaster.
Output: [stock market], [roller coaster]
Example 3:
Input: Government declined by 900 payroll jobs and the private sector added 4,100.
Output: [Government], [declined]
Example 4: I think the last time we strayed was when one of the guys went on a income tax rant .
Output: [income tax], [rant]
'''
