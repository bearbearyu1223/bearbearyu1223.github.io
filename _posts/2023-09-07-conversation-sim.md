---
layout: post
title: "Conversation Simulation : Watch AI Buddies Spark Conversations!"
categories: chatbot
author: 
- Han Yu
---
### TL;DR
 This project is a demonstration of a simple conversational simulation environment, powered by LLM agents. It takes inspiration from [ChatArena](https://github.com/chatarena/chatarena). The main objective here is to enable conversation simulations through LLM agents (e.g., the converstation can be task-oriented or just a chitchat). To accomplish this, the frameworks and concepts originally developed in ChatArena has been leveraged and tailored specifically for this conversational simulation experiment. 
 More information can be found on [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/bearbearyu1223/Multi-Agents-Simulation-Powered-by-LLMs).

### Set Up Local Virtual Environment
* Step 1: Install Miniconda on MacOS, see instruction [here](https://docs.conda.io/en/latest/miniconda.html). 
* Step 2: Create a default conda env with Python 3.10: 
``conda create --name agent_sim python=3.10 -y``
* Step 3: Activate the conda env created above: 
``conda activate agent_sim``
* Step 4: Install required libraries in the conda env: 
``` 
pip3 install -r requirements.txt
```
* Step 5: Deactivate the conda env when you are done (note: you need activate this virtual env to work on the app)
```
conda deactivate 
```

### Set Up Open AI Secrete for Local Dev 
`OPENAI_API_KEY` will be needed when calling the OpenAI API endpoints (DEFAULT_MODEL used in this project is "gpt-3.5-turbo"), so do recommend exporting `OPENAI_API_KEY` as an enviroment variable on your local dev machine. 

### Run the Simulation 
You can play with any examples under the `examples` folder and run the converstation simulation in an interactive mode:
```
python examples/example_food_qna_zh_CN.py 
```

In this example, "Crystal" is utilizing an AI assistant named "Nova" to brainstorm ideas for a fast and nutritious school lunch for her daughter. "Ella" serves as a "moderator" responsible for evaluating whether the conversations between Nancy and Nova adhere to the following rules for termination:

* The conversation can conclude if Nova successfully fulfills all of Crystal's requests, and Crystal expresses satisfaction with Nova's assistance.
* The conversation can also end if Nova is unable to fulfill Crystal's requests.

If Ella believes the conversation should conclude, she will state "<<<<<<END_OF_CONVERSATION>>>>>>"; otherwise, she will say "converstation can continue". 

Please refer to the screenshot below for an illustration of the simulation results.![example_1](/assets/picture/2023_09_07_conversation_sim/2023_09_07_conversation_sim.png)

Once simulation ends, one can save the converstation history as a json file. See a sample json file below:
```
[
    {
        "agent_name": "Crystal",
        "content": "嗨，Nova！我需要一些关于为我女儿准备健康午餐的建议。她现在7岁，我希望能给她提供快速但营养丰富的选项。你有什么建议吗？",
        "turn": 0,
        "timestamp": "1694143996023456000",
        "visible_to": "all",
        "msg_type": "text"
    },
    {
        "agent_name": "Nova",
        "content": "嗨，Crystal！为了给你的女儿提供快速又营养丰富的午餐，我有几个建议。首先，你可以考虑给她提供一份均衡的三明治，里面包含蔬菜、蛋白质和全麦面包。你还可以添加一些水果或酸奶作为伴侣。另外，你可以尝试制作一些健康的小吃，例如蔬菜切片搭配花生酱或者酸奶蘸酱。如果你想让她多摄入蛋白质，可以考虑给她准备一份鸡肉或鱼肉配米饭和蔬菜。最后，记得给她提供一杯牛奶或水来保持水分补充。希望这些建议对你有帮助！如果你还有其他问题，请随时告诉我。",
        "turn": 0,
        "timestamp": "1694144002581689000",
        "visible_to": "all",
        "msg_type": "text"
    },
    {
        "agent_name": "Ella",
        "content": "continue",
        "turn": 1,
        "timestamp": "1694144025822335000",
        "visible_to": "all",
        "msg_type": "text"
    },
    {
        "agent_name": "Crystal",
        "content": "谢谢你的建议，Nova！这些都是很不错的选择。我想问一下，有没有一些简单的健康午餐食谱，可以让我在家做并带给我的女儿？我想确保她吃的东西新鲜又健康。你有什么建议吗？",
        "turn": 2,
        "timestamp": "1694144032890695000",
        "visible_to": "all",
        "msg_type": "text"
    },
    {
        "agent_name": "Nova",
        "content": "当然，我可以给你一些简单的健康午餐食谱。以下是几个建议：\n\n1. 蔬菜寿司卷：用紫菜、糙米、鳗鱼、黄瓜和胡萝卜卷起来做成寿司卷。切成小段，这是一个方便携带的午餐选择。\n\n2. 鸡肉沙拉三明治：将烤鸡胸肉切成薄片，加入蔬菜（例如生菜、番茄和黄瓜），用全麦面包做成三明治。可以配以低脂酸奶或者蔬菜片。\n\n3. 彩虹色蔬菜饼：将胡萝卜、西兰花、洋葱和玉米混合在一起，加入蛋液和面粉，煎成小饼。这是一个快速又健康的午餐选择。\n\n4. 水果沙拉：切碎一些新鲜水果，例如苹果、草莓、蓝莓和香蕉。可以加入一些坚果或者酸奶作为调味品。\n\n这些都是简单而健康的午餐选择，你可以根据你女儿的口味和喜好进行调整。希望这些建议能给你一些灵感！如果你需要更多食谱或有其他问题，请随时告诉我。",
        "turn": 2,
        "timestamp": "1694144054290083000",
        "visible_to": "all",
        "msg_type": "text"
    },
    {
        "agent_name": "Ella",
        "content": "continue",
        "turn": 3,
        "timestamp": "1694144059307035000",
        "visible_to": "all",
        "msg_type": "text"
    },
    {
        "agent_name": "Crystal",
        "content": "谢谢你的建议，Nova！这些食谱听起来很棒，我会尝试制作它们。我还想问一下，有没有一些健康的零食选项，我可以给我的女儿放进她的午餐盒里？我希望能给她一些有营养但又容易携带的零食。你有什么建议吗？",
        "turn": 4,
        "timestamp": "1694144062870150000",
        "visible_to": "all",
        "msg_type": "text"
    },
    {
        "agent_name": "Nova",
        "content": "当然，我可以给你一些健康的零食选项，适合你女儿的午餐盒。以下是几个建议：\n\n1. 蔬菜棒和酸奶蘸酱：将胡萝卜、黄瓜和芹菜切成条状，搭配低脂酸奶或者花生酱作为蘸酱。这是一个轻松携带的营养零食。\n\n2. 水果串：将各种水果（例如葡萄、蓝莓、草莓等）穿在竹签上，制作成水果串。这是一个方便吃又有趣的零食选择。\n\n3. 坚果和干果混合：将坚果（例如杏仁、核桃）和干果（例如葡萄干、无花果）混合在一起，制作成小袋子。这是一个高能量和营养丰富的零食。\n\n4. 酸奶杯：将低脂酸奶加入一些新鲜水果和谷物（例如燕麦片、坚果），放入一个密封杯中。这是一个轻松携带的健康零食。\n\n记住，在选择零食时，尽量选择低糖和低盐的选项，并注意控制食用量。希望这些建议能给你一些灵感！如果你还有其他问题，请随时告诉我。",
        "turn": 4,
        "timestamp": "1694144078840960000",
        "visible_to": "all",
        "msg_type": "text"
    },
    {
        "agent_name": "Ella",
        "content": "<<<<<<END_OF_CONVERSATION>>>>>>",
        "turn": 5,
        "timestamp": "1694144091154109000",
        "visible_to": "all",
        "msg_type": "text"
    }
]
```

Feel free to create a new example and try to experiment with it further! Have fun ㋡ ~
