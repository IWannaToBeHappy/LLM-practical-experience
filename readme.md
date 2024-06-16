# 大模型微调实践踩坑经验

## 目录
- [大模型实践踩坑经验](#大模型实践踩坑经验)
  - [目录](#目录)
  - [技术栈概览](#技术栈概览)
  - [数据准备](#数据准备)
  - [模型微调](#模型微调)
  - [模型推理](#模型推理)
  - [模型部署](#模型部署)

## 数据处理
### 数据标注
经过实验验证，对大模型进行无监督的CFT训练并不是一个好的选择，因为大模型微调的目的是挖掘模型在特定任务上的能力，并按照预期格式与人进行对话。新知识的学习完全可以通过SFT进行灌输，并同时规划大模型的预期输出。因此，将不同源的数据转化为SFT数据是一项必须的工程工作。

同样经过实验，大模型在微调后普遍出现了惜字如金的现象，其原因很可能是SFT数据中对模型的预期输出长度过短。如使用选择题形式做为数据形式，预期输出可能只有选项本身，而没有选项的描述。因此，在数据标注时，需要将预期输出塑造得尽可能详尽，以保证大模型的预期输出尽可能完整。

一般而言，500字的输出可以满足人的视觉需求。
- **书籍类数据转SFT**
    书籍类数据的灌输目的主要是让模型学习到书籍中的知识，我们应当合理假设大模型在预训练阶段已经接收过类似的知识，尤其是对同一知识的不同语言、不同表达方式、不同呈现方式的灌输。因此，我们的目的是将这些知识挖掘出来，其具体的表现形式就是多使用文本生成，少使用阅读理解。
    ```
    例：
    {
        "human": "根据金庸的风格，生成小说", 
        "assistant": "。两人相见之后，长谈王霸之道，各有所见，却互相投机之极，当真相见恨晚。\n　　两人都觉中原诸国暮气沉沉，楚国邦大而乱，东南其势兴旺，当有霸兆。于是文种辞去官位，与范蠡同往吴国。其时吴王正重用伍子胥，言听计从，国势正盛。\n　　文种和范蠡在吴国京城姑苏住了数月，见伍子胥的种种兴革措施确是才识卓越，切中时弊，令人钦佩，自己未必能胜得他过。两人一商量，以越国和吴国邻近，风俗相似，虽然地域较小，却也大可一显身手，于是来到越国。勾践接见之下，于二人议论才具颇为赏识，均拜为大夫。\n　　后来勾践不听文种、范蠡劝谏，兴兵和吴国交战，以石买为将，在钱塘江边一战大败，勾践在会稽山被围，几乎亡国殒身。勾践在危急之中用文种、范蠡之计，买通了吴王身边的奸臣太宰伯嚭，为越王陈说。吴王夫差不听伍子胥的忠谏，答允与越国讲和，将勾践带到吴国，后来又放他归国。其后勾践卧薪尝胆，决定复仇，采用了文种的灭吴九术。\n　　那九术第一是尊天地，事鬼神，神道设教，令越王有必胜之心。第二是赠送吴王大量财币，既使他习于奢侈，又去其防越之意。第三是先向吴国借粮，再以蒸过的大谷归还，吴王见谷大，发给农民当谷种，结果稻不生长，吴国大饥。第四是赠送美女西施和郑旦，让吴王迷恋美色，不理政事。第五是赠送巧匠，引诱吴王大起宫室高台，耗其财力民力。第六是贿赂吴王左右奸臣，使之败坏朝政。第七是离间吴王忠臣，终于迫得伍子胥自杀。第八是积蓄粮草，充实国家财力。第九是铸造武器，训练士卒，待机攻吴。据后人评论，其时吴国文明，越国野蛮，吴越相争，越国常不守当时中原通行之礼法规范，不少手段卑鄙恶劣，以致吴国受损。\n　　文种八术都已成功，最后的第九术却在这时遇上了重大困难。眼见吴王派来剑士八人，所显示的兵刃之利、剑术之精，实非越国武士所能匹敌。\n　　\n　　范蠡将适才比剑的情形告知了文种。文种皱眉道：“范贤弟，吴国剑士剑利术精，固是大患，而他们在群斗之时，善用孙武子遗法，更加难破难当。”范蠡道：“正是，当年孙武子辅佐吴王，统兵破楚，攻入郢都，用兵如神，天下无敌。虽齐晋大国，亦畏其锋。他兵法有言道：‘我专为一，敌分为十，是以十攻其一也，则我众而敌寡。能以众击寡者，则吾之所与战者，约矣。’吴士四人与我越士四人相斗，吴士以二人挡我三人，以二人专攻一人，以众击寡，战无不胜。”\n　　言谈之间，二人到了越王面前，只见勾践手中提着那柄其薄如纸的利剑，兀自出神。\n　　过了良久，勾践抬起头来，说道：“文大夫，当年吴国有"
    }
    ```
- **文档类数据转SFT**
    文档类数据比起书籍类数据，一大优势是天然具备一定格式。这使得我们可以诱导模型进行更加细致的专一化阐述。然而，同样我们期望模型能尽可能全面地回答问题，考虑到500字的预期输出，可以将过长的文档数据按标题、章节等信息分段并诱导输出，而将短文档以类似书籍类数据的形式进行构建。
    ```
    原始数据：
    {"语法": "size_t _aligned_msize_dbg( void *memblock, size_t alignment, size_t offset);参数memblock指向内存块的指针。alignment对齐值，必须是 2 的整数次幂。offset用于强制对齐的内存分配中的偏移量。", "返回值": "返回无符号整数形式的大小（以字节为单位）。", "备注": "alignment 和 offset 值必须与传递给分配该块的函数的值相同。_aligned_msize_dbg 是 _aligned_msize 函数的调试版本。 未定义 _DEBUG 时，每个对 _aligned_msize_dbg 的调用都简化为对 _aligned_msize 的调用。 _aligned_msize 和 _aligned_msize_dbg 都计算基堆中的内存块的大小，但 _aligned_msize_dbg 增加了一个调试功能：它在返回大小的内存块用户部分的任一侧包括缓冲区。此函数验证其参数。 如果 memblock 为空指针或 alignment 不是 2 的幂，则 _msize 会调用无效参数句柄，如参数验证中所述。 如果处理了错误，则该函数将 errno 设置为 EINVAL 并返回 -1。若要了解如何在基堆的调试版本中分配、初始化和管理内存块，请参阅 CRT 调试堆详细信息。 若要了解分配块类型及其使用方式，请参阅调试堆上的块类型。 有关标准堆函数与调试版本之间的差异的信息，请参阅堆分配函数的调试版本。", "要求": "例程必需的标头_aligned_msize_dbg<crtdbg.h>有关兼容性的详细信息，请参阅 兼容性。", "库": "仅限 C 运行时库的调试版本。", "另请参阅": "内存分配"}
    构建方式1：
        类似书籍类数据
    构建方式2：
        [
            {
                "human":"介绍一下C语言函数_aligned_msize_dbg的语法。"
                "assistant":"size_t _aligned_msize_dbg( void *memblock, size_t alignment, size_t offset);参数memblock指向内存块的指针。alignment对齐值，必须是 2 的整数次幂。offset用于强制对齐的内存分配中的偏移量。"
            },
            {
                "human":"介绍一下C语言函数_aligned_msize_dbg的返回值。"
                "assistant":"返回无符号整数形式的大小（以字节为单位）。"
            }
        ]
    ```

- **日志类数据转SFT**
    日志类数据的灌输目的主要是让模型能对日志数据进行关键信息提取，自动研判。因此，应当将日志类数据构建为摘要提取及总结类型数据。***然而日志的摘要信息和类型总结难以自动生成***，此类信息尚未有很好的自动化处理方式。
- **代码类数据转SFT**
    代码类数据的灌输目的是代码生成与代码解析。这类任务往往有非常明确的预期输出格式，我们可以人为构建**trigger**以在推理时定向触发模型的格式化输出。
    ```
    原始数据：
    {"code":"int main(int argc, char *argv[]) {\n    printf(\"Hello, world!\");\n    return 0;\n}"}
    trigger构建：
        [
            {
                "human":"trigger:将以下代码翻译为汇编代码。\nint main(int argc, char *argv[]) {\n    printf(\"Hello, world!\");\n    return 0;\n}",
                "assistant":"mov rdi, 0x12345678  \ncall printf\nmov eax, 0\nret "
            },
            {
                "human":"trigger:功能解析C语言代码。\nint main(int argc, char *argv[]) {\n    printf(\"Hello, world!\");\n    return 0;\n}",
                "assistant":"打印出 "Hello, world!" 到控制台。"
            }
        ]
    ```
#### self-instruct
#### evolve instruction
### 数据清洗
如何筛选高质量数据是一个难题.llama2的训练只用到了27,540条数据，论文直接表示"Quality is ALL You Need"。
大模型的数据准备正走向两个分支，数量取胜的数据通常以百万起步，而质量为王的数据则不超过十万条。
[data-juicer](https://github.com/modelscope/data-juicer)规定了一些数据清洗规则，在项目中被命名为算子，清洗依据可以分为两种，一种是基于文本特征进行清洗，如异常的字符比例，异常的重复字符，另一种是基于文本质量分类器（如GPT-3实践）对数据集进行质量评估。

· **异常**的定义——3σ定律
    ![alt text](resource/3sigma.png)

data-juicer中出现的算子被枚举在[config_all.yaml](https://github.com/modelscope/data-juicer/blob/main/configs/config_all.yaml)。以下展示部分与文本相关的算子。更多详细内容可见[data-juicer文档](https://github.com/modelscope/data-juicer/blob/main/docs)。
```
Process:
  - clean_email_mapper:
  - clean_links_mapper:
  - fix_unicode_mapper:
  - punctuation_normalization_mapper:
  - whitespace_normalization_mapper:
  - clean_copyright_mapper:

  - alphanumeric_filter:
      tokenization: False
      min_ratio: 0.4
      max_ratio: 0.8
  - alphanumeric_filter:
      tokenization: True
      min_ratio: 1.5
      max_ratio: 3
  - average_line_length_filter:
      min_len: 15
      max_len: 100
  - character_repetition_filter:
      rep_len: 10
      min_ratio: 0.05
      max_ratio: 0.3
  - maximum_line_length_filter:
      min_len: 50
      max_len: 500
  - text_length_filter:
      min_len: 300
  - words_num_filter:
      lang: en
      tokenization: False
      min_num: 30
      max_num: 5000
  - word_repetition_filter:
      lang: en
      tokenization: False
      rep_len: 10
      max_ratio: 0.1
  - document_simhash_deduplicator:
      tokenization: space
      window_size: 6
      lowercase: true
      ignore_pattern: '\p{P}'
      num_blocks: 6
      hamming_distance: 4
```

另一些筛选原则基于大模型训练效果，这里做出记录，尚未进行实验验证。
#### IFD
IFD（Instruction-Following Difficulty）指令跟随难度，是一种量化每个样本对模型难度的筛选方法，通过评估prompt优化前后模型对问题的回答准确度来量化问题难度。若IFD分高，说明问题本身容易被模型学习理解，若IFD分数低，则说明问题对模型处于困难边界。
#### Super filtering
以小参数模型类比大参数模型进行筛选，筛选原理类似于前面提到的文本质量分类器。
#### MoDS
通过打分选择高质量数据集，聚类筛选种子数据集，使用种子数据集进行训练初始化LLM，使用拓展高质量数据集进一步进行训练。


![alt text](resource/MoDS.png)
### 数据增强
TODO
### 数据集灌输
构建的数据集不能直接简单地用来训练模型，必须夹杂一定量的通识数据以使模型具备正常的对话能力，并避免模型退化。
SFT数据占所有数据的比例建议为 **20%~40%**
这里给出一些通识数据集库，供参考使用。
[BAAI智源](https://data.baai.ac.cn/data)
[hugging face](https://huggingface.co/)
### 对齐数据准备
TODO

## 模型微调
### 模型选择
可以根据各大排行榜进行模型选择
[hugging face通识能力排行](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
[SecBench安全能力排行](https://secbench.org/board)
[代码生成能力排行](https://www.datalearner.com/ai-models/leaderboard/datalearner-llm-coding-leaderboard?modelSize=7b)
[中文能力排行](https://huggingface.co/spaces/BAAI/open_cn_llm_leaderboard)

**不建议使用llama3模型**，对中文的支持度较低。
正在尝试qwen2-7B

### 模型量化
直接使用全量模型进行推理与训练具备较大的显存负担，例如qwen2-7B模型的推理峰值显存占用为25G，量化技术通过将模型的浮点数类型参数（bf16、fp16）转化为定点数（INT4、INT8）来降低显存负担。以下对各量化技术做简单排列，具体原理解析可参考量化原理文档。
#### bnb量化
bnb量化全称BitsAndBytes，原理为浮点数到定点数的直接映射。量化过程无需数据集辅助，量化后**模型大小变小，推理及训练速度提升**。
```bash
pip install bitsandbytes
```
**缺陷**： vllm不支持BNB量化，因此若采用vllm进行推理加速，不可使用bnb对模型进行量化
#### gptq量化
gptq量化脱胎自模型剪枝算法，原理是尽可能缩小量化前后对模型loss的影响。因此存在loss计算过程，量化过程**需要数据集辅助**，量化后**模型大小变小。推理及训练速度显著下降**。
```bash
pip install auto_gptq
```
**缺陷**：因为GPTQ量化依赖于loss计算，且时间复杂度为O(N^3)，量化微调时间将显著增大。
#### awq量化
awq量化修改了GPTQ对权重的划分算法，在算法复杂度上与gptq一致，因此同样**需要数据集辅助**，量化后**模型大小变小。推理及训练速度显著下降**。
```bash
pip install autoawq
```
**缺陷1**：因为GPTQ量化依赖于loss计算，且时间复杂度为O(N^3)，量化微调时间将显著增大。
**缺陷2**：autoawq库量化要求GPU算力达到7.5，V100无法满足该算力等级。
#### hqq量化
hqq量化优化了bnb量化中零点与缩放倍数固定的问题，以量化和逆量化操作后的权重差异作为损失函数，以零点和缩放倍数作为参数进行训练，量化过程无需训练集辅助，量化后性能**本人尚未评估**，算力要求**尚未实践**。
```bash
pip install hqq
```

### 分布式训练
常用的分布式训练框架包括Deepspeed和colossalai
#### Deepspeed
Deepspeed是一个深度学习框架，支持多种分布式训练策略，如DDP、ZeRO等。
```bash
pip install deepspeed
```
#### colossalai
TODO 官方宣传其效率优于deepspeed。

### 模型微调

### 模型推理

## 模型对齐

## prompt优化

## 模型部署

思考模型 思考过程训练（Agent），方式：读说明书，学习工具使用
代码智能体：大模型生成思考，写代码，提交给代码生成、代码执行、返回代码结果给大模型，进行下一步迭代。