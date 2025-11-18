```markdown
# RAG（检索增强生成）知识库构建与问答系统

## 项目简介
本项目是一个基于 RAG（Retrieval-Augmented Generation）框架的知识库构建与问答系统。通过结合文档加载、语义分块、向量化存储和检索增强生成等技术，能够实现基于文档内容的智能问答。支持多领域知识库的构建与切换，适用于智能问答、信息检索等场景。

---

## 功能特性
### 1. 知识库构建
- 从用户指定的文档目录加载文档，支持多种文档格式（如 PDF、DOCX、TXT、CSV 等）。
- 使用 DeepSeek API 对文档进行语义分块，生成父块和子块。
- 使用 HuggingFace 嵌入模型对文档内容进行向量化。
- 将生成的向量存储在 FAISS 向量数据库中，并建立父子块的映射关系。

### 2. 知识库加载与问答
- 支持用户选择已有知识库或重新构建知识库。
- 加载知识库后，用户可以输入问题，系统基于知识库内容生成回答。
- 支持多轮对话，保留上下文历史，提升问答的连贯性。

### 3. 多领域支持
- 用户可选择不同领域的知识库（如汽车维修、人工智能、智慧交通等）。
- 支持动态注入领域角色（如“汽车维修专家”），增强回答的专业性。

### 4. 错误处理与优化
- 针对文档加载、API 调用等环节提供详细的错误提示。
- 支持流式输出，优化用户体验。
- 提供参考文档的标注与去重，确保回答的可追溯性。

---

## 环境依赖
- **Python** >= 3.8
- **依赖库**：
  - `python-dotenv`
  - `langchain-huggingface`
  - `langchain-community`
  - `langchain-deepseek`
  - `transformers`
  - `faiss-cpu` 或 `faiss-gpu`
  - `torch`
  - 其他依赖详见 `requirements.txt` 或 `environment.yml`

---

## 快速开始

### 1. 克隆项目
```bash
git clone <项目地址>
cd 00_rag_parent_new
```

### 2. 安装依赖
#### 使用 `requirements.txt` 安装
```bash
pip install -r requirements.txt
```

#### 使用 Conda 环境安装
```bash
conda env create -f environment.yml
conda activate rag_test
```

### 3. 配置 `.env` 文件
在项目根目录下创建 `.env` 文件，并添加以下内容：
```plaintext
deepseek_api_key=your_deepseek_api_key
```
将 `your_deepseek_api_key` 替换为有效的 DeepSeek API Key。

### 4. 构建知识库
运行以下命令，构建知识库：
```bash
python build_knowledge.py
```
根据提示输入文档目录和知识库名称。

### 5. 问答系统
运行以下命令，加载知识库并开始问答：
```bash
python query.py
```

---

## 文件结构
```plaintext
00_rag_parent_new/
├── build_knowledge.py       # 知识库构建脚本
├── query.py                 # 问答系统脚本
├── rag_parent_new_01.py     # 核心功能实现
├── requirements.txt         # Python 依赖文件
├── environment.yml          # Conda 环境配置文件
├── README.md                # 项目说明文档
└── knowledge_bases/         # 知识库存储目录
```

---

## 注意事项
1. 确保 `.env` 文件中包含有效的 DeepSeek API Key。
2. 确保源文档目录存在且包含支持的文档格式。
3. 如果遇到 "Content Exists Risk" 错误，请检查文档内容是否包含敏感信息。
4. 如果使用 GPU，请安装 `faiss-gpu` 和支持 CUDA 的 `torch`。

---

## 贡献
欢迎提交 Issue 和 Pull Request 来改进本项目。

---

## 作者
**Winger Liu**

---

## 许可证
本项目遵循 MIT 许可证。
```

---

### 说明：
1. **项目简介**：简要介绍项目的功能和用途。
2. **功能特性**：列出项目的主要功能模块。
3. **环境依赖**：列出运行项目所需的依赖库和 Python 版本。
4. **快速开始**：提供从安装到运行的完整流程。
5. **文件结构**：展示项目的主要文件和目录结构。
6. **注意事项**：列出运行项目时需要注意的关键点。
7. **贡献**：鼓励用户参与项目开发。
8. **许可证**：说明项目的开源协议。

将上述内容保存到 README.md 文件中即可。
