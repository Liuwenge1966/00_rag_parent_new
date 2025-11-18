
### 程序运行所需的库：
1. **Python 标准库**：
   - `os`
   - `time`
   - `pickle`
   - `re`

2. **第三方库**：
   - `dotenv`：用于加载 .env 文件中的环境变量。
   - `langchain_huggingface`：用于 HuggingFace 嵌入模型。
   - `langchain_community`：用于 FAISS 向量存储。
   - `langchain_deepseek`：用于 DeepSeek API 的调用。
   - `transformers`：用于加载和使用 HuggingFace 模型。
   - `faiss-cpu` 或 `faiss-gpu`：用于向量检索。
   - `tqdm`：用于显示进度条（如果有）。
   - `pandas`：用于数据处理（如果有需要）。
   - `torch`：用于深度学习模型的运行。
   - `numpy`：用于数值计算。

---

### `requirements.txt` 内容：
```plaintext
python-dotenv==1.0.0
langchain-huggingface==0.0.1
langchain-community==0.0.1
langchain-deepseek==0.0.1
transformers==4.33.0
faiss-cpu==1.7.4  # 如果使用 CPU
# faiss-gpu==1.7.4  # 如果使用 GPU，请替换 faiss-cpu
tqdm==4.66.1
pandas==2.1.1
torch==2.0.1
numpy==1.26.0
```

---

### 说明：
1. **`faiss-cpu` 和 `faiss-gpu`**：
   - 如果你的环境使用 CPU，请安装 `faiss-cpu`。
   - 如果你的环境使用 GPU，请安装 `faiss-gpu`。

2. **版本号**：
   - 版本号是根据常见的最新稳定版本指定的，你可以根据需要调整。

3. **安装命令**：
   - 将上述内容保存为 `requirements.txt` 文件。
   - 使用以下命令安装依赖：
     ```bash
     pip install -r requirements.txt
     ```

4. **环境检查**：
   - 确保你的 Python 版本为 3.8 或更高版本，以支持所有依赖库。