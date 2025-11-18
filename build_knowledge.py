"""
构建知识库工具

本程序的主要功能是从用户提供的文档目录中加载文档，利用 DeepSeek API 和 HuggingFace 嵌入模型生成知识库，
并将生成的知识库存储在指定的目录中。知识库可以用于后续的问答系统。

主要功能：
1. 从用户指定的目录加载文档。
2. 使用 DeepSeek API 对文档进行分块处理，生成父块和子块。
3. 使用 HuggingFace 嵌入模型对文档内容进行向量化。
4. 将生成的向量存储在 FAISS 向量数据库中。
5. 将知识库文件保存到指定的目录中。

使用说明：
1. 用户需要提供源文档所在的目录。
2. 用户需要输入知识库的名称（仅限英文、数字、下划线组合）。
3. 程序会自动创建 "knowledge_bases" 目录，并在其中创建以知识库名称命名的子目录，用于存储生成的知识库文件。
4. 程序会从 .env 文件中读取 DeepSeek API Key。
5. 如果文档内容存在敏感信息，程序会提示用户检查并修改文档内容。

注意事项：
- 确保 .env 文件中包含有效的 DeepSeek API Key。
- 确保源文档目录存在且包含支持的文档格式。
- 如果遇到 "Content Exists Risk" 错误，请检查文档内容是否包含敏感信息。

依赖：
- Python 环境
- DeepSeek API
- HuggingFace 嵌入模型
- FAISS 向量数据库
- dotenv 用于加载环境变量

作者：<Winger Liu>
日期：<2025-11-18>
"""

import os
import time
import pickle
import re
from dotenv import load_dotenv  # 用于加载 .env 文件
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_deepseek import ChatDeepSeek
from rag_parent_new_01 import load_documents_from_directory, llm_whole_doc_chunking, generate_child_chunks

# 加载 .env 文件
load_dotenv()

def build_knowledge_base(directory, knowledge_base_path, deepseek_api_key, Who_Are_You="全面专家"):
    try:
        # 创建知识库目录
        os.makedirs(knowledge_base_path, exist_ok=True)
        vector_store_path = os.path.join(knowledge_base_path, "faiss_index")
        parent_map_path = os.path.join(knowledge_base_path, "parent_map.pkl")

        print("##############")
        print("构建知识库(只执行一次)")
        print("初始化 DeepSeek LLM")
        
        # 初始化 DeepSeek LLM
        llm = ChatDeepSeek(
            model="deepseek-chat",
            api_key=deepseek_api_key,
            api_base="https://api.deepseek.com/v1",
        )

        # 加载文档
        print("加载文档...")
        documents = load_documents_from_directory(directory)

        # LLM 生成父块
        try:
            parent_chunks = llm_whole_doc_chunking(documents, llm, max_input_length=500, Who_Are_You=Who_Are_You)
        except Exception as llm_error:
            if "Content Exists Risk" in str(llm_error):
                print("错误：文档内容可能包含敏感信息，DeepSeek API 拒绝处理。")
                print("建议：")
                print("1. 检查文档内容，确保不含敏感信息")
                print("2. 修改文档内容，删除可能的敏感部分")
                print("3. 如果确认内容无误，可以尝试：")
                print("   - 减小 max_input_length 的值")
                print("   - 将文档拆分成更小的部分")
                return
            else:
                raise llm_error

        # 生成子块
        child_chunks, parent_map = generate_child_chunks(parent_chunks)
        
        # 初始化嵌入模型
        print("初始化嵌入模型 ./BAAI/beg-m3")
        model_name = "../models/BAAI/bge-m3"
        model_kwargs = {"device": "cuda:0"}  # 使用 GPU
        encode_kwargs = {"normalize_embeddings": True}
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        # 初始化向量存储和文档存储
        print("初始化向量存储和文档存储")
        vector_store = FAISS.from_documents(child_chunks, embedding_model)
        # 保存知识库
        vector_store.save_local(vector_store_path)

        # 保存父块映射
        with open(parent_map_path, "wb") as f:
            pickle.dump(parent_map, f)

        print(f"知识库已保存至: {knowledge_base_path}")
    except Exception as e:
        print(f"构建知识库时发生错误:")
        if "Content Exists Risk" in str(e):
            print("• 错误类型：内容风险")
            print("• 原因：文档内容可能包含敏感信息")
            print("• 建议：检查并修改文档内容，确保不含敏感信息")
        else:
            print(f"• 错误信息：{str(e)}")
            print("• 建议：检查以下可能的问题：")
            print("  1. API Key 是否正确")
            print("  2. 网络连接是否正常")
            print("  3. 文档格式是否支持")
            print("  4. 磁盘空间是否充足")

if __name__ == "__main__":
    while True:
        print("\n========================================================\n")
        print("欢迎使用知识库构建工具！")
        print("输入q或quit退出程序。")

        # 获取源文档目录
        directory = input("请输入源文档所在的目录: ").strip()
        if directory.lower() in ('q', 'quit'):
            print("退出程序...")
            break
        if not os.path.isdir(directory):
            print("错误：输入的目录不存在，请重新输入！")
            continue

        # 获取知识库名称
        knowledge_base_name = input("请输入知识库名称（仅限英文、数字、下划线组合）: ").strip()
        if knowledge_base_name.lower() in ('q', 'quit'):
            print("退出程序...")
            break
        if not re.match(r'^[a-zA-Z0-9_]+$', knowledge_base_name):
            print("错误：知识库名称只能包含英文、数字和下划线，请重新输入！")
            continue

        # 从 .env 文件中读取 DeepSeek API Key
        deepseek_api_key = os.getenv("deepseek_api_key")
        if not deepseek_api_key:
            print("错误：未在 .env 文件中找到 DeepSeek API Key，请检查 .env 文件！")
            break

        # 创建 "knowledge_bases" 目录
        base_path = os.path.join(os.getcwd(), "knowledge_bases")
        os.makedirs(base_path, exist_ok=True)

        # 知识库路径
        knowledge_base_path = os.path.join(base_path, knowledge_base_name)

        # 构建知识库
        print(f"开始构建知识库: {knowledge_base_name}")
        build_knowledge_base(directory, knowledge_base_path, deepseek_api_key, Who_Are_You="全面专家")  # 减小输入长度
        print("知识库构建完成！")
        break