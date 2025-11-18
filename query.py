import os
import time
from dotenv import load_dotenv  # 用于加载 .env 文件
from langchain_huggingface import HuggingFaceEmbeddings
from rag_parent_new_01 import load_rag_pipeline, test_rag

# 加载 .env 文件
load_dotenv()

if __name__ == "__main__":
    while True:
        print("\n========================================================\n")
        print("欢迎使用知识库问答系统！")
        print("输入q或quit退出程序。")

        # 列出 knowledge_bases 子目录
        base_path = os.path.join(os.getcwd(), "knowledge_bases")
        if not os.path.exists(base_path):
            print("错误：未找到 'knowledge_bases' 目录，请先构建知识库！")
            break

        subdirectories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        if not subdirectories:
            print("错误：'knowledge_bases' 目录下没有可用的知识库，请先构建知识库！")
            break

        print("可用的知识库列表：")
        for idx, subdir in enumerate(subdirectories, start=1):
            print(f"{idx} - {subdir}")

        # 用户选择知识库
        selected_index = input("请输入知识库对应的序号: ").strip()
        if selected_index.lower() in ('q', 'quit'):
            print("退出程序...")
            break
        if not selected_index.isdigit() or int(selected_index) < 1 or int(selected_index) > len(subdirectories):
            print("错误：输入的序号无效，请重新输入！")
            continue

        selected_knowledge_base = subdirectories[int(selected_index) - 1]
        vector_store_path = os.path.join(base_path, selected_knowledge_base, "faiss_index")
        parent_map_path = os.path.join(base_path, selected_knowledge_base, "parent_map.pkl")

        # 询问用户希望 AI 扮演的角色
        Who_Are_You = input("希望AI充当什么角色(如：信息化专家): ").strip()
        if Who_Are_You.lower() in ('q', 'quit'):
            print("退出程序...")
            break
        if not Who_Are_You:
            print("错误：角色不能为空，请重新输入！")
            continue

        # 加载 RAG 管道
        print(f"加载知识库 '{selected_knowledge_base}' 中的 RAG 管道...")
        deepseek_api_key = os.getenv("deepseek_api_key")
        if not deepseek_api_key:
            print("错误：未在 .env 文件中找到 DeepSeek API Key，请检查 .env 文件！")
            break

        try:
            rag_pipeline = load_rag_pipeline(vector_store_path, deepseek_api_key, parent_map_path, Who_Are_You)
            print("RAG 管道加载完成！")
        except Exception as e:
            print(f"加载 RAG 管道时发生错误: {e}")
            break

        # 多轮对话历史
        history = []
        while True:
            print("\n========================================================\n")
            print(f"我是{Who_Are_You}, 请输入你的问题 (q退出, help帮助, 回车重复上一个问题): ")
            query = input(": ").strip()
            if query.lower() in ('q', 'quit', 'exit'):
                print("退出问答系统...")
                break
            if query.lower() == 'help':
                print("直接输入你的问题，支持多轮上下文。输入q/quit/exit退出，回车重复上一个问题。")
                continue
            if query == '' and history:
                query = history[-1][0]  # 上一个问题
                print(f"重复上一个问题: {query}")
            elif query == '':
                print("请输入问题。")
                continue

            print("正在努力思考中...")
            start_time = time.time()
            # 多轮对话支持：将历史传递
            test_rag(rag_pipeline, query, history)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"运行时间: {elapsed_time:.2f} 秒")
            # 记录历史
            history.append((query, "[AI回答见上文]"))
            if len(history) > 10:
                history.pop(0)  # 删除最旧的记录