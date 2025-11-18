import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, SpacyTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredImageLoader, UnstructuredFileLoader, UnstructuredMarkdownLoader, UnstructuredPDFLoader
from langchain.schema import Document
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
import markdown
from langchain_deepseek import ChatDeepSeek
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from loader import RapidOCRPDFLoader, RapidOCRDocLoader
# from self_loader import load_document_with_ocr
# from text_splitter import AliTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore, LocalFileStore
import pickle
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import re
import time
from langchain_core.runnables import RunnableSequence

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyMuPDFLoader
# import office
# import comtypes.client
# import win32com.client

# def convert_doc_to_docx(input_doc_path):
#     # word = comtypes.client.CreateObject('Word.Application')
#     word = win32com.client.Dispatch("Word.Application")
#     word.Visible = False
#     doc = word.Documents.Open(input_doc_path)
#     output_docx_path = input_doc_path.replace('.doc', '.docx')  # 生成输出路径
#     doc.SaveAs(output_docx_path, FileFormat=16)  # 16 表示 docx 格式
#     doc.Close()
#     word.Quit()
#     # os.remove(input_doc_path)
#     # print(f"文件 {input_doc_path} 删除成功！")
#     return output_docx_path  # 返回新生成的 .docx 文件路径

import subprocess
def convert_doc_to_docx(doc_path):
    output_docx_path = doc_path.replace('.doc', '.docx')

    command = ['libreoffice', '--headless', '--convert-to', 'docx', doc_path, '--outdir', os.path.dirname(output_docx_path)]

    subprocess.run(command, check=True)

    return output_docx_path

# 1. 定义文档加载函数
def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        # loader = PyPDFLoader(file_path)
        # return loader.load()
        loader = RapidOCRPDFLoader(file_path)
        return loader.load()
        # loader = PyMuPDFLoader(file_path)
        # data = loader.load()
        # print(data[0])
        # return data
        # text = loader._get_elements()
        # return [Document(page_content=text, metadata={"source": file_path})]
    elif ext == ".docx" or ext == ".doc":
        if ext == ".doc":
            # office.word.doc2docx(file_path, file_path+'x')
            # os.remove(file_path)
            # print(f"文件 {file_path} 删除成功！")
            # file_path = file_path+'x'
            file_path_new = convert_doc_to_docx(file_path)
            os.remove(file_path)
            print(f"文件 {file_path} 删除成功！")
            file_path = file_path_new
        # return loader.load()
        loader = RapidOCRDocLoader(file_path)
        return loader.load()
        # text = loader._get_elements()
        # return [Document(page_content=text, metadata={"source": file_path})]
    elif ext == ".txt":
        loader = TextLoader(file_path)
        return loader.load()
    elif ext == ".md":
        with open(file_path, "r", encoding="utf-8") as f:
            md_content = f.read()
            html = markdown.markdown(md_content)
            return [Document(page_content=html, metadata={"source": file_path})]
    else:
        raise ValueError(f"不支持的文件类型: {ext}")


# 2. 从文件夹加载所有文档
def load_documents_from_directory(directory):
    documents = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            # 构造完整路径
            file_path = os.path.join(root, file_name)
            print(file_path)
            try:
                docs = load_document(file_path)
                # print(docs)
                # docs = load_document_with_ocr(file_path)
                documents.extend(docs)
                print(f"已加载: {file_name}")
            except Exception as e:
                print(f"加载 {file_name} 失败: {e}")
    # for file_name in os.listdir(directory):
    #     file_path = os.path.join(directory, file_name)
    #     try:
    #         docs = load_document(file_path)
    #         print(docs)
    #         # docs = load_document_with_ocr(file_path)
    #         documents.extend(docs)
    #         print(f"已加载: {file_name}")
    #     except Exception as e:
    #         print(f"加载 {file_name} 失败: {e}")
    #     # print(documents)
    return documents


# # 使用 DeepSeek API 进行 LLM 分段
# def llm_parent_chunking(documents, llm, max_chunk_size=300):
#     # 定义分块 Prompt
#     chunk_prompt_template = """
#             你是一个专业的政策文件段落分割助手，用于后续RAG检索，判断以下文本是否为一个新的段落（回答“是”或“否”或“可能”）：
#                 当前块末尾: {current_chunk}
#                 新句子: {new_sentence}
#             """
#     chunk_prompt = PromptTemplate(
#         template=chunk_prompt_template,
#         input_variables=["current_chunk", "new_sentence"]
#     )
#     chunk_chain = LLMChain(llm=llm, prompt=chunk_prompt)
#
#     parent_chunks = []
#     for doc in documents:
#         text = doc.page_content
#         # sentences = text.split("。")  # 初始按句子分割
#         sentences = text.split("。")  # 初始按句子分割
#         current_chunk = ""
#
#         for sentence in sentences:
#             # print(sentence)
#             # print(sentence)
#             # print(sentence.strip())
#             if not sentence.strip():
#                 continue
#             if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
#                 # 使用 DeepSeek API 判断语义边界
#                 response = chunk_chain.run(
#                     current_chunk=current_chunk,
#                     new_sentence=sentence
#                 ).strip()
#                 # print(current_chunk)
#                 # print(sentence)
#                 print(response)
#
#                 if "是" in response:
#                     parent_chunks.append(Document(page_content=current_chunk.strip(), metadata=doc.metadata))
#                     current_chunk = sentence
#                 else:
#                     current_chunk += "。" + sentence
#             else:
#                 current_chunk += "。" + sentence
#
#         if current_chunk:
#             parent_chunks.append(Document(page_content=current_chunk.strip(), metadata=doc.metadata))
#
#     return parent_chunks


# LLM 整篇分段
def llm_whole_doc_chunking(documents, llm, max_input_length=2000, overlap=500):
    # 定义分段 Prompt
    chunk_prompt = PromptTemplate(
        input_variables=["text"],
        template="""你是一个政策文件分段专家，请分析以下输入文本并给出语义分段点（分为1段或多段）用于后续RAG检索，上下文相关的不要分开，每段至少1000字以上（可以更长不能更短），你认为意思连贯且不算很长的文档可以直接分成一段。对于每个分段，提供一个主题描述该段的主要内容及该段在文档中的意义，主题必须从原文提取凝练或完全符合原文意思，不得随意生成。若你认为无需分段，则输出“无需分段”。
            输入文本： {text}
            输出格式（每两行一个分段点和主题，一般来说最后一个分段点应该是段落最后完整的一句话）：
            - 段落1
            - 分段点： [分段内的最末尾完整的一句话，一定要完全符合原文，保留所有的字符]
            - 主题： [段落的主题，说明段落包含了什么内容，要求简练且全面（50字以内），完全尊重原文意思，包含上下文和标题的意思，若无法确定主题，则输出“无”]
            - 段落2
            - 分段点： [分段内的最末尾完整的一句话，一定要完全符合原文，保留所有的字符]
            - 主题： [段落的主题，说明段落包含了什么内容，要求简练且全面（50字以内），完全尊重原文意思，包含上下文和标题的意思，若无法确定主题，则输出“无”]
            ...
            
            """
    )
    # 版本太老，弃用
    chunk_chain = LLMChain(llm=llm, prompt=chunk_prompt)

    # 显式声明为 RunnableSequence
    # from langchain_core.runnables import RunnableSequence
    # chunk_chain = RunnableSequence(prompt=chunk_prompt, llm=llm)

    parent_chunks = []
    for doc in documents:
        print(doc.metadata["source"])
        text = doc.page_content

        # 如果文本超长，分片处理
        if len(text) > max_input_length:
            sub_texts = [text[i:i + max_input_length] for i in range(0, len(text), max_input_length)]
        else:
            sub_texts = [text]

        # print(len(text))

        # # 如果文本超长，分片并添加重叠
        # if len(text) > max_input_length:
        #     sub_texts = []
        #     for i in range(0, len(text), max_input_length - overlap):
        #         start = i
        #         end = min(i + max_input_length, len(text))
        #         print("分段结果：")
        #         print(start)
        #         print(end)
        #         print(text[start:end])
        #         sub_texts.append(text[start:end])
        # else:
        #     sub_texts = [text]

        left = None

        for text_i in range(len(sub_texts)):
        # for sub_text in sub_texts:
            if left is None:
                sub_text = re.sub(r'\s+', '', sub_texts[text_i])
            else:
                sub_text = re.sub(r'\s+', '', left + sub_texts[text_i])
                left = None

            print("分段：")
            print(sub_text)

            response = chunk_chain.run(text=sub_text).strip()
            # response = chunk_chain.invoke({"text": sub_text}).strip()

            print(response)

            # if "无需分段" in response:
            #     parent_chunks.append(Document(page_content=sub_text, metadata=doc.metadata))
            #     continue

            # 解析分段点
            split_points = []
            topics = []
            lines = response.split("\n")
            i = 0
            while i < len(lines):
                if "分段点：" in lines[i]:
                    point = lines[i].split("分段点：")[1].strip("[]")
                    print(point)
                    if (i+1) < len(lines):
                        if "主题：" in lines[i + 1]:
                            topic = lines[i + 1].split("主题：")[1].strip("[]")
                            # print(topic)
                            i += 2
                    else:
                        topic = "无"
                        i += 1
                    if point and topic:
                        try:
                            point = re.sub(r'\s+', '', point)
                            idx = sub_text.index(point)
                            split_points.append((idx + len(point), topic))
                            # split_points.append(idx + len(point))
                            # topics.append(topic)
                        except ValueError:
                            pass
                else:
                    i += 1

            # 根据分段点切分
            split_points = sorted(set(split_points), key=lambda x: x[0])  # 去重并排序
            print(split_points)
            last_idx = 0
            save_text = None
            for i in range(len(split_points)):
                # idx = split_points[i]
                # topic = topics[i]
                idx = split_points[i][0]
                topic = split_points[i][1]
                if idx > last_idx:
                    if idx >= len(sub_text) and text_i != len(sub_texts) - 1:
                        print("放到下一段")
                        left = f"[主题：{topic}]\n{sub_text[last_idx:idx].strip()}"
                        print(left)
                        last_idx = idx
                        continue
                    source = doc.metadata["source"]
                    content_with_topic = f"[文档路径：{source}]\n[主题：{topic}]\n{sub_text[last_idx:idx].strip()}"
                    print(content_with_topic)
                    parent_chunks.append(Document(
                        page_content=content_with_topic,
                        metadata=doc.metadata
                    ))
                    last_idx = idx
            if last_idx < len(sub_text):
                print("没被检测出来")
                if text_i != len(sub_texts) - 1:
                    print("放到下一段")
                    left = sub_text[last_idx:].strip()
                    print(left)
                else:
                    source = doc.metadata["source"]
                    content_with_topic = f"[文档路径：{source}]\n{sub_text[last_idx:].strip()}"
                    print(content_with_topic)
                    parent_chunks.append(Document(
                        page_content=content_with_topic,
                        metadata=doc.metadata
                    ))

    return parent_chunks


# 生成子块并关联父块
def generate_child_chunks(parent_chunks):
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    child_chunks = []
    parent_map = {}  # 存储父块内容用于检索

    for i, parent in enumerate(parent_chunks):
        # print(i)
        # print(parent.page_content)
        parent_id = f"parent_{i}"
        parent_map[parent_id] = parent.page_content

        # 提取父块的主题（从开头到第一个换行符）
        parent_content = parent.page_content
        path_match = re.match(r"(\[文档路径：.*?\])\n", parent_content)
        path_prefix = path_match.group(1) + "\n" if path_match else ""
        topic_match = re.match(r"(\[主题：.*?\])\n", parent_content)
        topic_prefix = topic_match.group(1) + "\n" if topic_match else ""
        # print(topic_prefix)
        # print(path_prefix)

        child_texts = child_splitter.split_text(parent.page_content)

        for child_text in child_texts:
            # 如果子块已包含主题（因分割位置），避免重复添加
            if not child_text.startswith("[文档路径："):
                child_text = f"{path_prefix}{topic_prefix}{child_text}"

            child_chunks.append(Document(
                page_content=child_text,
                metadata={**parent.metadata, "parent_id": parent_id}
            ))

    return child_chunks, parent_map


# 构建知识库（只执行一次）
def build_knowledge_base(directory, deepseek_api_key, vector_store_path="faiss_index", parent_map_path="parent_map.pkl"):
    # 初始化 DeepSeek LLM
    llm = ChatDeepSeek(
        model="deepseek-chat",
        api_key=deepseek_api_key,
        api_base="https://api.deepseek.com/v1",
    )

    # 加载文档
    print("加载文档...")
    documents = load_documents_from_directory(directory)

    # for doc in documents:
    #     text = doc.page_content
    #     print(len(text))
    # exit(0)

    # LLM 生成父块
    # parent_chunks = llm_parent_chunking(documents, llm)
    parent_chunks = llm_whole_doc_chunking(documents, llm, max_input_length=4000)
    # exit(0)
    # print(parent_chunks)
    child_chunks, parent_map = generate_child_chunks(parent_chunks)
    # print(child_chunks)
    # print(parent_map)
    # for i, child_chunk in enumerate(child_chunks):
    #     print(i)
    #     print(child_chunk.page_content)
    #     print(child_chunk.metadata)

    # 初始化嵌入模型
    model_name = "./bge-m3"
    model_kwargs = {"device": "cuda:1"}  # 可以设置为 "cuda" 以利用GPU
    encode_kwargs = {"normalize_embeddings": True}
    # embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        # cache_folder='./bge-m3',
        encode_kwargs=encode_kwargs)


    # 初始化向量存储和文档存储
    vector_store = FAISS.from_documents(child_chunks, embedding_model)
    # 保存知识库
    vector_store.save_local(vector_store_path)

    # 保存父块映射
    with open(parent_map_path, "wb") as f:
        pickle.dump(parent_map, f)

    print(f"知识库已保存至: {vector_store_path} 和 {parent_map_path}")
    return


class CustomRetriever(BaseRetriever):
    vector_store: FAISS
    parent_map: dict
    child_k: int
    max_parent_k: int
    # def __init__(self, vector_store, parent_map, k):
    #     self.vector_store = vector_store
    #     self.parent_map = parent_map
    #     self.k = 5

    # def invoke(self, query):
    #     return self.custom_retriever(query)

    # 自定义检索器：返回父文档
    def _get_relevant_documents(self, query: str) -> list[Document]:
        vector_store = self.vector_store
        parent_map = self.parent_map
        k = self.child_k
        child_docs = vector_store.similarity_search(query, k=k)
        parent_docs = []
        seen_parents = set()

        parent_num = 0
        for child in child_docs:
            parent_id = child.metadata.get("parent_id")
            if parent_id and parent_id not in seen_parents:
                parent_num += 1
                if parent_num > self.max_parent_k:
                    break
                parent_content = parent_map.get(parent_id, "")
                parent_docs.append(Document(
                    page_content=parent_content,
                    metadata=child.metadata
                ))
                seen_parents.add(parent_id)

        return parent_docs


# 后处理：基于回答内容修正标号和去重
def post_process_answer(answer_text):
    # 分离正文和参考文献
    body_match = re.search(r"(.*?)- 参考文档:(.*)", answer_text, re.DOTALL)
    if not body_match:
        return answer_text  # 未找到参考文献，直接返回原文本

    body = body_match.group(1).strip()
    ref_text = body_match.group(2).strip()

    # 提取正文中的标号
    body_refs = re.findall(r"\[(\d+)\]", body)
    if not body_refs:
        return f"{body}\n- 参考文档: 无"  # 无标号时，清空参考文献

    # 提取参考文献列表
    ref_lines = [line.strip() for line in ref_text.split("\n") if line.strip()]
    ref_docs = {}
    for line in ref_lines:
        match = re.match(r"\[(\d+)\]\s*(.+)", line)
        if match:
            old_idx = match.group(1)
            doc_name = match.group(2).strip()
            ref_docs[old_idx] = doc_name

    # 去重并生成新标号映射
    unique_docs = {}
    new_idx = 1
    old_to_new_map = {}
    for old_idx, doc_name in ref_docs.items():
        if doc_name not in unique_docs:
            unique_docs[doc_name] = str(new_idx)
            new_idx += 1
        old_to_new_map[old_idx] = unique_docs[doc_name]

    # 更新正文中的标号
    new_body = body
    for old_idx in set(body_refs):
        old_ref = f"[{old_idx}]"
        if old_idx in old_to_new_map:
            new_ref = f"[{old_to_new_map[old_idx]}]"
            new_body = new_body.replace(old_ref, new_ref)

    # 重建参考文献列表
    new_refs = [f"[{idx}] {doc}" for doc, idx in unique_docs.items()]

    return f"{new_body}\n- 参考文档:\n" + "\n".join(new_refs)


def clean_references(response):
    # 1. 解析参考文献部分
    # ref_section_match = re.search(r"(.*?)- 参考文档:(.*)", response, re.DOTALL)
    ref_section_match = re.search(r"\n- 参考文档：\n(.*)", response, re.DOTALL)
    if not ref_section_match:
        return response  # 没有参考文献部分，直接返回

    ref_section = ref_section_match.group(1)
    # ref_section = ref_section_match.group(2).strip()
    print(ref_section)
    ref_lines = ref_section.strip().split("\n")

    # 2. 构建【原标号 -> 文档】映射
    ref_map = {}
    for line in ref_lines:
        match = re.match(r"\[(\d+)\] (.+)", line)
        if match:
            old_index, doc_name = match.groups()
            if doc_name not in ref_map.values():
                ref_map[old_index] = doc_name

    # 3. 重新编号
    new_ref_map = {doc: str(i+1) for i, doc in enumerate(ref_map.values())}
    old_to_new = {old: new_ref_map[doc] for old, doc in ref_map.items()}

    # 4. 提取正文中的所有引用，并删除无效的标号
    valid_indices = set(old_to_new.keys())  # 参考文献中存在的旧标号
    def replace_citation(match):
        old_index = match.group(1)
        return f"[{old_to_new[old_index]}]" if old_index in valid_indices else ""

    response = re.sub(r"\[(\d+)\]", replace_citation, response)

    # 5. 处理重复标号（如 [1][1] -> [1]）
    response = re.sub(r"(\[\d+\])\1+", r"\1", response)

    # 6. 重新生成参考文献列表
    new_ref_section = "\n".join(f"[{i}] {doc}" for doc, i in new_ref_map.items())

    return response[:ref_section_match.start()] + f"\n- 参考文档：\n{new_ref_section}"


# 4. 自定义 Prompt 并加载 RAG 管道
def load_rag_pipeline(vector_store_path, deepseek_api_key, parent_map_path):
    # 加载嵌入模型和向量数据库
    embedding_model = HuggingFaceEmbeddings(model_name="./bge-m3", model_kwargs={"device": "cuda"})
    vector_store = FAISS.load_local(vector_store_path, embedding_model, allow_dangerous_deserialization=True)

    # 配置DeepSeek大模型
    llm = ChatDeepSeek(
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        model="deepseek-chat",
        # model="deepseek-reasoner",
        api_key=deepseek_api_key,
        api_base="https://api.deepseek.com/v1",
        # model="deepseek-r1-distill-qwen",
        # api_key="EMPTY",
        # api_base="http://127.0.0.1:9997/v1",
        # model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        # api_key="0",
        # api_base="https://lf-service-x-ns-lf-x-vcvae5yvxqrl.sproxy.hd-01.alayanew.com:22443/v1",
    )

    # model_path = "./DeepSeek-R1-Distill-Qwen-7B"
    # # tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # # model = AutoModelForCausalLM.from_pretrained(
    # #     model_path,
    # #     local_files_only=True,
    # #     device_map="cuda:0",
    # #     # load_in_4bit=True,
    # #     trust_remote_code=True
    # # )
    # #
    # # # 创建 transformers pipeline，支持流式输出
    # # pipe = pipeline(
    # #     "text-generation",
    # #     model=model,
    # #     tokenizer=tokenizer,
    # #     max_new_tokens=8192,
    # #     # do_sample=True,
    # #     # temperature=0.7,
    # #     # top_p=0.9,
    # #     # return_full_text=False  # 不返回输入 Prompt
    # # )
    # #
    # # # 封装为 LangChain LLM，支持流式输出
    # # llm = HuggingFacePipeline(
    # #     pipeline=pipe,
    # #     model_kwargs={"streaming": True},  # 启用流式（实际由 pipeline 控制）
    # #     callbacks=[StreamingStdOutCallbackHandler()]  # 实时打印 token
    # # )
    #
    # llm = HuggingFacePipeline.from_model_id(
    #     model_id=model_path,
    #     task="text-generation",
    #     device=-1,
    #     # pipeline_kwargs={"streaming": True},
    #     # pipeline_kwargs={"max_new_tokens": 8192},
    #     # model_kwargs={"streaming": True},  # 启用流式（实际由 pipeline 控制）
    #     callbacks=[StreamingStdOutCallbackHandler()]  # 实时打印 token
    # )

    with open(parent_map_path, "rb") as f:
        parent_map = pickle.load(f)
        print(parent_map)

    # 定义自定义 Prompt
    custom_prompt_template = """
           你是一个专业的政策问答知识助手，基于提供的文档内容回答问题。请根据以下上下文准确、完整且全面地回答用户的问题，并附上参考的文档名称（严格按照文档名称输出，去掉路径部分，保留文件格式后缀）。如果问题超出文档范围，请明确说明“根据现有文档无法回答”，不用输出文档名称。

           上下文：
           {context}

           问题：
           {question}

           回答：
            - 根据当前政策，...
           - 参考文档:逐行列出回答参考的文档名称（根据现有文档无法回答，则无需回答此项）
           """
    # custom_prompt_template = """
    #    你是一个专业的政策问答知识助手，基于提供的文档内容回答问题。请根据以下上下文准确、简洁且全面地回答用户的问题，并附上参考的文档名称（严格按照文档名称输出，去掉路径部分，保留文件格式后缀）。如果问题超出文档范围，请明确说明“根据现有文档无法回答”，不用输出文档名称。
    #
    #    上下文：
    #    {context}
    #
    #    问题：
    #    {question}
    #
    #    回答：
    #    回答格式：
    #        - 根据当前政策，...（针对问题的回答内容，需要保证尊重原文，在回答的相应位置标注参考文档标号，不能出现参考文献列表里没有的标号）
    #        - 参考文档:逐行列出回答参考的文档名称，使用中括号进行文档标号[1]-[n]，标号需要是连续的，保证回答正文和此处的参考文献列表的标号是对应的
    #     输出前将回答正文中没有在参考文档列表里出现的标号删除。
    #    """
    # custom_prompt_template = """
    # 你是一个专业的政策问答知识助手，基于提供的文档内容回答问题。请根据以下上下文准确、简洁且全面地回答用户的问题，并附上参考的文档名称（严格按照文档名称输出，去掉路径部分，保留文件格式后缀）。如果问题超出文档范围，请明确说明“根据现有文档无法回答”，不用输出文档名称。
    #
    # 上下文：
    # {context}
    #
    # 问题：
    # {question}
    #
    # 回答：
    # 回答格式：
    #     - 根据当前政策，...（针对问题的回答内容，需要保证尊重原文，在回答的相应位置标注参考文档标号，不能出现参考文献列表里没有的标号）
    #     - 参考文档:逐行列出回答参考的文档名称，使用中括号进行文档标号[1]-[n]，标号需要是连续的，保证回答正文和此处的参考文献列表的标号是对应的
    # 注意输出前回答请按照以下步骤检查：
    # （1）回答正文中不能出现参考文档列表中没有的标号，回答正文里的标号必须是列表里有的。
    # （2）（重要！保证文档的标号唯一性）如果不同标号对应相同的文档需要进行合并，保证不同标号对应的文档是不同的，（重要！）同时需要更改回答正文中的标号保持对应关系准确（也就是说如果多个标号进行合并后正文中对应的标号也需要改为合并后的标号），回答正文中不能出现参考文档列表中没有的标号。
    # （3）之后，需要修改成连续的标号，列表里的标号必须需要是连续的，修改列表标号的同时修改回答正文的标号保持对应。
    # （4）（重要！保证回答中标号的正确性）最后回答正文中不能出现参考文档列表中没有的标号！！！如果出现这种情况将这种标号删除
    # """
    custom_prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

    # base_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    # base_retriever = vector_store.as_retriever(search_kwargs={"k": 20, 'fetch_k': 20})
    # base_retriever = vector_store.as_retriever(
    #     search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
    # )

    base_retriever = CustomRetriever(vector_store=vector_store, parent_map=parent_map, child_k=50, max_parent_k=10)
    # base_retriever = CustomRetriever(vector_store=vector_store, parent_map=parent_map, child_k=100, max_parent_k=10)

    rerank_model = HuggingFaceCrossEncoder(model_name="./bge-reranker-v2-m3")
    compressor = CrossEncoderReranker(model=rerank_model, top_n=5)
    # # compressor = LLMChainExtractor.from_llm(llm) # 使用 LLM 链来决定过滤掉哪些最初检索到的文档以及返回哪些文档，而无需操作文档内容
    #
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    # # 创建检索器
    # retriever = CustomRetriever(vector_store=vector_store, parent_map=parent_map, child_k=20, max_parent_k=10)

    # 创建RAG链，注入自定义 Prompt
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        # retriever=retriever,
        # retriever=base_retriever,
        retriever=compression_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_prompt}  # 注入自定义 Prompt
    )
    return rag_chain


# 5. 测试查询（保持不变）
def test_rag(rag_chain, query):
    # result = rag_chain({"query": query})
    # result = rag_chain.invoke(query)
    # result_post = post_process_answer(result['result'])
    print(f"问题: {query}")
    # rag_chain.run(query)
    result = rag_chain.stream({"query": query})
    # print(result)
    for re in result:
        print("\n")
        print("检索出的信息:")
        for doc in re["source_documents"]:
            print(f"- {doc.metadata['source']}: {doc.page_content}")
        # break
    # result = rag_chain.invoke(query)
    # print(f"回答: {result['result']}")
    # result_post = clean_references(result['result'])
    # print(f"回答: {result_post}")
    # print("检索出的信息:")
    # # print(result["source_documents"])
    # for doc in result["source_documents"]:
    #     # print(f"- {doc.metadata['source']}: {doc.page_content[:100]}...")
    #     print(f"- {doc.metadata['source']}: {doc.page_content}")


# 6. 主程序
if __name__ == "__main__":
    # document_directory = "./data"
    # vector_store_path = "./faiss_index_llm_nooverlap"
    # parent_map_path = "./parent_map_llm_nooverlap.pkl"

    document_directory = "./data_all"
    vector_store_path = "./faiss_index_llm_all"
    parent_map_path = "./parent_map_llm_all.pkl"

    # vector_store_path = "./faiss_index_llm_new_name"
    # parent_map_path = "./parent_map_new_name.pkl"
    # document_directory = "./test"
    # vector_store_path = "./test_faiss_index_llm"
    # parent_map_path = "./test_parent_map.pkl"
    deepseek_api_key = "sk-1dcdc022fa73474a8002f65505959b12"
    renew = True
    # renew = False

    if not os.path.exists(vector_store_path) or renew is True:
        build_knowledge_base(document_directory, deepseek_api_key, vector_store_path, parent_map_path)

    rag_pipeline = load_rag_pipeline(vector_store_path, deepseek_api_key, parent_map_path)

    while(1):
        query = input("请输入一些你的问题: ")
        print("思考中...")
        start_time = time.time()
        test_rag(rag_pipeline, query)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"运行时间: {elapsed_time} 秒")

    # start_time = time.time()
    # # query = "如何建立健全与企业常态化沟通机制？"
    # # query = "享受个人所得税优惠政策的高端紧缺人才覆盖哪些行业？"
    # # test_rag(rag_pipeline, query)
    # query = "我打算在海南购置一台微波炉，请问其是否符合海南自由贸易港的“零关税”政策？如果符合，其在清单中对应的税则号列是多少？"
    # test_rag(rag_pipeline, query)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"运行时间: {elapsed_time} 秒")
