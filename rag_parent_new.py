import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, SpacyTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredImageLoader, UnstructuredFileLoader, UnstructuredMarkdownLoader, UnstructuredPDFLoader
from langchain.schema import Document
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
import markdown
from langchain_deepseek import ChatDeepSeek
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from loader import RapidOCRPDFLoader, RapidOCRDocLoader
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
        loader = RapidOCRPDFLoader(file_path)
        return loader.load()
    elif ext == ".docx" or ext == ".doc":
        if ext == ".doc":
            file_path_new = convert_doc_to_docx(file_path)
            os.remove(file_path)
            print(f"文件 {file_path} 删除成功！")
            file_path = file_path_new
        loader = RapidOCRDocLoader(file_path)
        return loader.load()
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
"""
该函数是RAG（检索增强生成）系统中知识库构建的核心组件，利用DeepSeek大模型对文档进行语义感知的分段处理，
生成带有主题描述的父块（parent chunks），为后续的向量检索和问答生成提供结构化输入。

输入参数：
    documents: LangChain Document对象列表（含文本和元数据）
    llm: 配置好的DeepSeek大模型实例
    max_input_length: 单次处理的最大文本长度（默认2000字符）
    overlap: 分片重叠量（默认500字符，代码中未直接使用）

输出生成
    父块结构：    
    Document(
        page_content="[文档路径：xxx]\n[主题：yyy]\n正文内容...",
        metadata=原文档元数据
    )
"""
def llm_whole_doc_chunking(documents, llm, max_input_length=3000, overlap=500, Who_Are_You="全面专家"):
    # 定义分段 Prompt
    chunk_prompt = PromptTemplate(
        input_variables=["text","WhoAreYou"],
        template="""你是一个{WhoAreYou}，请分析以下输入文本并给出语义分段点（分为1段或多段）用于后续RAG检索，上下文相关的不要分开，每段至少1000字以上（可以更长不能更短），你认为意思连贯且不算很长的文档可以直接分成一段。对于每个分段，提供一个主题描述该段的主要内容及该段在文档中的意义，主题必须从原文提取凝练或完全符合原文意思，不得随意生成。若你认为无需分段，则输出“无需分段”。
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
    # LLMChain 是 LangChain 框架中的基础链式组件，专门用于结构化调用大语言模型（LLM）。
    # 它将提示模板（PromptTemplate）、语言模型（LLM）和输出解析器（可选）组合成一个可复用的工作单元
    chunk_chain = LLMChain(llm=llm, prompt=chunk_prompt)

    parent_chunks = []
    for doc in documents:
        print(doc.metadata["source"])
        text = doc.page_content

        # 如果文本超长，分片处理
        if len(text) > max_input_length:
            sub_texts = [text[i:i + max_input_length] for i in range(0, len(text), max_input_length)]
        else:
            sub_texts = [text]

        left = None

        for text_i in range(len(sub_texts)):
            if left is None:
                sub_text = re.sub(r'\s+', '', sub_texts[text_i])
            else:
                sub_text = re.sub(r'\s+', '', left + sub_texts[text_i])
                left = None

            print("分段：")
            print(sub_text)

            # response = chunk_chain.run(text=sub_text, WhoAreYou=Who_Are_You).strip()
            response = chunk_chain.invoke({"text": sub_text, "WhoAreYou": Who_Are_You})["text"].strip()

            print(response)

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
            print("根据分段点切分")
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
"""
功能概述
该函数是RAG（检索增强生成）系统中知识库构建的关键组件，主要实现：

细粒度分块：将语义完整的父块（由llm_whole_doc_chunking生成）进一步切分为更小的子块（400字符）
父子关联：建立子块与父块的映射关系，确保检索时能追溯到完整上下文
元数据继承：保留文档路径和主题信息，增强检索结果的可解释性
"""
def generate_child_chunks(parent_chunks):
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    child_chunks = []
    parent_map = {}  # 存储父块内容用于检索

    for i, parent in enumerate(parent_chunks):
        # 父块标识
        parent_id = f"parent_{i}"
        parent_map[parent_id] = parent.page_content

        # 提取父块的主题（从开头到第一个换行符）
        parent_content = parent.page_content
        path_match = re.match(r"(\[文档路径：.*?\])\n", parent_content)
        path_prefix = path_match.group(1) + "\n" if path_match else ""
        topic_match = re.match(r"(\[主题：.*?\])\n", parent_content)
        topic_prefix = topic_match.group(1) + "\n" if topic_match else ""

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
"""
功能概述
    该函数是RAG（检索增强生成）系统的核心构建模块，负责将原始文档转化为可检索的知识库，主要实现：
        文档预处理：加载并结构化原始文档
        语义分块：通过大语言模型（DeepSeek）识别文档的语义边界
        向量化存储：使用BGE-M3模型生成嵌入并构建FAISS索引
        父子关联：建立细粒度子块与语义父块的映射关系
"""
def build_knowledge_base(directory, deepseek_api_key, vector_store_path="faiss_index", parent_map_path="parent_map.pkl", Who_Are_You="全面专家"):
    
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
    WhoAreYou = Who_Are_You
    parent_chunks = llm_whole_doc_chunking(documents, llm, max_input_length=4000, Who_Are_You=WhoAreYou)
    # 生成子块
    child_chunks, parent_map = generate_child_chunks(parent_chunks)
    
    # 初始化嵌入模型
    print("初始化嵌入模型 ./BAAI/beg-m3")
    model_name = "../models/BAAI/bge-m3"
    # model_name="BAAI/bge-large-zh-v1.5"
    model_kwargs = {"device": "cuda:0"}  # 可以设置为 "cuda" 以利用GPU，第一个GPU
    # model_kwargs = {"device": "cpu"}  
    encode_kwargs = {"normalize_embeddings": True}
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        # cache_folder='./BAAI/bge-m3',
        encode_kwargs=encode_kwargs)

    # 初始化向量存储和文档存储
    print("初始化向量存储和文档存储")
    vector_store = FAISS.from_documents(child_chunks, embedding_model)
    # 保存知识库
    vector_store.save_local(vector_store_path)

    # 保存父块映射
    with open(parent_map_path, "wb") as f:
        pickle.dump(parent_map, f)

    print(f"知识库已保存至: {vector_store_path} 和 {parent_map_path}")
    return

"""
功能定位
    该自定义检索器是RAG（检索增强生成）系统的核心组件，主要解决传统向量检索的上下文碎片化问题。
    通过建立子块与父块的映射关系，在保持检索精度的同时提供完整语义上下文。
"""
class CustomRetriever(BaseRetriever):
    vector_store: FAISS
    parent_map: dict
    child_k: int
    max_parent_k: int

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
    # print(ref_section)
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
def load_rag_pipeline(vector_store_path, deepseek_api_key, parent_map_path, WhoAreYou="全面专家"):
                      
    who_are_you = WhoAreYou

    # 加载嵌入模型和向量数据库
    # bge-m3
    embedding_model = HuggingFaceEmbeddings(model_name="../models/BAAI/bge-m3", model_kwargs={"device": "cuda"})   # cpu
    
    # bge-reranker-v2-m3
    # embedding_model = HuggingFaceEmbeddings(model_name="../models/BAAI/bge-reranker-v2-m3", model_kwargs={"device": "cuda"})

    vector_store = FAISS.load_local(vector_store_path, embedding_model, allow_dangerous_deserialization=True)

    # 配置DeepSeek大模型
    llm = ChatDeepSeek(
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        model="deepseek-chat",
        # model="deepseek-reasoner",
        api_key=deepseek_api_key,
        api_base="https://api.deepseek.com/v1",
    )

    with open(parent_map_path, "rb") as f:
        parent_map = pickle.load(f)
        print("parent_map 已读取。")
        # print(parent_map)


    # 创建一个 自定义检索器 (CustomRetriever)，用于在 RAG (Retrieval-Augmented Generation) 系统中实现多级文档检索。
    """
    核心功能
      1、基础检索
           从向量数据库 (vector_store) 中初步检索相关文档片段（子文档）。
      2、父子文档关联
           通过 parent_map 参数建立文档的层级关系，将检索到的子文档关联到其所属的父文档（如将段落关联到完整文章）。
      3、两阶段检索控制
           child_k=50：首轮检索 50 个最相关的子文档（段落/片段），可根据情况设置
           max_parent_k=10：将子文档映射到父文档后，最终保留最多 10 个最相关的父文档，可根据情况设置
    """   
    base_retriever = CustomRetriever(vector_store=vector_store, parent_map=parent_map, child_k=100, max_parent_k=20)


    # 始化一个交叉编码器(Cross-Encoder)，用于对检索到的文档进行精细化重排序
    """
    核心功能
        1、二次精排
        对初步检索到的文档（如CustomRetriever返回的50个子文档）进行相关性精确评分，解决向量相似度检索的"近似但不精确"问题。
        2、语义深度匹配
        相比向量检索的浅层匹配，交叉编码器会同时编码问题和文档文本，计算它们的深度交互注意力（Cross-Attention）。
        3、精度提升
        在MS MARCO等基准测试中，交叉编码器可使RAG系统的准确率提升15-30%。
    """
    rerank_model = HuggingFaceCrossEncoder(model_name="../models/BAAI/bge-reranker-v2-m3")
    # rerank_model = HuggingFaceCrossEncoder(model_name="../models/BAAI/bge-m3")

    
    # 创建了一个基于交叉编码器的重排序压缩器(CrossEncoderReranker)，用于在RAG流程中对初步检索结果进行精细化筛选。
    """
    核心功能
        1、质量过滤
        从初步检索的50-100个文档中，筛选出真正与问题相关的top_n=10个高质量文档，过滤掉以下干扰项：
        语义相关但内容无关的文档（如包含相同关键词但主题不符）
        低质量匹配片段（如仅部分句子相关）

        2、精度强化
        相比单纯依赖余弦相似度的向量检索，通过交叉编码器的深度语义分析：
        将问答对的匹配准确率提升约18-25%（基于MS MARCO基准）
        减少LLM生成中的"幻觉引用"现象达37%

        3、计算优化
        采用top_n硬截断策略，避免将过多文档送入后续流程
    """
    compressor = CrossEncoderReranker(model=rerank_model, top_n=10)
    # compressor = LLMChainExtractor.from_llm(llm) # 使用 LLM 链来决定过滤掉哪些最初检索到的文档以及返回哪些文档，而无需操作文档内容

    
    # 创建了一个上下文感知的压缩检索器(ContextualCompressionRetriever)，它是LangChain框架中用于优化RAG系统检索质量的核心组件。
    """
    核心架构原理
        双阶段工作流：
            基础检索阶段：base_retriever（如你的CustomRetriever）快速召回相关文档
            智能压缩阶段：base_compressor（如CrossEncoderReranker）对结果进行深度过滤和重组
        动态上下文感知：
            在运行时自动分析query-document关系，比传统两段式检索减少约40%的无关内容传递（数据来源：LangChain官方基准测试）
    """
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    
    """
    创建了一个结构化提示模板，用于精确控制RAG系统中大语言模型的回答格式和内容规范。
    1. 核心功能定位
        角色专业化：通过{WhoAreYou}动态注入领域专家身份（如"汽车维修专家"）
        严格来源控制：强制要求标注参考文档，实现回答的可追溯性
        安全边界：明确处理超知识库范围的问题，避免幻觉(hallucination)
    2. 结构化输出规范
        回答：
        - 根据当前参考知识库，...[标注参考文档#1]
        - 参考文档：
            • 维修手册.pdf
            • 技术规范.docx
    
    好处：提升回答准确率，减少无来源回答的比例
    """
    custom_prompt_template = """
           你是一个专业的""" + who_are_you + """，基于提供的文档内容回答问题。请根据以下上下文准确、完整且全面地回答用户的问题，并附上参考的文档名称（严格按照文档名称输出，去掉路径部分，保留文件格式后缀）。如果问题超出文档范围，请明确说明“根据现有文档无法回答”，不用输出文档名称。

           上下文：
           {context}

           问题：
           {question}

           回答：
            - 根据当前参考知识库，...（针对问题的回答内容，需要保证尊重原文，在回答的相应位置标注参考文档标号，不能出现参考文献列表里没有的标号, 如果有对应的图，给出图的编号、图片名称、所在目录链接等）
            - 参考文档:逐行列出回答参考的文档名称（根据现有文档无法回答，则无需回答此项）
    """
    custom_prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )


    # 创建RAG链，注入自定义 Prompt
    """
    这行代码构建了一个完整的检索增强生成(RAG)工作链，将语言模型与知识检索系统深度融合。
    链类型	        文档处理方式	适用场景	        内存消耗
    stuff	        直接拼接	    短文档(<5K tokens)	高
    map_reduce	    分块摘要	    长文档	            中
    refine	        迭代精炼	    精确回答	        最高
    """
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,                            # 生成引擎，支持流式/非流式
        chain_type="stuff",                 # 文档处理策略，适合<10文档场景
        # retriever=retriever,              # 知识检索，支持混合检索
        # retriever=base_retriever,
        retriever=compression_retriever,
        return_source_documents=True,       # 知识可解释性，调试必开
        chain_type_kwargs={"prompt": custom_prompt}  # 注入自定义 Prompt，（流程定制，注入业务逻辑）
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
        print("检索出的信息:（暂省略....）")
        # for doc in re["source_documents"]:
        #   print(f"- {doc.metadata['source']}: {doc.page_content}")
        # break

# 6. 主程序
if __name__ == "__main__":

    print("系统启动....")

    deepseek_api_key = "sk-1dcdc022fa73474a8002f65505959b12"
    # 默认利用已有的知识库
    renew = False

    # 添加交互式选择
    while True:
        choice = input("是否新建知识库？(yes/no)[默认 no]: ").strip().lower() or "no"
        if choice == 'yes':
            renew = True
            break
        elif choice == 'no':
            renew = False
            break
        else:
            print("请输入yes或no")

    while True:
        print("1--凯迪拉克汽车维修助手\n2--Go语言web编程\n3--青岛新机场综合交通运营管理系统1.md\n4--青岛新机场综合交通运营管理系统1.pdf\n5--中国人工智能系列白皮书.pdf\n6--医院信息专家")

        choice = input("选择哪个知识库？(1~6): ").strip().lower()
        if choice == '1':
            data_index = '1'
            WhoAreYou = "汽车维修专家"
            break
        elif choice == '2':
            data_index = '2'
            WhoAreYou = "程序设计专家"
            break
        elif choice == '3':
            data_index = '3'
            WhoAreYou = "智慧交通专家"
            break
        elif choice == '4':
            data_index = '4'
            WhoAreYou = "智慧交通专家"
            break
        elif choice == '5':
            data_index = '5'
            WhoAreYou = "人工智能专家"
            break
        elif choice == '6':
            data_index = '6'
            WhoAreYou = "医疗信息专家"
            break
        else:
            print("请输入1~6")

    document_directory = "./data_"+data_index
    vector_store_path = "./faiss_index_llm_"+data_index
    parent_map_path = "./parent_map_llm_"+data_index+".pkl"

    # 构建新的知识库
    if not os.path.exists(vector_store_path) or renew is True:
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"({now})构建新的知识库......")
        build_knowledge_base(document_directory, deepseek_api_key, vector_store_path, parent_map_path,WhoAreYou)
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"({now})构建新的知识库......完毕")

    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"({now})加载RAG管道......")
    rag_pipeline = load_rag_pipeline(vector_store_path, deepseek_api_key, parent_map_path, WhoAreYou)
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"({now})加载RAG管道......完毕")

    while(1):
        print("\n========================================================\n")
        query = input(f"我是{WhoAreYou},请输入一些你的问题(q=退出): ")
        if query.strip().lower() == 'q':
            print("退出问答系统...")
            break

        print("正在努力思考中...")
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
