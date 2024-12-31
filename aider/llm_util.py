import os
import re

from litellm import completion
from openai import OpenAI

from aider.utils import get_aidoc_dir
from aider.graph_alg import topological_sort_paths

# Bilingual prompts for code analysis
PROMPTS = {
    "system_prompt": {
        "en": "You are an AI model designed to analyze code and provide concise summaries and insights.",
        "zh": "你是一个资深的软件项目分析专家，精通所有编程语言、技术栈、框架，特别擅长从整体上分析软件仓库并提供简洁的分析和总结。"
    },
    "file_overview": {
        "en": "Read the following code and provide a concise overview of the purpose of the file, excluding any introduction, explanation, or unnecessary details.",
        "zh": "请阅读代码，使用一句中文简要概述此文件的用途，不需要包含介绍、解释或其他不必要的细节。"
    },
    "file_with_dependencies": {
        "en": "\nThis file is used by other files with these purposes:\n",
        "zh": "\n此文件被以下用途的其他文件使用：\n"
    },
    "community_overview": {
        "en": "Based on these file descriptions, please provide:\n1. A concise module name (2-4 words) that best represents this community's functionality\n2. A concise overview of the purpose of this code community/module\nFormat the response as:\nName: [module name]\nDescription: [description]",
        "zh": "基于这些文件描述，请提供：\n1. 一个最能代表这个社区功能的简短模块名称（2-4个词）\n2. 一句话简要概述这个模块的用途\n请按如下格式回复：\n模块名：[模块名]\n描述：[描述]"
    },
    "project_summary": {
        "en": "Based on these community descriptions, please provide a concise summary of the entire project, focusing on its main purpose and how different modules work together:\n",
        "zh": "基于这些社区描述，请用中文简要总结整个项目，重点说明项目的主要用途以及不同模块是如何协同工作的：\n"
    },
    "flow_diagram": {
        "en": "Based on the topological sort path of these files and their descriptions, generate a Mermaid flowchart that represents the business logic flow. For each node and edge, provide a concise label that describes its role in the flow. The flowchart should be in the format:\n```mermaid\ngraph TD\n...\n```\nFocus on the high-level business logic rather than implementation details. The generated flowchart must be free of syntax errors and must be renderable directly in markdown.",
        "zh": "基于这些文件的拓扑排序路径和它们的描述，生成一个表示业务逻辑流程的Mermaid流程图。为每个节点和边提供简洁的标签来描述其在流程中的角色。流程图应该采用以下格式：\n```mermaid\ngraph TD\n...\n```\n请关注高层业务逻辑而不是实现细节。生成的流程图不能有语法错误，必须可以在markdown中直接可以渲染显示。"
    }
}

def generate_description(content, system_prompt, task_prompt):
    """调用DeepSeek生成描述
    
    Args:
        content: str 代码内容
        task_prompt: dict 包含中英文的提示语字典
    """
    client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": f"{system_prompt}"},
                {"role": "user", "content": f"{task_prompt}\n\nCode:\n```\n{content}\n```"},
            ],
            stream=False
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Failed to generate description: {e}")
        print(f"Error details: {str(e)}")
        return ""


def generate_description_ollama(content, system_prompt, task_prompt):
    """调用Ollama生成描述
    
    Args:
        content: str 代码内容
        task_prompt: dict 包含中英文的提示语字典
    """
    ollama_base_url = 'http://localhost:11434'
    try:
        response = completion(
            model="ollama/qwen2.5:7b",
            messages=[{
                "role": "system",
                "content": f"{system_prompt}"
            }, {
                "role": "user",
                "content": f"{task_prompt}\n\nCode:\n```\n{content}\n```"
            }],
            api_base=ollama_base_url
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Failed to generate description: {e}")
        print(f"Error details: {str(e)}")
        return ""

def get_file_content(repo_path, fname):
    """获取文件内容"""
    try:
        # 尝试将相对路径转换为绝对路径
        abs_path = os.path.join(repo_path, fname) if not os.path.isabs(fname) else fname
        with open(str(abs_path), "r", encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Failed to read file {fname}: {e}")
        return ""

def process_file_recursively(community_graph, repo_path, node, visited, descriptions, lang):
    """递归处理文件，生成描述"""
    if node in visited:
        return descriptions.get(node, "")
        
    visited.add(node)
    content = get_file_content(repo_path, node)
    
    # 获取依赖于当前文件的其他文件
    dependent_files = list(community_graph.predecessors(node))
    dependent_descriptions = []
    
    # 递归处理依赖文件
    for dep in dependent_files:
        desc = process_file_recursively(community_graph, repo_path, dep, visited, descriptions, lang)
        if desc:
            dependent_descriptions.append(desc)
            
    # 生成当前文件的描述
    system_prompt = PROMPTS["system_prompt"][lang]
    prompt = PROMPTS["file_overview"][lang]
    if dependent_descriptions:
        prompt = prompt + PROMPTS["file_with_dependencies"][lang]
        prompt += "\n".join(f"- {desc}" for desc in dependent_descriptions)
        
    description = generate_description(system_prompt, content, prompt)
    descriptions[node] = description
    return description

def generate_community_descriptions(G, repo_path, communities, lang='en'):
    """
    为每个社区生成描述
    
    Args:
        G: networkx.MultiDiGraph 代码依赖关系图
        communities: list 社区列表，key是社区ID, value是包含文件名的集合
        ollama_base_url: str Ollama服务的基础URL
        lang: str 语言选择，'en' 表示英文，'zh' 表示中文，默认为英文
        
    Returns:
        dict: 社区描述字典，key是社区ID，value是CommunityInfo对象
    """
    try:
        from litellm import completion
    except ImportError:
        print("litellm package not installed, community description disabled")
        return None
        
    class CommunityInfo:
        def __init__(self):
            self.files = set()  # 社区包含的文件
            self.file_descriptions = {}  # 文件的描述，key是文件路径
            self.description = ""  # 社区整体描述
            self.module_name = ""  # 模块名称
            self.dependency_paths = []  # 社区内的依赖路径
            
    # 验证语言参数
    if lang not in ['en', 'zh']:
        print(f"Invalid language '{lang}', falling back to English")
        lang = 'en'
            
    community_infos = {}
    
    for i, community in communities.items():
        print(f"Processing community {i+1}...")
            
        info = CommunityInfo()
        info.files = community
        
        # 创建社区子图
        community_graph = G.subgraph(community)
        
        # 获取社区内的拓扑排序路径
        paths = topological_sort_paths(community_graph)
        if not paths:
            continue

        paths, _ = paths
        info.dependency_paths = paths
        
        # 从每个路径的终点开始生成描述
        descriptions = {}
        for path in paths:
            if not path:
                continue
            end_node = path[-1]
            visited = set()
            process_file_recursively(community_graph, repo_path, end_node, visited, descriptions, lang)
            
        info.file_descriptions = descriptions
        
        # 生成社区整体描述
        if descriptions:
            all_descriptions = "\n".join(descriptions.values())
            system_prompt = PROMPTS["system_prompt"][lang]
            prompt = PROMPTS["community_overview"][lang] + all_descriptions
            response = generate_description(system_prompt, "", prompt)
            
            # 解析响应获取模块名和描述
            if lang == 'zh':
                module_name_match = re.search(r'模块名：(.+?)\n', response)
                description_match = re.search(r'描述：(.+)', response)
            else:
                module_name_match = re.search(r'Name: (.+?)\n', response)
                description_match = re.search(r'Description: (.+)', response)
                
            if module_name_match and description_match:
                info.module_name = module_name_match.group(1).strip()
                info.description = description_match.group(1).strip()
            else:
                info.module_name = f"模块 {i}"
                info.description = response.strip()
        else:
            info.module_name = f"模块 {i}"
            info.description = ""

        community_infos[i] = info

        if hasattr(info, 'description'):
            print(f"Community {i+1} description: {info.description}")
            
    return community_infos

def generate_flow_diagram(G, repo_path, path, community_info, lang='zh'):
    """为给定的拓扑排序路径生成Mermaid流程图
    
    Args:
        G: networkx.MultiDiGraph 代码依赖关系图
        repo_path: str 仓库根目录路径
        path: list 拓扑排序路径中的文件列表
        community_info: CommunityInfo 社区信息对象
        lang: str 语言选择，'en' 或 'zh'
        
    Returns:
        str: Mermaid格式的流程图
    """
    # 准备文件描述和路径信息
    path_info = []
    for fname in path:
        content = get_file_content(repo_path, fname)
        description = community_info.file_descriptions.get(fname, '')
        path_info.append({
            'file': fname,
            'content': content,
            'description': description
        })
        
    # 构建提示语
    prompt = PROMPTS["flow_diagram"][lang] + "\n\nFiles in path:\n"
    for info in path_info:
        prompt += f"\n{info['file']}:\n{info['description']}\n"
    
    prompt += f"\nCommunity purpose: {community_info.description}\n"
    
    # 调用模型生成流程图
    system_prompt = PROMPTS["system_prompt"][lang]
    mermaid = generate_description(content=prompt, system_prompt=system_prompt, task_prompt=prompt)
    
    # 提取Mermaid代码块
    import re
    match = re.search(r'```mermaid\n(.*?)```', mermaid, re.DOTALL)
    if match:
        return match.group(1).strip()
    return mermaid

def save_flow_diagram(repo_root, G, community_infos=None):
    """生成README.md内容
    
    Args:
        repo_root: str 仓库根目录路径
        G: networkx.MultiDiGraph 代码依赖关系图
        community_infos: dict 社区信息字典，key是社区ID，value是CommunityInfo对象
    """
    if not community_infos:
        return
        
    # 获取项目名称
    project_name = os.path.basename(repo_root)
    
    # 生成项目总结
    project_summary = generate_project_summary(repo_root, community_infos, lang='zh')
    
    # 构建README.md内容
    content = f"# {project_name}\n\n"
    content += "## 项目总结\n"
    content += f"{project_summary}\n\n"
    content += f"## 项目概览\n\n"
    content += "[点击在浏览器中打开](repo_overview_with_communities.html)\n\n"
    content += "## 主要模块\n"
    
    # 为每个社区生成内容
    for community_id, info in community_infos.items():
        content += f"### {info.module_name}\n"
        content += f"{info.description}\n\n"
        
        # 为每个社区的依赖路径生成流程图
        for path in info.dependency_paths:
            if path:  # 确保路径非空
                flow_diagram = generate_flow_diagram(G, repo_root, path, info, lang='zh')
                if flow_diagram:
                    content += "```mermaid\n"
                    content += flow_diagram
                    content += "\n```\n\n"
    
    # 保存README.md
    output_path = os.path.join(get_aidoc_dir(repo_root), "README.md")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

def generate_project_summary(repo_root, community_infos, lang='zh'):
    """生成项目总结
    
    Args:
        repo_root: str 项目路径
        community_infos: dict 社区信息，key是社区id，value是CommunityInfo对象
        lang: str 语言，'zh'或'en'
    """
    project_name = os.path.basename(repo_root)
    
    # 尝试读取README.md文件
    readme_content = ""
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            readme_content = f.read()
    
    # 准备模块描述信息
    module_descriptions = []
    for community_id, info in community_infos.items():
        module_descriptions.append(f"- {info.description}")
    
    # 构建提示语
    prompt = PROMPTS["project_summary"][lang]
    content = f"""
项目名称：{project_name}

README内容：
{readme_content}

模块描述：
{'\n'.join(module_descriptions)}
"""
    
    try:
        # 调用DeepSeek生成项目总结
        client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": PROMPTS["system"][lang]},
                {"role": "user", "content": f"{prompt}\n\n{content}"},
            ],
            max_tokens=1000,  # 允许更长的总结
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating project summary: {e}")
        # 如果API调用失败，生成一个基本的总结
        basic_summary = f"项目 {project_name} 包含以下主要模块：\n" + "\n".join(module_descriptions)
        return basic_summary
