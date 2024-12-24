import os
from litellm import completion

from aider.graph_alg import topological_sort_paths

# Bilingual prompts for code analysis
PROMPTS = {
    "system_prompt": {
        "en": "You are an AI model designed to analyze code and provide concise summaries and insights.",
        "zh": "你是一个AI模型，专门用于分析代码并提供简洁的总结和见解。"
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
        "en": "Based on these file descriptions, please provide a concise overview of the purpose of this code community/module:\n",
        "zh": "基于这些文件描述，请使用一句 中文简要概述这个模块的用途：\n"
    }
}


def generate_description(content, system_prompt, task_prompt):
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
            info.description = generate_description(system_prompt, "", prompt)
        else:
            info.description = ""

        community_infos[i] = info

        if hasattr(info, 'description'):
            print(f"Community {i+1} description: {info.description}")
            
    return community_infos