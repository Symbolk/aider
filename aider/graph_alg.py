from collections import defaultdict
from loguru import logger

def to_mermaid(G, max_edges=100):
    """
    将NetworkX图转换为Mermaid格式，生成更紧凑的图表
    
    Args:
        G: networkx.MultiDiGraph 需要转换的图
        max_edges: int 最大边数限制
        
    Returns:
        str: Mermaid格式的图表字符串
    """
    # Mermaid图表头部
    mermaid = ["graph TD;"]
    
    # 记录已处理的边和节点
    processed_edges = set()
    processed_nodes = set()
    
    # 简化文件名的函数
    def simplify_filename(fname):
        # 只保留最后两层路径
        parts = fname.split('/')
        if len(parts) > 2:
            return '/'.join(parts[-2:])
        return fname
    
    # 处理所有边，限制最大边数
    for i, (u, v, data) in enumerate(G.edges(data=True)):
        if i >= max_edges:
            break
        
        # 创建唯一的边标识
        edge_id = (u, v)
        
        # 如果这条边已经处理过，跳过
        if edge_id in processed_edges:
            continue
            
        # 获取这对节点之间的所有边的权重总和
        weight = sum(d.get('weight', 1) for _, _, d in G.edges(data=True) if (_, _) == edge_id)
        
        # 简化节点标签
        u_simple = simplify_filename(u)
        v_simple = simplify_filename(v)
        
        # 清理节点标签
        u_clean = u_simple.replace('/', '_').replace('.', '_')
        v_clean = v_simple.replace('/', '_').replace('.', '_')
        
        # 添加节点定义（如果还没添加过）
        if u_clean not in processed_nodes:
            mermaid.append(f"    {u_clean}[{u_simple}]")
            processed_nodes.add(u_clean)
        if v_clean not in processed_nodes:
            mermaid.append(f"    {v_clean}[{v_simple}]")
            processed_nodes.add(v_clean)
        
        # 根据权重设置线条粗细
        if weight > 5:
            thickness = "==>"    # 粗线
        elif weight > 2:
            thickness = "-->"    # 中等线
        else:
            thickness = "-.->"   # 虚线
            
        # 添加边定义，不显示权重标签以减少复杂度
        edge_line = f"    {u_clean} {thickness} {v_clean}"
        mermaid.append(edge_line)
        
        # 记录已处理的边
        processed_edges.add(edge_id)
    
    # 返回完整的Mermaid图表定义
    return "\n".join(mermaid)

def save_mermaid_diagram(repo_root, G, output_path=None):
    """
    将依赖图渲染为图片并保存
    
    Args:
        G: networkx.MultiDiGraph 代码依赖关系图
        output_path: str 输出文件路径，默认为 root/repo_overview.png
    """
    try:
        from mdutils import MdUtils
        import subprocess
        import tempfile
        import os
        import json
        
        # 生成Mermaid图表内容
        mermaid_content = to_mermaid(G)
        
        # 如果没有指定输出路径，使用默认路径
        if output_path is None:
            output_path = os.path.join(repo_root, "repo_overview.png")
            
        # 创建临时配置文件，增加最大文本限制
        config = {
            "theme": "default",
            "maxTextSize": 50000,  # 增加文本大小限制
            "fontFamily": "arial",
            "fontSize": 12,        # 减小字体大小
            "flowchart": {
                "htmlLabels": True,
                "curve": "basis",
                "padding": 15,
                "useMaxWidth": False,
                "diagramPadding": 20,
                "nodeSpacing": 30,
                "rankSpacing": 30
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_config:
            json.dump(config, temp_config)
            temp_config_path = temp_config.name
            
        # 创建临时markdown文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as temp_md:
            md = MdUtils(file_name=temp_md.name)
            
            # 添加Mermaid图表，使用较小的节点
            mermaid_content = mermaid_content.replace('["', '[/"').replace('"]', '"/]')  # 使用更紧凑的节点样式
            md.write(f"```mermaid\n{mermaid_content}\n```\n")
            md.create_md_file()
            temp_md_path = temp_md.name
            
        # 使用mermaid-cli渲染图片
        try:
            subprocess.run([
                'mmdc',
                '-i', temp_md_path,
                '-o', output_path,
                '-b', 'transparent',
                '-w', '3840',     # 增加宽度
                '-H', '2160',     # 增加高度
                '-c', temp_config_path,  # 使用配置文件
                '-s', '2'         # 增加缩放因子
            ], check=True)
            
            print(f"Mermaid diagram saved to: {output_path}")
                
        except subprocess.CalledProcessError as e:
            print(f"Error rendering Mermaid diagram: {e}")
            print("Please ensure mermaid-cli is installed (npm install -g @mermaid-js/mermaid-cli)")
        except FileNotFoundError:
            print("mermaid-cli not found")
            print("Please install mermaid-cli: npm install -g @mermaid-js/mermaid-cli")
        
        # 清理临时文件
        try:
            os.unlink(temp_md_path)
            os.unlink(temp_config_path)
        except:
            pass
            
    except ImportError as e:
        print(f"Required package not found: {e}")
        print("Please install required packages:")
        print("pip install mdutils")


def detect_communities(G):
        """
        使用Louvain算法对代码库图进行社区检测
        
        Args:
            G: networkx.MultiDiGraph 代码依赖关系图
            
        Returns:
            list: 社区列表，key是社区id，value是一个包含文件名的集合
        """
        try:
            import community  # python-louvain package
        except ImportError:
            logger.error("python-louvain package not installed, community detection disabled")
            return None
            
        # 将MultiDiGraph转换为无向图以便进行社区检测
        undirected_G = G.to_undirected()
        
        # 如果图为空或只有一个节点，返回None
        if len(undirected_G.nodes()) <= 1:
            return None
            
        # 使用Louvain算法进行社区检测
        partition = community.best_partition(undirected_G)
        
        # 将结果组织成社区列表
        communities = defaultdict(set)
        for node, community_id in partition.items():
            communities[community_id].add(node)
            
        return communities

def topological_sort_paths(G, source=None, target=None, max_paths=100):
    """
    对图进行拓扑排序，返回所有可能的排序路径。
    如果图中存在环，使用Tarjan算法进行缩点处理后再排序。
    
    Args:
        G: networkx.MultiDiGraph 代码依赖关系图
        source: str, optional 起始节点，如果为None则从所有入度为0的节点开始
        target: str, optional 目标节点，如果为None则到所有出度为0的节点结束
        max_paths: int, optional 最大返回路径数，防止路径数过多
        
    Returns:
        list: 拓扑排序路径列表，每个路径是一个节点序列
        dict: 强连通分量映射，key是缩点后的节点ID，value是原始节点集合
    """
    import networkx as nx
    
    def _find_sccs(G):
        """使用Tarjan算法找出图中的强连通分量"""
        sccs = list(nx.strongly_connected_components(G))
        scc_mapping = {}  # 原始节点到SCC ID的映射
        scc_nodes = {}    # SCC ID到原始节点集合的映射
        
        for i, scc in enumerate(sccs):
            scc_id = f"scc_{i}"
            scc_nodes[scc_id] = set(scc)
            for node in scc:
                scc_mapping[node] = scc_id
                
        return scc_mapping, scc_nodes
        
    def _create_condensation_graph(G, scc_mapping):
        """创建缩点后的图"""
        C = nx.DiGraph()
        
        # 添加缩点后的节点
        for scc_id in set(scc_mapping.values()):
            C.add_node(scc_id)
            
        # 添加缩点后的边
        for u, v in G.edges():
            scc_u = scc_mapping[u]
            scc_v = scc_mapping[v]
            if scc_u != scc_v:  # 只在不同的强连通分量之间添加边
                C.add_edge(scc_u, scc_v)
                
        return C
        
    def _all_paths_topological_sort(G, start_nodes, end_nodes, path=None, visited=None, all_paths=None):
        if path is None:
            path = []
        if visited is None:
            visited = set()
        if all_paths is None:
            all_paths = []
            
        # 如果已经找到足够多的路径，提前返回
        if len(all_paths) >= max_paths:
            return all_paths
            
        # 如果当前路径到达了终点
        if path and path[-1] in end_nodes:
            all_paths.append(path[:])
            return all_paths
            
        # 获取当前可访问的节点
        if not path:
            candidates = start_nodes
        else:
            current = path[-1]
            candidates = set()
            for successor in G.successors(current):
                if successor not in visited:
                    # 检查successor的所有前驱是否都已经在路径中
                    predecessors = set(G.predecessors(successor))
                    if predecessors.issubset(set(path)):
                        candidates.add(successor)
                        
        # 递归访问每个候选节点
        for next_node in sorted(candidates):  # 排序以确保结果的确定性
            if next_node not in visited:
                visited.add(next_node)
                path.append(next_node)
                _all_paths_topological_sort(G, start_nodes, end_nodes, path, visited, all_paths)
                path.pop()
                visited.remove(next_node)
                
        return all_paths
        
    def _expand_path(path, scc_nodes):
        """展开缩点后的路径，将每个SCC节点替换为其包含的原始节点"""
        expanded_path = []
        for node in path:
            if node.startswith('scc_'):
                # 对于SCC节点，将其包含的所有原始节点添加到路径中
                expanded_path.extend(sorted(scc_nodes[node]))
            else:
                expanded_path.append(node)
        return expanded_path
        
    # 检查是否需要进行缩点处理
    try:
        nx.find_cycle(G)
        has_cycle = True
        logger.info("Graph contains cycles, applying Tarjan's algorithm for condensation")
    except nx.NetworkXNoCycle:
        has_cycle = False
        
    if has_cycle:
        # 使用Tarjan算法找出强连通分量
        scc_mapping, scc_nodes = _find_sccs(G)
        
        # 创建缩点后的图
        C = _create_condensation_graph(G, scc_mapping)
        
        # 调整source和target到对应的SCC
        if source:
            source = scc_mapping[source]
        if target:
            target = scc_mapping[target]
            
        # 在缩点图上进行拓扑排序
        if source:
            start_nodes = {source}
        else:
            start_nodes = {n for n in C.nodes() if C.in_degree(n) == 0}
            
        if target:
            end_nodes = {target}
        else:
            end_nodes = {n for n in C.nodes() if C.out_degree(n) == 0}
    else:
        # 环图直接处理
        C = G
        scc_nodes = None
        
        if source:
            start_nodes = {source}
        else:
            start_nodes = {n for n in C.nodes() if C.in_degree(n) == 0}
            
        if target:
            end_nodes = {target}
        else:
            end_nodes = {n for n in C.nodes() if C.out_degree(n) == 0}
        
    # 如果没有起始节点或终止节点，返回空列表
    if not start_nodes or not end_nodes:
        logger.debug("No valid start or end nodes found for topological sort")
        return [], {}
        
    # 获取所有可能的拓扑排序路径
    paths = _all_paths_topological_sort(C, start_nodes, end_nodes)
    
    # 如果进行了缩点处理，展开路径
    if has_cycle and paths:
        expanded_paths = [_expand_path(path, scc_nodes) for path in paths]
        paths = expanded_paths
        
    if paths:
        logger.info(f"Found {len(paths)} topological sort paths")
        if len(paths) == max_paths:
            logger.info(f"Results limited to first {max_paths} paths")
        if has_cycle:
            logger.info(f"Detected {len(scc_nodes)} strongly connected components")
    else:
        logger.info("No valid topological sort paths found")
            
    return paths, scc_nodes if has_cycle else {}