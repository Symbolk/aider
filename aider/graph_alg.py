from collections import defaultdict
from loguru import logger
import math
import json
import os
from collections import defaultdict

from aider.utils import get_aidoc_dir


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
        """展开缩点后的路径，每个SCC节点替换为其包含的原始节点"""
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

def save_d3_visualization(repo_root, G, communities=None, output_file_name=None):
    """
    将依赖图保存为可交互的d3.js可视化HTML文件
    
    Args:
        repo_root: str 仓库根目录
        G: networkx.MultiDiGraph 代码依赖关系图
        communities: dict 社区检测结果，key是社区id，value是文件集合
        output_file_name: str 输出文件名，默认为 repo_overview.html
    """
    # 如果没有指定输出路径，使用默认路径
    if output_file_name is None:
        output_file_path = os.path.join(get_aidoc_dir(repo_root), "repo_overview.html")
    else:
        output_file_path = os.path.join(get_aidoc_dir(repo_root), output_file_name)

    # 简化文件名的函数
    def simplify_filename(fname):
        return os.path.basename(fname)
    
    # 准备节点数据
    nodes = []
    node_id_map = {}  # 用于将文件名映射到数字id
    
    # 计算每个节点的度
    node_degrees = defaultdict(int)
    for u, v in G.edges():
        node_degrees[u] += 1
        node_degrees[v] += 1
    
    # 过滤掉度数太小的节点（可选）
    min_degree = 2  # 可以调整这个阈值
    significant_nodes = {node for node, degree in node_degrees.items() if degree >= min_degree}
    
    # 生成社区颜色映射和描述
    community_colors = {}
    community_descriptions = {}
    if communities:
        # 预定义一些好看的颜色
        colors = [
            "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33",
            "#A65628", "#F781BF", "#999999", "#66C2A5", "#FC8D62", "#8DA0CB"
        ]
        for i, community_id in enumerate(communities.keys()):
            community_colors[community_id] = colors[i % len(colors)]
            # 获取社区描述
            community_obj = communities[community_id]
            if isinstance(community_obj, set):
                community_descriptions[community_id] = f"Community {community_id}"
            else:  # CommunityInfo object
                community_descriptions[community_id] = community_obj.description or f"Community {community_id}"

    # 添加文件节点
    for i, node in enumerate(G.nodes()):
        if node in significant_nodes:
            # 获取文件描述
            file_description = None
            if communities:
                for comm_id, comm_obj in communities.items():
                    if isinstance(comm_obj, set):
                        if node in comm_obj:
                            file_description = G.nodes[node].get('description', '')
                            break
                    else:  # CommunityInfo object
                        if node in comm_obj.files:
                            file_description = comm_obj.file_descriptions.get(node, '')
                            break

            # 如果社区中没有文件描述，尝试从图节点属性中获取
            if not file_description:
                file_description = G.nodes[node].get('description', '')

            node_data = {
                "id": i,
                "name": simplify_filename(node),
                "full_name": node,
                "type": "file",
                "community_id": None,
                "color": "#666666",
                "degree": node_degrees[node],
                "radius": max(8, math.sqrt(node_degrees[node]) * 3),
                "description": file_description,
                "community_description": ""
            }
            
            # 如果节点属于某个社区，添加社区信息
            if communities:
                for comm_id, comm_obj in communities.items():
                    if isinstance(comm_obj, set):
                        if node in comm_obj:
                            node_data["community_id"] = comm_id
                            node_data["color"] = community_colors[comm_id]
                            node_data["community_description"] = community_descriptions[comm_id]
                            break
                    else:  # CommunityInfo object
                        if node in comm_obj.files:
                            node_data["community_id"] = comm_id
                            node_data["color"] = community_colors[comm_id]
                            node_data["community_description"] = comm_obj.description
                            node_data["description"] = comm_obj.file_descriptions.get(node, '')
                            break

            nodes.append(node_data)
            node_id_map[node] = i
    
    # 准备边数据，合并相同节点间的多条边
    edge_weights = defaultdict(float)
    edge_idents = defaultdict(set)
    
    # 设置边的权重阈值
    min_weight = 1.0  # 可以调整这个阈值
    
    for u, v, data in G.edges(data=True):
        if u not in significant_nodes or v not in significant_nodes:
            continue
        key = (node_id_map[u], node_id_map[v])
        weight = data.get("weight", 1)
        edge_weights[key] += weight
        if "ident" in data:
            edge_idents[key].add(data["ident"])
            
    # 转换为边列表，只保留权重大于阈值的边
    links = []
    for (source, target), weight in edge_weights.items():
        if weight >= min_weight:
            links.append({
                "source": source,
                "target": target,
                "weight": weight,
                "idents": list(edge_idents.get((source, target), set()))
            })
    
    # 准备图数据
    graph_data = {
        "nodes": nodes,
        "links": links
    }
    
    # 生成HTML模板
    html_template = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Code Dependencies Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        /* 添加禁用文本选择的样式类 */
        .no-select {
            -webkit-user-select: none;
            -moz-user-select: none;
            -ms-user-select: none;
            user-select: none;
        }
        
        body { margin: 0; display: flex; }
        #sidebar { 
            width: 300px; 
            height: 100vh; 
            background: white;
            border-right: 1px solid #e0e0e0;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            color: #333;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            flex-shrink: 0;
        }
        #resizer {
            width: 8px;
            height: 100vh;
            background: #f0f0f0;
            cursor: col-resize;
            flex-shrink: 0;
            transition: background 0.2s;
        }
        #resizer:hover, #resizer.active {
            background: #1a73e8;
        }
        #file-tree {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
        }
        #file-tree::-webkit-scrollbar {
            width: 8px;
        }
        #file-tree::-webkit-scrollbar-track {
            background: #f5f5f5;
        }
        #file-tree::-webkit-scrollbar-thumb {
            background: #cdcdcd;
            border-radius: 4px;
        }
        #file-tree::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
        }
        #file-content {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
            display: none;
            background: white;
            color: #333;
        }
        #graph { 
            flex: 1;
            height: 100vh;
            background: white;
            overflow: hidden;
        }
        .tree-node {
            cursor: pointer;
            padding: 4px 8px;
            border-radius: 3px;
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 13px;
            white-space: nowrap;
            transition: all 0.1s ease;
            color: #444;
            border: 1px solid transparent;
        }
        .tree-node:hover {
            background: #f0f0f0;
            border: 1px solid #e0e0e0;
        }
        .tree-node.active {
            background: #e8f0fe;
            border: 1px solid #cce0ff;
            color: #1a73e8;
        }
        .tree-node i {
            font-size: 14px;
            width: 16px;
            text-align: center;
        }
        .tree-node i.fa-folder {
            color: #ffd04c;  /* 更鲜艳的文件夹黄色 */
        }
        .tree-node i.fa-folder-open {
            color: #ffba00;  /* 打开状态的文件夹颜色 */
        }
        .tree-node i.fa-file {
            color: #42a5f5;  /* 默认文件图标使用蓝色 */
        }
        .tree-node i.fa-java {
            color: #f89820;  /* Java文件图标使用橙色 */
        }
        .tree-node i.fa-python {
            color: #3776ab;  /* Python文件图标使用蓝色 */
        }
        .tree-node i.fa-js {
            color: #f7df1e;  /* JavaScript文件图标使用黄色 */
        }
        .tree-node i.fa-html5 {
            color: #e34f26;  /* HTML文件图标使用红色 */
        }
        .tree-node i.fa-css3-alt {
            color: #1572b6;  /* CSS文件图标使用蓝色 */
        }
        .tree-node.folder i.fa-folder-open {
            display: none;
        }
        .tree-node.folder.expanded i.fa-folder {
            display: none;
        }
        .tree-node.folder.expanded i.fa-folder-open {
            display: inline;
        }
        .tree-node .node-content {
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .tree-children {
            margin-left: 12px;
            display: none;
            border-left: 1px solid #e6e6e6;
            margin-top: 4px;
            margin-bottom: 4px;
            padding-left: 8px;
        }
        .tree-children.expanded {
            display: block;
        }
        pre {
            margin: 0;
            padding: 16px;
            background: #f8f9fa;
            border-radius: 4px;
            overflow-x: auto;
            font-family: 'Fira Code', 'Consolas', 'Monaco', monospace;
            font-size: 13px;
            line-height: 1.5;
            border: 1px solid #e9ecef;
        }
        .back-button {
            padding: 8px 12px;
            margin: 10px;
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            cursor: pointer;
            color: #495057;
            display: none;
            font-size: 13px;
            transition: all 0.2s ease;
            align-items: center;
            gap: 6px;
        }
        .back-button:hover {
            background: #e9ecef;
            border-color: #ced4da;
            color: #212529;
        }
        .back-button i {
            font-size: 12px;
        }
        .node { cursor: pointer; }
        .node text { 
            font-family: Arial, sans-serif;
            font-size: 12px;
            fill: #333;
            text-anchor: middle;
            dominant-baseline: middle;
        }
        .link { 
            stroke: #999; 
            stroke-opacity: 0.6; 
        }
        .tooltip {
            position: absolute;
            padding: 12px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            border-radius: 6px;
            font-family: Arial, sans-serif;
            font-size: 14px;
            pointer-events: none;
            max-width: 400px;
            word-wrap: break-word;
            white-space: normal;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            line-height: 1.4;
        }
        .tooltip h4 {
            margin: 0 0 8px 0;
            font-size: 16px;
            border-bottom: 1px solid #666;
            padding-bottom: 5px;
        }
        .tooltip .path {
            word-break: break-all;
            margin-bottom: 8px;
        }
        .tooltip .description {
            font-style: italic;
            color: #ccc;
            margin-top: 8px;
        }
        /* 添加节点选中效果 */
        .node circle.selected {
            stroke: #1a73e8;
            stroke-width: 2px;
            filter: drop-shadow(0 0 3px rgba(26, 115, 232, 0.4));
        }
        .node text.selected {
            fill: #1a73e8;
            font-weight: bold;
        }
        /* 添加文件树高亮效果 */
        .tree-node.highlighted {
            background: #e8f0fe;
            border: 1px solid #cce0ff;
            color: #1a73e8;
            font-weight: 500;
        }
        .tree-node.highlighted i {
            color: #1a73e8;
        }
    </style>
</head>
<body>
    <div id="sidebar">
        <button class="back-button"><i class="fas fa-arrow-left"></i>返回文件树</button>
        <div id="file-tree"></div>
        <div id="file-content"></div>
    </div>
    <div id="resizer"></div>
    <div id="graph"></div>
    <script>
        const data = ''' + json.dumps({"nodes": nodes, "links": links}) + ''';
        
        // 创建文件树数据结构
        function createFileTree(files) {
            const root = { name: '', children: {} };
            files.forEach(file => {
                const parts = file.full_name.split('/');
                let current = root;
                parts.forEach((part, i) => {
                    if (!current.children[part]) {
                        current.children[part] = { 
                            name: part,
                            path: parts.slice(0, i + 1).join('/'),
                            children: {},
                            isFile: i === parts.length - 1,
                            description: i === parts.length - 1 ? file.description : ''
                        };
                    }
                    current = current.children[part];
                });
            });
            return root;
        }

        // 渲染文件树
        function renderFileTree(node, container, level = 0) {
            const items = Object.values(node.children).sort((a, b) => {
                if (a.isFile === b.isFile) return a.name.localeCompare(b.name);
                return a.isFile ? 1 : -1;
            });

            items.forEach(item => {
                const nodeDiv = document.createElement('div');
                nodeDiv.className = 'tree-node' + (item.isFile ? '' : ' folder');
                
                // 添加图标
                const icon = document.createElement('i');
                if (item.isFile) {
                    icon.className = 'fas fa-file';
                    // 根据文件扩展名设置不同的图标
                    const ext = item.name.split('.').pop().toLowerCase();
                    switch(ext) {
                        case 'java':
                            icon.className = 'fab fa-java';
                            break;
                        case 'py':
                            icon.className = 'fab fa-python';
                            break;
                        case 'js':
                            icon.className = 'fab fa-js';
                            break;
                        case 'html':
                            icon.className = 'fab fa-html5';
                            break;
                        case 'css':
                            icon.className = 'fab fa-css3-alt';
                            break;
                    }
                } else {
                    nodeDiv.innerHTML = `
                        <i class="fas fa-folder"></i>
                        <i class="fas fa-folder-open"></i>
                    `;
                }
                if (item.isFile) {
                    nodeDiv.appendChild(icon);
                }
                
                // 添加文件名
                const content = document.createElement('span');
                content.className = 'node-content';
                content.textContent = item.name;
                nodeDiv.appendChild(content);
                
                container.appendChild(nodeDiv);
                
                if (!item.isFile) {
                    const childrenDiv = document.createElement('div');
                    childrenDiv.className = 'tree-children';
                    container.appendChild(childrenDiv);
                    
                    // 添加文件夹点击展开/折叠事件
                    nodeDiv.onclick = (e) => {
                        e.stopPropagation();
                        nodeDiv.classList.toggle('expanded');
                        childrenDiv.classList.toggle('expanded');
                    };
                    
                    renderFileTree(item, childrenDiv, level + 1);
                } else {
                    // 修改文件节点的点击事件
                    nodeDiv.onclick = (e) => {
                        e.stopPropagation();
                        // 清除所有文件节点的高亮
                        document.querySelectorAll('.tree-node').forEach(node => {
                            node.classList.remove('highlighted');
                        });
                        // 添加当前节点的高亮
                        nodeDiv.classList.add('highlighted');
                        
                        // 高亮对应的d3节点
                        const fullPath = item.path;
                        highlightD3Node(fullPath);
                        
                        // 显示文件内容
                        showFileContent(fullPath);
                    };
                }
            });
        }

        // 添加高亮d3节点的函数
        function highlightD3Node(fullPath) {
            // 清除所有节点的选中状态
            node.selectAll("circle").classed("selected", false);
            node.selectAll("text").classed("selected", false);
            
            // 找到并高亮对应的节点
            node.each(function(d) {
                if (d.full_name === fullPath) {
                    const selectedNode = d3.select(this);
                    selectedNode.select("circle").classed("selected", true);
                    selectedNode.select("text").classed("selected", true);
                    
                    // 将节点移动到视图中心
                    const transform = d3.zoomTransform(svg.node());
                    const scale = transform.k;
                    const x = -d.x * scale + width / 2;
                    const y = -d.y * scale + height / 2;
                    
                    svg.transition()
                        .duration(750)
                        .call(
                            zoom.transform,
                            d3.zoomIdentity
                                .translate(x, y)
                                .scale(scale)
                        );
                }
            });
        }

        // 显示文件内容
        async function showFileContent(path) {
            const fileTree = document.getElementById('file-tree');
            const fileContent = document.getElementById('file-content');
            const backButton = document.querySelector('.back-button');
            
            try {
                const response = await fetch(path);
                const content = await response.text();
                
                const pre = document.createElement('pre');
                pre.textContent = content;
                
                fileContent.innerHTML = '';
                fileContent.appendChild(pre);
                
                fileTree.style.display = 'none';
                fileContent.style.display = 'block';
                backButton.style.display = 'flex';
                
                // 高亮当前文件在文件树中的节点
                document.querySelectorAll('.tree-node').forEach(node => {
                    node.classList.remove('active');
                    if (node.querySelector('.node-content').textContent === path.split('/').pop()) {
                        node.classList.add('active');
                    }
                });
            } catch (error) {
                console.error('Error loading file:', error);
                fileContent.innerHTML = `
                    <div style="padding: 16px; color: #f44336;">
                        <i class="fas fa-exclamation-circle"></i>
                        Error loading file: ${path}
                    </div>
                `;
            }
        }

        // 返回文件树视图
        document.querySelector('.back-button').onclick = () => {
            document.getElementById('file-tree').style.display = 'block';
            document.getElementById('file-content').style.display = 'none';
            document.querySelector('.back-button').style.display = 'none';
        };

        // 初始化文件树
        const fileTree = createFileTree(data.nodes);
        renderFileTree(fileTree, document.getElementById('file-tree'));

        /* 创建SVG */
        const width = document.getElementById('graph').offsetWidth;
        const height = window.innerHeight;
        const svg = d3.select("#graph")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
            
        /* 创建缩放和平移行为 */
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => {
                container.attr("transform", event.transform);
            });
            
        svg.call(zoom);
        
        const container = svg.append("g");
        
        /* 创建力导向图布局 */
        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links)
                .id(d => d.id)
                .distance(d => {
                    const nodeCount = data.nodes.length;
                    const linkCount = data.links.length;
                    const baseDist = 50;
                    const nodeRatio = nodeCount / 100;  // 每100个节点增加一倍基础距离
                    const linkRatio = linkCount / 500;  // 每500条边增加一倍基础距离
                    return baseDist * (1 + nodeRatio) * (1 + linkRatio);
                })
                .strength(0.2))  /* 减小link force的strength */
            .force("charge", d3.forceManyBody()
                .strength(d => {
                    const nodeCount = data.nodes.length;
                    const baseCharge = -100;
                    const chargeRatio = nodeCount / 100;  // 每100个节点增加一倍斥力
                    return baseCharge * (1 + chargeRatio);
                }))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("x", d3.forceX(width / 2).strength(0.05))
            .force("y", d3.forceY(height / 2).strength(0.05))
            .force("collide", d3.forceCollide()
                .radius(d => d.radius * 2.5)  /* 增加碰撞半径 */
                .strength(1)
                .iterations(3))
            .alphaDecay(0.01)  /* 减缓alpha衰减速度 */
            .velocityDecay(0.5);  /* 减缓速度衰减 */
            
        /* 绘制连接线 */
        const link = container.append("g")
            .selectAll("line")
            .data(data.links)
            .join("line")
            .attr("class", "link")
            .attr("stroke-width", d => Math.sqrt(d.weight));
            
        /* 创建节点组 */
        const node = container.append("g")
            .selectAll(".node")
            .data(data.nodes)
            .join("g")
            .attr("class", "node")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        /* 添加节点圆圈 */
        node.append("circle")
            .attr("r", d => d.radius)
            .attr("fill", d => d.color)
            .attr("stroke", "#fff")
            .attr("stroke-width", 1.5);

        /* 添加节点标签 */
        node.append("text")
            .text(d => d.name)
            .attr("x", 0)
            .attr("y", d => -d.radius - 5);

        /* 添加提示框 */
        const tooltip = d3.select("body").append("div")
            .attr("class", "tooltip")
            .style("opacity", 0);

        // 在文件树中定位到指定文件
        function locateFile(fullPath) {
            const parts = fullPath.split('/');
            let currentNode = null;
            let currentPath = '';
            
            // 清除所有高亮
            document.querySelectorAll('.tree-node').forEach(node => {
                node.classList.remove('highlighted');
            });
            
            // 展开所有父文件夹
            for (let i = 0; i < parts.length; i++) {
                currentPath += (i === 0 ? '' : '/') + parts[i];
                const treeNodes = document.querySelectorAll('.tree-node');
                
                for (const node of treeNodes) {
                    const content = node.querySelector('.node-content');
                    if (content && content.textContent === parts[i]) {
                        currentNode = node;
                        // 如果是文件夹且未展开，则展开它
                        if (i < parts.length - 1 && !node.classList.contains('expanded')) {
                            node.click();
                        }
                        // 如果是目标文件，添加高亮效果
                        if (i === parts.length - 1) {
                            node.classList.add('highlighted');
                        }
                        break;
                    }
                }
            }
            
            // 如果找到了目标文件节点
            if (currentNode) {
                // 滚动到该节点
                currentNode.scrollIntoView({ behavior: 'smooth', block: 'center' });
                // 触发点击事件以显示文件内容
                if (currentNode.classList.contains('highlighted')) {
                    showFileContent(fullPath);
                }
            }
        }

        /* 节点交互 */
        node.on("click", (event, d) => {
                event.stopPropagation(); // 阻止事件冒泡
                
                // 清除之前的选中状态
                node.selectAll("circle").classed("selected", false);
                node.selectAll("text").classed("selected", false);
                
                // 添加新的选中状态
                const clickedNode = d3.select(event.currentTarget);
                clickedNode.select("circle").classed("selected", true);
                clickedNode.select("text").classed("selected", true);
                
                locateFile(d.full_name);
            })
            .on("mouseover", (event, d) => {
                tooltip.transition()
                    .duration(200)
                    .style("opacity", .9);
                    
                let tooltipContent = `<h4>${d.name}</h4>`;
                tooltipContent += `<div class="path"><strong>Full path:</strong><br>${d.full_name}</div>`;
                tooltipContent += `<strong>Degree:</strong> ${d.degree}<br>`;
                
                if (d.community_id !== null) {
                    tooltipContent += `<div class="description"><strong>Community:</strong><br>${d.community_description}</div>`;
                }
                
                if (d.description) {
                    tooltipContent += `<div class="description"><strong>File:</strong><br>${d.description}</div>`;
                }
                
                tooltip.html(tooltipContent)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
            })
            .on("mouseout", () => {
                tooltip.transition()
                    .duration(500)
                    .style("opacity", 0);
            });

        /* 模拟tick事件 */
        simulation.on("tick", () => {
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node.attr("transform", d => `translate(${d.x},${d.y})`);
        });

        /* 拖拽函数 */
        function dragstarted(event) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }

        function dragged(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }

        function dragended(event) {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }

        // 处理窗口大小变化
        window.addEventListener('resize', () => {
            const newWidth = document.getElementById('graph').offsetWidth;
            svg.attr("width", newWidth);
            simulation.force("center", d3.forceCenter(newWidth / 2, height / 2))
                .alpha(0.3)
                .restart();
        });

        // 添加拖动调整宽度的功能
        const resizer = document.getElementById('resizer');
        const sidebar = document.getElementById('sidebar');
        const graph = document.getElementById('graph');
        let isResizing = false;
        let startX;
        let startWidth;

        // 禁用文本选择的函数
        function disableSelect() {
            document.body.classList.add('no-select');
            sidebar.style.pointerEvents = 'none';
            graph.style.pointerEvents = 'none';
        }

        // 启用文本选择的函数
        function enableSelect() {
            document.body.classList.remove('no-select');
            sidebar.style.pointerEvents = 'auto';
            graph.style.pointerEvents = 'auto';
        }

        resizer.addEventListener('mousedown', (e) => {
            isResizing = true;
            startX = e.pageX;
            startWidth = parseInt(window.getComputedStyle(sidebar).width, 10);
            resizer.classList.add('active');
            disableSelect();
            e.preventDefault(); // 防止拖动开始时的文本选择
        });

        document.addEventListener('mousemove', (e) => {
            if (!isResizing) return;
            
            const width = startWidth + (e.pageX - startX);
            // 限制最小宽度为200px，最大宽度为800px
            if (width >= 200 && width <= 800) {
                sidebar.style.width = `${width}px`;
                // 触发图表区域的resize事件
                window.dispatchEvent(new Event('resize'));
            }
            e.preventDefault(); // 防止拖动过程中的文本选择
        });

        document.addEventListener('mouseup', () => {
            if (!isResizing) return;
            
            isResizing = false;
            resizer.classList.remove('active');
            enableSelect();
        });

        // 确保在鼠标离开窗口时也能正确处理
        document.addEventListener('mouseleave', () => {
            if (isResizing) {
                isResizing = false;
                resizer.classList.remove('active');
                enableSelect();
            }
        });
    </script>
</body>
</html>'''

    # 写入HTML文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(html_template)