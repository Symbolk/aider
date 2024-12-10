import colorsys
import math
import os
import random
import shutil
import sqlite3
import sys
import time
import warnings
from collections import Counter, defaultdict, namedtuple
from importlib import resources
from pathlib import Path

from diskcache import Cache
from grep_ast import TreeContext, filename_to_lang
from litellm import completion
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from tqdm import tqdm

from aider.dump import dump
from aider.io import InputOutput
from aider.models import Model
from aider.special import filter_important_files
from aider.utils import Spinner

import matplotlib.pyplot as plt
import networkx as nx

# tree_sitter is throwing a FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)
from tree_sitter_languages import get_language, get_parser  # noqa: E402

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
        "zh": "基于这些文件描述，请使用一句 中文简要概述这个代码社区/模块的用途：\n"
    }
}

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

Tag = namedtuple("Tag", "rel_fname fname line name kind".split())


SQLITE_ERRORS = (sqlite3.OperationalError, sqlite3.DatabaseError, OSError)


class RepoMap:
    CACHE_VERSION = 3
    TAGS_CACHE_DIR = f".aider.tags.cache.v{CACHE_VERSION}"

    warned_files = set()

    def __init__(
        self,
        map_tokens=1024,
        root=None,
        main_model=None,
        io=None,
        repo_content_prefix=None,
        verbose=False,
        max_context_window=None,
        map_mul_no_files=8,
        refresh="auto",
    ):
        self.io = io
        self.verbose = verbose
        self.refresh = refresh

        if not root:
            root = os.getcwd()
        self.root = root

        self.load_tags_cache()
        self.cache_threshold = 0.95

        self.max_map_tokens = map_tokens
        self.map_mul_no_files = map_mul_no_files
        self.max_context_window = max_context_window

        self.repo_content_prefix = repo_content_prefix

        self.main_model = main_model

        self.tree_cache = {}
        self.tree_context_cache = {}
        self.map_cache = {}
        self.map_processing_time = 0
        self.last_map = None

        if self.verbose:
            self.io.tool_output(
                f"RepoMap initialized with map_mul_no_files: {self.map_mul_no_files}"
            )

    def token_count(self, text):
        len_text = len(text)
        if len_text < 200:
            return self.main_model.token_count(text)

        lines = text.splitlines(keepends=True)
        num_lines = len(lines)
        step = num_lines // 100 or 1
        lines = lines[::step]
        sample_text = "".join(lines)
        sample_tokens = self.main_model.token_count(sample_text)
        est_tokens = sample_tokens / len(sample_text) * len_text
        return est_tokens

    def get_repo_map(
        self,
        chat_files,
        other_files,
        mentioned_fnames=None,
        mentioned_idents=None,
        force_refresh=False,
    ):
        if self.max_map_tokens <= 0:
            return
        if not other_files:
            return
        if not mentioned_fnames:
            mentioned_fnames = set()
        if not mentioned_idents:
            mentioned_idents = set()

        max_map_tokens = self.max_map_tokens

        # With no files in the chat, give a bigger view of the entire repo
        padding = 4096
        if max_map_tokens and self.max_context_window:
            target = min(
                int(max_map_tokens * self.map_mul_no_files),
                self.max_context_window - padding,
            )
        else:
            target = 0
        if not chat_files and self.max_context_window and target > 0:
            max_map_tokens = target

        try:
            files_listing = self.get_ranked_tags_map(
                chat_files,
                other_files,
                max_map_tokens,
                mentioned_fnames,
                mentioned_idents,
                force_refresh,
            )
        except RecursionError:
            self.io.tool_error("Disabling repo map, git repo too large?")
            self.max_map_tokens = 0
            return

        if not files_listing:
            return

        if self.verbose:
            num_tokens = self.token_count(files_listing)
            self.io.tool_output(f"Repo-map: {num_tokens / 1024:.1f} k-tokens")

        if chat_files:
            other = "other "
        else:
            other = ""

        if self.repo_content_prefix:
            repo_content = self.repo_content_prefix.format(other=other)
        else:
            repo_content = ""

        repo_content += files_listing

        return repo_content

    def get_rel_fname(self, fname):
        try:
            return os.path.relpath(fname, self.root)
        except ValueError:
            # Issue #1288: ValueError: path is on mount 'C:', start on mount 'D:'
            # Just return the full fname.
            return fname

    def tags_cache_error(self, original_error=None):
        """Handle SQLite errors by trying to recreate cache, falling back to dict if needed"""

        if self.verbose and original_error:
            self.io.tool_warning(f"Tags cache error: {str(original_error)}")

        if isinstance(getattr(self, "TAGS_CACHE", None), dict):
            return

        path = Path(self.root) / self.TAGS_CACHE_DIR

        # Try to recreate the cache
        try:
            # Delete existing cache dir
            if path.exists():
                shutil.rmtree(path)

            # Try to create new cache
            new_cache = Cache(path)

            # Test that it works
            test_key = "test"
            new_cache[test_key] = "test"
            _ = new_cache[test_key]
            del new_cache[test_key]

            # If we got here, the new cache works
            self.TAGS_CACHE = new_cache
            return

        except SQLITE_ERRORS as e:
            # If anything goes wrong, warn and fall back to dict
            self.io.tool_warning(
                f"Unable to use tags cache at {path}, falling back to memory cache"
            )
            if self.verbose:
                self.io.tool_warning(f"Cache recreation error: {str(e)}")

        self.TAGS_CACHE = dict()

    def load_tags_cache(self):
        path = Path(self.root) / self.TAGS_CACHE_DIR
        try:
            self.TAGS_CACHE = Cache(path)
        except SQLITE_ERRORS as e:
            self.tags_cache_error(e)

    def save_tags_cache(self):
        pass

    def get_mtime(self, fname):
        try:
            return os.path.getmtime(fname)
        except FileNotFoundError:
            self.io.tool_warning(f"File not found error: {fname}")

    def get_tags(self, fname, rel_fname):
        # Check if the file is in the cache and if the modification time has not changed
        file_mtime = self.get_mtime(fname)
        if file_mtime is None:
            return []

        cache_key = fname
        try:
            val = self.TAGS_CACHE.get(cache_key)  # Issue #1308
        except SQLITE_ERRORS as e:
            self.tags_cache_error(e)
            val = self.TAGS_CACHE.get(cache_key)

        if val is not None and val.get("mtime") == file_mtime:
            try:
                return self.TAGS_CACHE[cache_key]["data"]
            except SQLITE_ERRORS as e:
                self.tags_cache_error(e)
                return self.TAGS_CACHE[cache_key]["data"]

        # miss!
        data = list(self.get_tags_raw(fname, rel_fname))

        # Update the cache
        try:
            self.TAGS_CACHE[cache_key] = {"mtime": file_mtime, "data": data}
            self.save_tags_cache()
        except SQLITE_ERRORS as e:
            self.tags_cache_error(e)
            self.TAGS_CACHE[cache_key] = {"mtime": file_mtime, "data": data}

        return data

    def get_tags_raw(self, fname, rel_fname):
        lang = filename_to_lang(fname)
        if not lang:
            return

        try:
            language = get_language(lang)
            parser = get_parser(lang)
        except Exception as err:
            print(f"Skipping file {fname}: {err}")
            return

        query_scm = get_scm_fname(lang)
        if not query_scm.exists():
            return
        query_scm = query_scm.read_text()

        code = self.io.read_text(fname)
        if not code:
            return
        tree = parser.parse(bytes(code, "utf-8"))

        # Run the tags queries
        query = language.query(query_scm)
        captures = query.captures(tree.root_node)

        captures = list(captures)

        saw = set()
        for node, tag in captures:
            if tag.startswith("name.definition."):
                kind = "def"
            elif tag.startswith("name.reference."):
                kind = "ref"
            else:
                continue

            saw.add(kind)

            result = Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=node.text.decode("utf-8"),
                kind=kind,
                line=node.start_point[0],
            )

            yield result

        if "ref" in saw:
            return
        if "def" not in saw:
            return

        # We saw defs, without any refs
        # Some tags files only provide defs (cpp, for example)
        # Use pygments to backfill refs

        try:
            lexer = guess_lexer_for_filename(fname, code)
        except Exception:  # On Windows, bad ref to time.clock which is deprecated?
            # self.io.tool_error(f"Error lexing {fname}")
            return

        tokens = list(lexer.get_tokens(code))
        tokens = [token[1] for token in tokens if token[0] in Token.Name]

        for token in tokens:
            yield Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=token,
                kind="ref",
                line=-1,
            )

    def get_ranked_tags(
        self,
        chat_fnames,
        other_fnames,
        mentioned_fnames,
        mentioned_idents,
        progress=None,
    ):
        import networkx as nx

        defines = defaultdict(set)
        references = defaultdict(list)
        definitions = defaultdict(set)

        personalization = dict()

        fnames = set(chat_fnames).union(set(other_fnames))
        chat_rel_fnames = set()

        fnames = sorted(fnames)

        # Default personalization for unspecified files is 1/num_nodes
        # https://networkx.org/documentation/stable/_modules/networkx/algorithms/link_analysis/pagerank_alg.html#pagerank
        personalize = 100 / len(fnames)

        try:
            cache_size = len(self.TAGS_CACHE)
        except SQLITE_ERRORS as e:
            self.tags_cache_error(e)
            cache_size = len(self.TAGS_CACHE)

        if len(fnames) - cache_size > 100:
            self.io.tool_output(
                "Initial repo scan can be slow in larger repos, but only happens once."
            )
            fnames = tqdm(fnames, desc="Scanning repo")
            showing_bar = True
        else:
            showing_bar = False

        for fname in fnames:
            if self.verbose:
                self.io.tool_output(f"Processing {fname}")
            if progress and not showing_bar:
                progress()

            try:
                file_ok = Path(fname).is_file()
            except OSError:
                file_ok = False

            if not file_ok:
                if fname not in self.warned_files:
                    self.io.tool_warning(f"Repo-map can't include {fname}")
                    self.io.tool_output(
                        "Has it been deleted from the file system but not from git?"
                    )
                    self.warned_files.add(fname)
                continue

            # dump(fname)
            rel_fname = self.get_rel_fname(fname)

            if fname in chat_fnames:
                personalization[rel_fname] = personalize
                chat_rel_fnames.add(rel_fname)

            if rel_fname in mentioned_fnames:
                personalization[rel_fname] = personalize

            tags = list(self.get_tags(fname, rel_fname))
            if tags is None:
                continue

            for tag in tags:
                if tag.kind == "def":
                    defines[tag.name].add(rel_fname)
                    key = (rel_fname, tag.name)
                    definitions[key].add(tag)

                elif tag.kind == "ref":
                    references[tag.name].append(rel_fname)

        ##
        # dump(defines)
        # dump(references)
        # dump(personalization)

        if not references:
            references = dict((k, list(v)) for k, v in defines.items())

        idents = set(defines.keys()).intersection(set(references.keys()))

        G = nx.MultiDiGraph()

        for ident in idents:
            if progress:
                progress()

            definers = defines[ident]
            if ident in mentioned_idents:
                mul = 10
            elif ident.startswith("_"):
                mul = 0.1
            else:
                mul = 1

            for referencer, num_refs in Counter(references[ident]).items():
                for definer in definers:
                    # dump(referencer, definer, num_refs, mul)
                    # if referencer == definer:
                    #    continue

                    # scale down so high freq (low value) mentions don't dominate
                    num_refs = math.sqrt(num_refs)

                    G.add_edge(referencer, definer, weight=mul * num_refs, ident=ident)

        save_mermaid_diagram(self.root, G)

        communities = self.detect_communities(G)
        community_infos = self.generate_community_descriptions(G, communities, lang='zh')
        # 查看结果
        if self.verbose and community_infos:
            for community_id, info in community_infos.items():
                self.io.tool_output(f"\nCommunity {community_id}:")
                self.io.tool_output(f"Description: {info.description}")
                self.io.tool_output("Files:")
                for file in info.files:
                    self.io.tool_output(f"- {file}: {info.file_descriptions.get(file, '')}")


        if not references:
            pass

        if personalization:
            pers_args = dict(personalization=personalization, dangling=personalization)
        else:
            pers_args = dict()

        try:
            ranked = nx.pagerank(G, weight="weight", **pers_args)
        except ZeroDivisionError:
            try:
                ranked = nx.pagerank(G, weight="weight")
            except ZeroDivisionError:
                return []

        # distribute the rank from each source node, across all of its out edges
        ranked_definitions = defaultdict(float)
        for src in G.nodes:
            if progress:
                progress()

            src_rank = ranked[src]
            total_weight = sum(data["weight"] for _src, _dst, data in G.out_edges(src, data=True))
            # dump(src, src_rank, total_weight)
            for _src, dst, data in G.out_edges(src, data=True):
                data["rank"] = src_rank * data["weight"] / total_weight
                ident = data["ident"]
                ranked_definitions[(dst, ident)] += data["rank"]

        ranked_tags = []
        ranked_definitions = sorted(
            ranked_definitions.items(), reverse=True, key=lambda x: (x[1], x[0])
        )

        # dump(ranked_definitions)

        for (fname, ident), rank in ranked_definitions:
            # print(f"{rank:.03f} {fname} {ident}")
            if fname in chat_rel_fnames:
                continue
            ranked_tags += list(definitions.get((fname, ident), []))

        rel_other_fnames_without_tags = set(self.get_rel_fname(fname) for fname in other_fnames)

        fnames_already_included = set(rt[0] for rt in ranked_tags)

        top_rank = sorted([(rank, node) for (node, rank) in ranked.items()], reverse=True)
        for rank, fname in top_rank:
            if fname in rel_other_fnames_without_tags:
                rel_other_fnames_without_tags.remove(fname)
            if fname not in fnames_already_included:
                ranked_tags.append((fname,))

        for fname in rel_other_fnames_without_tags:
            ranked_tags.append((fname,))

        return ranked_tags

    def get_ranked_tags_map(
        self,
        chat_fnames,
        other_fnames=None,
        max_map_tokens=None,
        mentioned_fnames=None,
        mentioned_idents=None,
        force_refresh=False,
    ):
        # Create a cache key
        cache_key = [
            tuple(sorted(chat_fnames)) if chat_fnames else None,
            tuple(sorted(other_fnames)) if other_fnames else None,
            max_map_tokens,
        ]

        if self.refresh == "auto":
            cache_key += [
                tuple(sorted(mentioned_fnames)) if mentioned_fnames else None,
                tuple(sorted(mentioned_idents)) if mentioned_idents else None,
            ]
        cache_key = tuple(cache_key)

        use_cache = False
        if not force_refresh:
            if self.refresh == "manual" and self.last_map:
                return self.last_map

            if self.refresh == "always":
                use_cache = False
            elif self.refresh == "files":
                use_cache = True
            elif self.refresh == "auto":
                use_cache = self.map_processing_time > 1.0

            # Check if the result is in the cache
            if use_cache and cache_key in self.map_cache:
                return self.map_cache[cache_key]

        # If not in cache or force_refresh is True, generate the map
        start_time = time.time()
        result = self.get_ranked_tags_map_uncached(
            chat_fnames, other_fnames, max_map_tokens, mentioned_fnames, mentioned_idents
        )
        end_time = time.time()
        self.map_processing_time = end_time - start_time

        # Store the result in the cache
        self.map_cache[cache_key] = result
        self.last_map = result

        return result

    def get_ranked_tags_map_uncached(
        self,
        chat_fnames,
        other_fnames=None,
        max_map_tokens=None,
        mentioned_fnames=None,
        mentioned_idents=None,
    ):
        if not other_fnames:
            other_fnames = list()
        if not max_map_tokens:
            max_map_tokens = self.max_map_tokens
        if not mentioned_fnames:
            mentioned_fnames = set()
        if not mentioned_idents:
            mentioned_idents = set()

        spin = Spinner("Updating repo map")

        ranked_tags = self.get_ranked_tags(
            chat_fnames,
            other_fnames,
            mentioned_fnames,
            mentioned_idents,
            progress=spin.step,
        )

        other_rel_fnames = sorted(set(self.get_rel_fname(fname) for fname in other_fnames))
        special_fnames = filter_important_files(other_rel_fnames)
        ranked_tags_fnames = set(tag[0] for tag in ranked_tags)
        special_fnames = [fn for fn in special_fnames if fn not in ranked_tags_fnames]
        special_fnames = [(fn,) for fn in special_fnames]

        ranked_tags = special_fnames + ranked_tags

        spin.step()

        num_tags = len(ranked_tags)
        lower_bound = 0
        upper_bound = num_tags
        best_tree = None
        best_tree_tokens = 0

        chat_rel_fnames = set(self.get_rel_fname(fname) for fname in chat_fnames)

        self.tree_cache = dict()

        middle = min(max_map_tokens // 25, num_tags)
        while lower_bound <= upper_bound:
            # dump(lower_bound, middle, upper_bound)

            spin.step()

            tree = self.to_tree(ranked_tags[:middle], chat_rel_fnames)
            num_tokens = self.token_count(tree)

            pct_err = abs(num_tokens - max_map_tokens) / max_map_tokens
            ok_err = 0.15
            if (num_tokens <= max_map_tokens and num_tokens > best_tree_tokens) or pct_err < ok_err:
                best_tree = tree
                best_tree_tokens = num_tokens

                if pct_err < ok_err:
                    break

            if num_tokens < max_map_tokens:
                lower_bound = middle + 1
            else:
                upper_bound = middle - 1

            middle = (lower_bound + upper_bound) // 2

        spin.end()
        return best_tree

    tree_cache = dict()

    def render_tree(self, abs_fname, rel_fname, lois):
        mtime = self.get_mtime(abs_fname)
        key = (rel_fname, tuple(sorted(lois)), mtime)

        if key in self.tree_cache:
            return self.tree_cache[key]

        if (
            rel_fname not in self.tree_context_cache
            or self.tree_context_cache[rel_fname]["mtime"] != mtime
        ):
            code = self.io.read_text(abs_fname) or ""
            if not code.endswith("\n"):
                code += "\n"

            context = TreeContext(
                rel_fname,
                code,
                color=False,
                line_number=False,
                child_context=False,
                last_line=False,
                margin=0,
                mark_lois=False,
                loi_pad=0,
                # header_max=30,
                show_top_of_file_parent_scope=False,
            )
            self.tree_context_cache[rel_fname] = {"context": context, "mtime": mtime}

        context = self.tree_context_cache[rel_fname]["context"]
        context.lines_of_interest = set()
        context.add_lines_of_interest(lois)
        context.add_context()
        res = context.format()
        self.tree_cache[key] = res
        return res

    def to_tree(self, tags, chat_rel_fnames):
        if not tags:
            return ""

        cur_fname = None
        cur_abs_fname = None
        lois = None
        output = ""

        # add a bogus tag at the end so we trip the this_fname != cur_fname...
        dummy_tag = (None,)
        for tag in sorted(tags) + [dummy_tag]:
            this_rel_fname = tag[0]
            if this_rel_fname in chat_rel_fnames:
                continue

            # ... here ... to output the final real entry in the list
            if this_rel_fname != cur_fname:
                if lois is not None:
                    output += "\n"
                    output += cur_fname + ":\n"
                    output += self.render_tree(cur_abs_fname, cur_fname, lois)
                    lois = None
                elif cur_fname:
                    output += "\n" + cur_fname + "\n"
                if type(tag) is Tag:
                    lois = []
                    cur_abs_fname = tag.fname
                cur_fname = this_rel_fname

            if lois is not None:
                lois.append(tag.line)

        # truncate long lines, in case we get minified js or something else crazy
        output = "\n".join([line[:100] for line in output.splitlines()]) + "\n"

        return output

    def detect_communities(self, G):
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
            if self.verbose:
                self.io.tool_warning("python-louvain package not installed, community detection disabled")
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

    def topological_sort_paths(self, G, source=None, target=None, max_paths=100):
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
            if self.verbose:
                self.io.tool_output("Graph contains cycles, applying Tarjan's algorithm for condensation")
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
            if self.verbose:
                self.io.tool_warning("No valid start or end nodes found for topological sort")
            return [], {}
            
        # 获取所有可能的拓扑排序路径
        paths = _all_paths_topological_sort(C, start_nodes, end_nodes)
        
        # 如果进行了缩点处理，展开路径
        if has_cycle and paths:
            expanded_paths = [_expand_path(path, scc_nodes) for path in paths]
            paths = expanded_paths
            
        if self.verbose:
            if paths:
                self.io.tool_output(f"Found {len(paths)} topological sort paths")
                if len(paths) == max_paths:
                    self.io.tool_output(f"Results limited to first {max_paths} paths")
                if has_cycle:
                    self.io.tool_output(f"Detected {len(scc_nodes)} strongly connected components")
            else:
                self.io.tool_output("No valid topological sort paths found")
                
        return paths, scc_nodes if has_cycle else {}

    def generate_description(self, content, system_prompt, task_prompt):
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
            if self.verbose:
                self.io.tool_warning(f"Failed to generate description: {e}")
                self.io.tool_warning(f"Error details: {str(e)}")
            return ""

    def get_file_content(self, fname):
        """获取文件内容"""
        try:
            # 尝试将相对路径转换为绝对路径
            abs_path = os.path.join(self.root, fname) if not os.path.isabs(fname) else fname
            return self.io.read_text(abs_path)
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to read file {fname}: {e}")
            return ""
    
    def process_file_recursively(self, community_graph, node, visited, descriptions, lang):
        """递归处理文件，生成描述"""
        if node in visited:
            return descriptions.get(node, "")
            
        visited.add(node)
        content = self.get_file_content(node)
        
        # 获取依赖于当前文件的其他文件
        dependent_files = list(community_graph.predecessors(node))
        dependent_descriptions = []
        
        # 递归处理依赖文件
        for dep in dependent_files:
            desc = self.process_file_recursively(community_graph, dep, visited, descriptions, lang)
            if desc:
                dependent_descriptions.append(desc)
                
        # 生成当前文件的描述
        system_prompt = PROMPTS["system_prompt"][lang]
        prompt = PROMPTS["file_overview"][lang]
        if dependent_descriptions:
            prompt = prompt + PROMPTS["file_with_dependencies"][lang]
            prompt += "\n".join(f"- {desc}" for desc in dependent_descriptions)
            
        description = self.generate_description(system_prompt, content, prompt)
        descriptions[node] = description
        return description

    def generate_community_descriptions(self, G, communities, lang='en'):
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
            if self.verbose:
                self.io.tool_warning("litellm package not installed, community description disabled")
            return None
            
        class CommunityInfo:
            def __init__(self):
                self.files = set()  # 社区包含的文件
                self.file_descriptions = {}  # 文件的描述，key是文件路径
                self.description = ""  # 社区整体描述
                self.dependency_paths = []  # 社区内的依赖路径
                
        # 验证语言参数
        if lang not in ['en', 'zh']:
            if self.verbose:
                self.io.tool_warning(f"Invalid language '{lang}', falling back to English")
            lang = 'en'
                
        community_infos = {}
        
        for i, community in communities.items():
            if self.verbose:
                self.io.tool_output(f"Processing community {i+1}...")
                
            info = CommunityInfo()
            info.files = community
            
            # 创建社区子图
            community_graph = G.subgraph(community)
            
            # 获取社区内的拓扑排序路径
            paths = self.topological_sort_paths(community_graph)
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
                self.process_file_recursively(community_graph, end_node, visited, descriptions, lang)
                
            info.file_descriptions = descriptions
            
            # 生成社区整体描述
            if descriptions:
                all_descriptions = "\n".join(descriptions.values())
                system_prompt = PROMPTS["system_prompt"][lang]
                prompt = PROMPTS["community_overview"][lang] + all_descriptions
                info.description = self.generate_description(system_prompt, "", prompt)
                
            community_infos[i] = info
            
            if self.verbose:
                self.io.tool_output(f"Community {i+1} description: {info.description}")
                
        return community_infos


def find_src_files(directory):
    if not os.path.isdir(directory):
        return [directory]

    src_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            src_files.append(os.path.join(root, file))
    return src_files


def get_random_color():
    hue = random.random()
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1, 0.75)]
    res = f"#{r:02x}{g:02x}{b:02x}"
    return res


def get_scm_fname(lang):
    """获取对应语言的tree-sitter查询文件路径"""
    try:
        # 获取当前文件所在目录
        current_dir = Path(__file__).parent
        # 构建查询文件的路径
        query_path = current_dir / "queries" / f"tree-sitter-{lang}-tags.scm"
        if query_path.exists():
            return query_path
        return None
    except Exception as e:
        print(f"Error loading query file for language {lang}: {e}")
        return None


def get_supported_languages_md():
    from grep_ast.parsers import PARSERS

    res = """
| Language | File extension | Repo map | Linter |
|:--------:|:--------------:|:--------:|:------:|
"""
    data = sorted((lang, ex) for ex, lang in PARSERS.items())

    for lang, ext in data:
        fn = get_scm_fname(lang)
        repo_map = "✓" if Path(fn).exists() else ""
        linter_support = "✓"
        res += f"| {lang:20} | {ext:20} | {repo_map:^8} | {linter_support:^6} |\n"

    res += "\n"

    return res


if __name__ == "__main__":
    # 支持的文件扩展名(根据queries文件夹中的tree-sitter查询文件)
    extensions = {
        # 系统/底层编程语言
        'C/C++': {'.c', '.h', '.cpp', '.hpp', '.cc', '.cxx', '.hxx'},
        'Rust': {'.rs'},
        'Go': {'.go'},
        
        # JVM语言
        'Java': {'.java'},
        
        # .NET语言
        'C#': {'.cs'},
        
        # 脚本/动态语言
        'Python': {'.py'},
        'Ruby': {'.rb'},
        'PHP': {'.php'},
        
        # Web开发语言
        'JavaScript': {'.js', '.jsx', '.mjs', '.cjs'},
        'TypeScript': {'.ts', '.tsx'},
        
        # 函数式编程语言
        'OCaml': {'.ml', '.mli'},
        'Elisp': {'.el'},
        'Elixir': {'.ex', '.exs'},
        'Elm': {'.elm'},
        
        # 移动开发语言
        'Dart': {'.dart'},
        
        # 查询语言
        'QL': {'.ql', '.qll'},
    }
   
    # 需要忽略的目录(常见的非项目源码目录)
    dirs_to_ignore = {
        # 包管理和依赖
        'node_modules',
        'vendor',
        'packages',
        'bower_components',
        '.npm',
        
        # Python相关
        'venv',
        '.venv',
        'env',
        '.env',
        '__pycache__',
        '.pytest_cache',
        '.mypy_cache',
        '.ruff_cache',
        'dist',
        'build',
        'eggs',
        '.eggs',
        
        # IDE和编辑器
        '.idea',
        '.vscode',
        '.vs',
        '.eclipse',
        
        # Git相关
        '.git',
        
        # 其他构建和缓存目录
        'target',      # Rust, Java等
        'bin',
        'obj',        # .NET
        '.gradle',    # Gradle
        '.m2',        # Maven
        'Debug',
        'Release',
        '.dart_tool', # Dart
        '.pub-cache', # Pub (Dart)
        'elm-stuff',  # Elm
        '_build',     # OCaml
        '.bundle',    # Ruby
        'coverage',   # 测试覆盖率报告
    }

    if len(sys.argv) < 2:
        print("Usage: python repomap.py <directory or files>")
        sys.exit(1)

    # 使用第一个参数作为root目录
    repo_root = sys.argv[1] if Path(sys.argv[1]).is_dir() else "."

    fnames = []

    extensions_to_include = extensions['Python']
    for fname in sys.argv[1:]:
        if Path(fname).is_dir():
            # 遍历目录下的所有文件
            for root, dirs, files in os.walk(fname):
                # 从dirs列表中移除需要忽略的目录
                dirs[:] = [d for d in dirs if d not in dirs_to_ignore]
                
                for file in files:
                    file_path = Path(root) / file
                    # 只包含指定扩展名的文件
                    if file_path.suffix.lower() in extensions_to_include:
                        fnames.append(str(file_path))
        else:
            # 对于直接指定的文件，检查扩展名
            if Path(fname).suffix.lower() in extensions_to_include:
                fnames.append(fname)

    chat_fnames = fnames[:5]
    other_fnames = fnames[5:]
    mentioned_fnames = fnames[1]

    rm = RepoMap(root=repo_root, main_model=Model('gpt-4o'), io=InputOutput(), verbose=True)
    repo_map = rm.get_ranked_tags_map(chat_fnames, other_fnames, max_map_tokens=16000, mentioned_fnames=mentioned_fnames, mentioned_idents=[])

    if repo_map:
        print(f"Build {len(repo_map.splitlines())} lines in repo map")
    else:
        print("No files found matching the supported extensions")