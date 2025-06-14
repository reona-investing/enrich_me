import os
import ast
from pathlib import Path

BASE_DIR = Path('project/modules')
DOC_BASE = Path('class_specs')

for root, _, files in os.walk(BASE_DIR):
    classes_per_file = {}
    for file in files:
        if not file.endswith('.py'):
            continue
        path = Path(root) / file
        try:
            source = path.read_text(encoding='utf-8')
        except Exception:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        classes = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                cls_doc = ast.get_docstring(node) or ''
                methods = []
                for n in node.body:
                    if isinstance(n, ast.FunctionDef):
                        method_doc = ast.get_docstring(n) or ''
                        methods.append((n.name, method_doc))
                classes.append({'name': node.name, 'doc': cls_doc, 'methods': methods})
        if classes:
            classes_per_file[file] = classes
    if not classes_per_file:
        continue
    rel_dir = Path(root).relative_to(BASE_DIR)
    doc_dir = DOC_BASE / rel_dir
    doc_dir.mkdir(parents=True, exist_ok=True)
    doc_file = doc_dir / 'README.md'
    with open(doc_file, 'w', encoding='utf-8') as f:
        f.write(f'# {rel_dir} のクラス仕様書\n\n')
        for file_name, classes in sorted(classes_per_file.items()):
            f.write(f'## {file_name}\n\n')
            for cls in classes:
                f.write(f'### class {cls["name"]}\n')
                if cls['doc']:
                    f.write(cls['doc'] + '\n')
                for m_name, m_doc in cls['methods']:
                    f.write(f'- {m_name}: {m_doc}\n')
                f.write('\n')
print('Documentation generated in', DOC_BASE)
