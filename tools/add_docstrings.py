"""
Docstring Generation Tool for MIMO Geometry Analysis Project

Author: Hossein Molhem
Date: November 12, 2025

This script scans Python files and generates Google-style docstrings
for all classes, methods, and functions.
"""

import ast
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime


AUTHOR = "Hossein Molhem"
CURRENT_YEAR = datetime.now().year

DOCSTRING_TEMPLATE_FUNCTION = '''"""
{summary}

Author: {author}

Args:
{args}

Returns:
{returns}

Raises:
{raises}
"""'''

DOCSTRING_TEMPLATE_CLASS = '''"""
{summary}

Author: {author}

Attributes:
{attributes}
"""'''


class DocstringGenerator:
    """
    Generate Google-style docstrings for Python code.
    
    Author: Hossein Molhem
    
    Attributes:
        author (str): Name of the code author
        exclude_dirs (List[str]): Directories to exclude from scanning
        exclude_patterns (List[str]): File patterns to exclude
    """
    
    def __init__(self, author: str = AUTHOR):
        """
        Initialize the docstring generator.
        
        Author: Hossein Molhem
        
        Args:
            author (str): Name of the author to include in docstrings
        """
        self.author = author
        self.exclude_dirs = [
            'mimo-geom-dev', '__pycache__', '.git', 'archives',
            'venv', 'env', '.venv', 'node_modules', 'build', 'dist'
        ]
        self.exclude_patterns = ['test_*.py', '*_test.py']
        
    def should_process_file(self, filepath: Path) -> bool:
        """
        Check if a file should be processed for docstrings.
        
        Author: Hossein Molhem
        
        Args:
            filepath (Path): Path to the Python file
            
        Returns:
            bool: True if file should be processed, False otherwise
        """
        # Check if path contains excluded directories
        for exclude_dir in self.exclude_dirs:
            if exclude_dir in filepath.parts:
                return False
                
        # Check if filename matches excluded patterns
        for pattern in self.exclude_patterns:
            if filepath.match(pattern):
                return False
                
        return filepath.suffix == '.py' and filepath.is_file()
    
    def extract_function_info(self, node: ast.FunctionDef) -> Dict:
        """
        Extract information from a function/method node.
        
        Author: Hossein Molhem
        
        Args:
            node (ast.FunctionDef): AST node representing a function
            
        Returns:
            Dict: Dictionary containing function metadata
        """
        info = {
            'name': node.name,
            'lineno': node.lineno,
            'args': [],
            'returns': None,
            'is_async': isinstance(node, ast.AsyncFunctionDef),
            'has_docstring': ast.get_docstring(node) is not None
        }
        
        # Extract arguments
        for arg in node.args.args:
            arg_name = arg.arg
            arg_type = ast.unparse(arg.annotation) if arg.annotation else 'Any'
            info['args'].append((arg_name, arg_type))
            
        # Extract return type
        if node.returns:
            info['returns'] = ast.unparse(node.returns)
        else:
            info['returns'] = 'Any'
            
        return info
    
    def extract_class_info(self, node: ast.ClassDef) -> Dict:
        """
        Extract information from a class node.
        
        Author: Hossein Molhem
        
        Args:
            node (ast.ClassDef): AST node representing a class
            
        Returns:
            Dict: Dictionary containing class metadata
        """
        info = {
            'name': node.name,
            'lineno': node.lineno,
            'bases': [ast.unparse(base) for base in node.bases],
            'methods': [],
            'attributes': [],
            'has_docstring': ast.get_docstring(node) is not None
        }
        
        # Extract methods
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                info['methods'].append(self.extract_function_info(item))
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                # Class attribute with type annotation
                attr_name = item.target.id
                attr_type = ast.unparse(item.annotation) if item.annotation else 'Any'
                info['attributes'].append((attr_name, attr_type))
                
        return info
    
    def generate_function_docstring(self, func_info: Dict, indent: str = "    ") -> str:
        """
        Generate a Google-style docstring for a function.
        
        Author: Hossein Molhem
        
        Args:
            func_info (Dict): Function metadata dictionary
            indent (str): Indentation string to use
            
        Returns:
            str: Generated docstring text
        """
        # Generate summary (placeholder)
        summary = f"TODO: Add description for {func_info['name']}"
        
        # Generate Args section
        args_lines = []
        for arg_name, arg_type in func_info['args']:
            if arg_name not in ['self', 'cls']:
                args_lines.append(f"{arg_name} ({arg_type}): TODO: Describe {arg_name}")
        args_section = '\n'.join(f"{indent}    {line}" for line in args_lines) if args_lines else f"{indent}    None"
        
        # Generate Returns section
        returns_section = f"{indent}    {func_info['returns']}: TODO: Describe return value"
        
        # Generate Raises section (placeholder)
        raises_section = f"{indent}    None"
        
        # Build docstring
        docstring = f'{indent}"""\n'
        docstring += f'{indent}{summary}\n'
        docstring += f'{indent}\n'
        docstring += f'{indent}Author: {self.author}\n'
        docstring += f'{indent}\n'
        docstring += f'{indent}Args:\n'
        docstring += f'{args_section}\n'
        docstring += f'{indent}\n'
        docstring += f'{indent}Returns:\n'
        docstring += f'{returns_section}\n'
        docstring += f'{indent}\n'
        docstring += f'{indent}Raises:\n'
        docstring += f'{raises_section}\n'
        docstring += f'{indent}"""\n'
        
        return docstring
    
    def generate_class_docstring(self, class_info: Dict, indent: str = "    ") -> str:
        """
        Generate a Google-style docstring for a class.
        
        Author: Hossein Molhem
        
        Args:
            class_info (Dict): Class metadata dictionary
            indent (str): Indentation string to use
            
        Returns:
            str: Generated docstring text
        """
        # Generate summary (placeholder)
        summary = f"TODO: Add description for {class_info['name']}"
        if class_info['bases']:
            summary += f" (inherits from {', '.join(class_info['bases'])})"
        
        # Generate Attributes section
        attrs_lines = []
        for attr_name, attr_type in class_info['attributes']:
            attrs_lines.append(f"{attr_name} ({attr_type}): TODO: Describe {attr_name}")
        attrs_section = '\n'.join(f"{indent}    {line}" for line in attrs_lines) if attrs_lines else f"{indent}    None"
        
        # Build docstring
        docstring = f'{indent}"""\n'
        docstring += f'{indent}{summary}\n'
        docstring += f'{indent}\n'
        docstring += f'{indent}Author: {self.author}\n'
        docstring += f'{indent}\n'
        docstring += f'{indent}Attributes:\n'
        docstring += f'{attrs_section}\n'
        docstring += f'{indent}"""\n'
        
        return docstring
    
    def scan_file(self, filepath: Path) -> Tuple[List[Dict], List[Dict]]:
        """
        Scan a Python file for classes and functions needing docstrings.
        
        Author: Hossein Molhem
        
        Args:
            filepath (Path): Path to the Python file to scan
            
        Returns:
            Tuple[List[Dict], List[Dict]]: Lists of classes and functions missing docstrings
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
                
            tree = ast.parse(source, filename=str(filepath))
            
            missing_classes = []
            missing_functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self.extract_class_info(node)
                    if not class_info['has_docstring']:
                        class_info['filepath'] = filepath
                        missing_classes.append(class_info)
                        
                    # Check methods
                    for method_info in class_info['methods']:
                        if not method_info['has_docstring']:
                            method_info['filepath'] = filepath
                            method_info['class_name'] = class_info['name']
                            missing_functions.append(method_info)
                            
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Top-level functions only
                    if isinstance(node.parent if hasattr(node, 'parent') else None, ast.Module):
                        func_info = self.extract_function_info(node)
                        if not func_info['has_docstring']:
                            func_info['filepath'] = filepath
                            func_info['class_name'] = None
                            missing_functions.append(func_info)
                            
            return missing_classes, missing_functions
            
        except Exception as e:
            print(f"Error scanning {filepath}: {e}")
            return [], []
    
    def scan_project(self, root_dir: Path) -> Dict:
        """
        Scan entire project for missing docstrings.
        
        Author: Hossein Molhem
        
        Args:
            root_dir (Path): Root directory of the project
            
        Returns:
            Dict: Summary of missing docstrings by file
        """
        results = {
            'total_files': 0,
            'files_with_issues': 0,
            'total_missing_class_docstrings': 0,
            'total_missing_function_docstrings': 0,
            'files': {}
        }
        
        for filepath in root_dir.rglob('*.py'):
            if not self.should_process_file(filepath):
                continue
                
            results['total_files'] += 1
            
            missing_classes, missing_functions = self.scan_file(filepath)
            
            if missing_classes or missing_functions:
                results['files_with_issues'] += 1
                results['total_missing_class_docstrings'] += len(missing_classes)
                results['total_missing_function_docstrings'] += len(missing_functions)
                
                relative_path = filepath.relative_to(root_dir)
                results['files'][str(relative_path)] = {
                    'classes': missing_classes,
                    'functions': missing_functions
                }
                
        return results
    
    def generate_report(self, results: Dict, output_file: Path = None):
        """
        Generate a report of missing docstrings.
        
        Author: Hossein Molhem
        
        Args:
            results (Dict): Scan results from scan_project()
            output_file (Path): Optional path to save report
            
        Returns:
            None
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DOCSTRING AUDIT REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Author: {self.author}")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append("SUMMARY:")
        report_lines.append(f"  Total Python files scanned: {results['total_files']}")
        report_lines.append(f"  Files with missing docstrings: {results['files_with_issues']}")
        report_lines.append(f"  Missing class docstrings: {results['total_missing_class_docstrings']}")
        report_lines.append(f"  Missing function/method docstrings: {results['total_missing_function_docstrings']}")
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        for filepath, data in sorted(results['files'].items()):
            report_lines.append(f"\nFile: {filepath}")
            report_lines.append("-" * 80)
            
            if data['classes']:
                report_lines.append(f"\n  Missing Class Docstrings ({len(data['classes'])}):")
                for cls in data['classes']:
                    report_lines.append(f"    - Line {cls['lineno']}: class {cls['name']}")
                    
            if data['functions']:
                report_lines.append(f"\n  Missing Function/Method Docstrings ({len(data['functions'])}):")
                for func in data['functions']:
                    context = f" (in class {func['class_name']})" if func.get('class_name') else ""
                    report_lines.append(f"    - Line {func['lineno']}: def {func['name']}{context}")
                    
            report_lines.append("")
        
        report_text = '\n'.join(report_lines)
        
        # Print to console
        print(report_text)
        
        # Save to file if specified
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"\nReport saved to: {output_file}")


def main():
    """
    Main entry point for docstring generation tool.
    
    Author: Hossein Molhem
    
    Args:
        None
        
    Returns:
        None
        
    Raises:
        None
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Scan Python project for missing docstrings"
    )
    parser.add_argument(
        '--root',
        type=str,
        default='.',
        help='Root directory of the project to scan'
    )
    parser.add_argument(
        '--author',
        type=str,
        default=AUTHOR,
        help='Author name to include in docstrings'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/docstring_audit_report.txt',
        help='Path to save the audit report'
    )
    parser.add_argument(
        '--generate',
        action='store_true',
        help='Generate docstring templates (not yet implemented)'
    )
    
    args = parser.parse_args()
    
    root_dir = Path(args.root).resolve()
    output_file = Path(args.output)
    
    print(f"Scanning project at: {root_dir}")
    print(f"Author: {args.author}\n")
    
    generator = DocstringGenerator(author=args.author)
    results = generator.scan_project(root_dir)
    generator.generate_report(results, output_file=output_file)
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("1. Review the audit report to identify priority files")
    print("2. Start with core modules (geometry_processors/, core/radarpy/)")
    print("3. Add docstrings incrementally, testing after each file")
    print("4. Use Google-style docstring format with Author field")
    print("=" * 80)


if __name__ == '__main__':
    main()
