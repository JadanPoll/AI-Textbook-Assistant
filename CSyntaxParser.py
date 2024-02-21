import ast
import math
from typing import Any
from typing import Union
import astor
import re
import ast
import queue
import textwrap
import json
import tkinter as tk
from tkinter import ttk
import json

import tkinter as tk
from tkinter import ttk
import json
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import json
import ChatGPT
from html import escape
from tkhtmlview import HTMLLabel


from tkinter import Tk, ttk, Button

import cssutils
from bs4 import BeautifulSoup


def inline_css(html_content):
    # Find the CSS styles within the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    style_tag = soup.find('style')

    if style_tag:
        try:
            # Parse the CSS styles
            css_styles = cssutils.parseString(style_tag.string)
        except cssutils.CSSParseException as e:
            print(f"Error parsing CSS: {e}")
            return html_content

        # Remove the original <style> tag
        style_tag.decompose()

        # Inline the CSS styles into the HTML content
        for rule in css_styles:
            if rule.type == rule.STYLE_RULE:
                styles = rule.style
                css_text = styles.getCssText()

                for selector in rule.selectorList:
                    # Find all HTML elements matching the selector
                    elements = soup.select(selector.selectorText)

                    for element in elements:
                        # Inline the styles into the element
                        element['style'] = f"{css_text}; {element.get('style', '')}"

    return str(soup)

class JsonViewerApp:
    def __init__(self, json_data):

        self.gpt_responses = {}  # Dictionary to store GPT responses
        self.load_saved_responses()  # Load saved responses when the app starts


        self.root = tk.Tk()
        self.root.title("JSON Viewer")

        # Create a Treeview widget to display the JSON data
        self.tree = ttk.Treeview(self.root, columns=("Value"),selectmode="extended")
        self.tree.heading("#0", text="Key", anchor=tk.W)
        self.tree.heading("Value", text="Value", anchor=tk.W)

        # Create a ScrolledText widget for displaying the value of the 'code' key
        self.text_widget = tk.Text(self.root, wrap=tk.WORD, width=50, height=10)
        self.text_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.text_widget.pack_forget()  # Initially hide the text widget

        # Create an HTMLLabel widget for displaying HTML content
        self.html_label = HTMLLabel(self.root, html="<h1>Hello, world!</h1>")
        self.html_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.html_label.pack_forget()  # Initially hide the HTMLLabel widget


        # Insert the JSON data into the Treeview
        self.insert_json_data("", json_data)

        # Pack the Treeview widget
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bind a click event to the tree items to display the 'code' value in the text widget
        self.tree.bind("<Double-1>", self.on_tree_select)
        self.tree.bind('<Control-ButtonRelease-1>', self.on_ctrl_click)

        # Create a text box and an "Enter" button
        self.gpt_widget = tk.Text(self.root, height=1)  # Start with 1 line
        self.gpt_widget.pack(side=tk.BOTTOM, anchor=tk.S)
        self.gpt_widget.bind('<KeyPress-Return>', self.on_enter_key)

        self.enter_button = tk.Button(self.root, text="Enter", command=self.enter_chat_function)
        self.enter_button.pack(side=tk.BOTTOM)

        # Create an "Analyze" button
        self.analyze_button = tk.Button(self.root, text="Analyze", command=self.analyze_code_show_html_content)
        self.analyze_button.pack(side=tk.BOTTOM)


    def save_responses_to_file(self, filename="gpt_responses.json"):
        with open(filename, 'w') as file:
            print("Saving gpt_responses to...",filename)
            json.dump(self.gpt_responses, file)


    def load_saved_responses(self, filename="gpt_responses.json"):
        try:
            with open(filename, 'r') as file:
                self.gpt_responses = json.load(file)
        except FileNotFoundError:
            print(f"No saved responses file found. Starting with an empty response dictionary.")
    def insert_json_data(self, parent, data):
        if isinstance(data, dict):
            for key, value in data.items():
                print(key)
                item = self.tree.insert(parent, "end", text=key)

                self.insert_json_data(item, value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                item_text = f"[{i}]"
                sub_item = self.tree.insert(parent, "end", text=item_text)
                self.insert_json_data(sub_item, item)
        else:
            self.tree.set(parent, "Value", str(data))

    def on_tree_select(self, event):
        # Get the selected item
        item = self.tree.selection()[0]
        item_text = self.tree.item(item, "text")
        
        # Check if the selected item is 'code'
        if item_text == "code":
            # Get the value of the 'code' item
            code_value = self.tree.item(item, "values")[0]
            # Insert the code value into the text widget
            self.text_widget.delete('1.0', tk.END)  # Clear the text widget
            self.text_widget.insert(tk.END, code_value)
            self.text_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)  # Show the text widget
        else:
            self.text_widget.pack_forget()  # Hide the text widget
    def on_ctrl_click(self, event):
        pass

    def on_enter_key(self, event):
        # Increase the height of the Text widget by 1 line
        current_height = self.gpt_widget.cget('height')
        new_height = int(current_height) + 1
        self.gpt_widget.configure(height=new_height)

    def enter_chat_function(self):
        # Get the content from the text box
        content = self.gpt_widget.get("1.0", tk.END)
        print(f"Entered content:\n{content}")

        # Get the GPT response
        chat_chatgpt_response = ChatGPT.GPT3ChatBot.chat(content)
        print(f"ChatGPT response: {chat_chatgpt_response}")

        selected_item = self.tree.selection()[0]
        selected_item_text = self.tree.item(selected_item, "text")

        # Check if the selected item is 'code'
        if selected_item_text == "code":
            # Get the current GPT response for the selected "code" key
            code_key = self.tree.parent(selected_item)  # Parent of 'code' is the actual key
            current_gpt_response = self.gpt_responses.get(code_key, "")

            # Update the GPT response
            self.gpt_responses[code_key] = current_gpt_response + inline_css(chat_chatgpt_response)

            # Show the HTMLLabel widget on the right of the text widget when "Analyze" is pressed
            self.html_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

            # Get the current vertical scrollbar position
            current_position = self.html_label.yview()[0]

            # Display the GPT response in the HTMLLabel widget
            self.html_label.set_html(self.gpt_responses[code_key])

            # Restore the vertical scrollbar position
            self.html_label.yview_moveto(current_position)


    def analyze_code_show_html_content(self):
        # Get the current content from the text widget
        current_content = self.text_widget.get("1.0", tk.END)
        current_content = "Convert this into documentation info I can use in my book. :" + current_content

        # Get the activated "code" item
        selected_item = self.tree.selection()[0]
        selected_item_text = self.tree.item(selected_item, "text")

        # Check if the selected item is 'code'
        if selected_item_text == "code":
            # Get the current GPT response for the selected "code" key
            code_key = self.tree.parent(selected_item)  # Parent of 'code' is the actual key
            current_gpt_response = self.gpt_responses.get(code_key, "")

            if not current_gpt_response:
                # If GPT response is not stored, generate a new one
                chatgpt_response = ChatGPT.GPT3ChatBot.chat(current_content)
                inlined_chatgpt_response = inline_css(chatgpt_response)

                # Store the generated GPT response for future use
                self.gpt_responses[code_key] = inlined_chatgpt_response
                current_gpt_response = inlined_chatgpt_response

            # Show the HTMLLabel widget on the right of the text widget when "Analyze" is pressed
            self.html_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)


            # Get the current vertical scrollbar position
            current_position = self.html_label.yview()[0]

            # Display the GPT response in the HTMLLabel widget
            self.html_label.set_html(current_gpt_response)

            # Restore the vertical scrollbar position
            self.html_label.yview_moveto(current_position)
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)  # Save responses before closing
        self.root.mainloop()

    def on_closing(self):
        # Save responses before closing the app
        self.save_responses_to_file()
        self.root.destroy()


pattern = re.compile(r'\b((class|struct)\s+(\w+)\s*|\b(\w+)\s+(\w+)\s*(\([^)]*\))\s*){', re.DOTALL)

def extract_functions_and_classes_from_c_code(code, processed_positions=None, prev_match_index_param=0):
    if processed_positions is None:
        processed_positions = set()

    result_dict = {}
    prev_id = 0  # Initialize the start location of the previous match
    _endBodyIndex = 0
    for match in pattern.finditer(code):
        if match.end() in processed_positions:
            continue  # Skip matches that have already been processed

        match_type = match.group(2) or match.group(4)  # Either class/struct or function
        name = match.group(3) or match.group(5)  # Either class/struct name or function name
        start_id = match.end()

        # Mark this position as processed
        processed_positions.add(match.end() + prev_match_index_param)

        # Store code between matches only if it's not blank
        if not code[prev_id:match.start()].isspace() and code[prev_id:match.start()] != "":
            result_dict[str(start_id)] = {"code": code[prev_id:match.start()].strip(), "type": "global"}

        if match_type:
            if match_type == "class" or match_type == "struct":
                # Extract class or struct body until the end of the string
                body, _endBodyIndex = extract_balanced_content('{' + code[match.end():])
                _endBodyIndex += 1
                full_qualifier = match_type + " " + name
                prev_match_index = match.end() + prev_match_index_param
                result_dict[full_qualifier] = {
                    "id": str(match.start()),
                    "code": full_qualifier + "\n" + body,
                    "functions": extract_functions_and_classes_from_c_code(body[1:-1], processed_positions,
                                                                           prev_match_index),
                    "type": match_type
                }
            else:
                # Extract function body until the end of the string
                parameters = match.group(6) if match.group(6) else ""
                body, _endBodyIndex = extract_balanced_content('{' + code[match.end():])
                full_qualifier = match_type + " " + name
                result_dict[full_qualifier] = {
                    "id": str(start_id),
                    "code": full_qualifier + parameters + "\n" + body.strip(),
                    "type": match_type
                }

        # Update the start location of the previous match
        prev_id = start_id + _endBodyIndex

    # Store code between the last match and the end of the string only if it's not blank
    if not code[prev_id:].isspace() and code[prev_id:] != "":
        result_dict[str(prev_id)] = {"code": code[prev_id:].strip(), "type": "global"}

    return result_dict


def extract_balanced_content(code):
    count = 0
    index = 0
    for char in code:
        if char == '{':
            count += 1
        elif char == '}':
            count -= 1
        index += 1
        if count == 0:
            break
    return code[:index], index


class CodeMetricsAnalyzer:
    def __init__(self):
        self.halstead_volume = 0
        self.cyclomatic_complexity = 0
        self.lines_of_code = 0
        self.maintainability_index = 0

    def calculate_metrics_from_file(self, file_name: str):
        try:
            with open(file_name, 'r') as file:
                code = file.read()
                if file_name.endswith('.py'):
                    return self.calculate_metrics(code)
                elif file_name.endswith(('.c', '.cpp')):
                    return extract_functions_and_classes_from_c_code(code)
                else:
                    return {"Error": f"Unsupported file type: {file_name}"}
        except FileNotFoundError:
            return {"Error": f"File not found: {file_name}"}

    def calculate_metrics(self, code: str):
        try:
            # Parse the code using AST (Abstract Syntax Tree)
            tree = ast.parse(code)

            # Calculate Halstead Volume
            self.halstead_volume = self.calculate_halstead_volume(tree)

            # Calculate Cyclomatic Complexity
            self.cyclomatic_complexity = self.calculate_cyclomatic_complexity(tree)

            # Calculate Lines of Code
            self.lines_of_code = self.calculate_lines_of_code(tree)

            # Calculate Maintainability Index
            self.maintainability_index = self.calculate_maintainability_index()

            # Calculate metrics for each class and function definition
            class_metrics = self.calculate_class_metrics(tree)

            # Check if class_metrics is None, then calculate function metrics
            if not class_metrics:

                function_metrics = self.calculate_function_metrics(tree)
                if not function_metrics:
                    return {
                         "Global Metrics": self.get_metrics_dict()
                    }
                return {
                    "Global Metrics": self.get_metrics_dict(),
                    "Function Metrics": function_metrics
                }
            else:
                # Return metrics as a dictionary
                return {
                    "Global Metrics": self.get_metrics_dict(),
                    "Class Metrics": class_metrics
                }

        except SyntaxError as e:
            return {"Error": f"Syntax error in the provided code: {e}"}

    def calculate_halstead_volume(self, tree: ast.AST) -> float:
        self.operators = set()
        self.operands = set()

        def visit(node):
            if isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.FloorDiv, ast.Pow, ast.LShift, ast.RShift,
                                 ast.BitOr, ast.BitXor, ast.BitAnd)):
                self.operators.add(type(node).__name__)
            elif isinstance(node, (ast.NameConstant, ast.Str, ast.Num)):
                self.operands.add(node.n)
            for child in ast.iter_child_nodes(node):
                visit(child)

        visit(tree)

        total_operators = len(self.operators)
        total_operands = len(self.operands)

        if total_operators > 0 and total_operands > 0:
            halstead_volume = total_operators * math.log2(total_operators) + total_operands * math.log2(total_operands)
            return halstead_volume
        else:
            return 0.0

    def calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        edges = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.For,
                                                                        ast.While, ast.If, ast.With, ast.Try)))
        nodes = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor,
                                                                      ast.Try, ast.ExceptHandler)))
        connected_components = 1  # For simplicity, assuming the entire code is one connected component

        return edges - nodes + 2 * connected_components

    def calculate_lines_of_code(self, tree: ast.AST) -> int:
        loc_count = 0
        EXECUTABLE_NODES = (
            ast.FunctionDef, ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.ClassDef, ast.For, ast.While, ast.If,
            ast.IfExp, ast.With, ast.AsyncWith, ast.Try, ast.ExceptHandler, ast.Assign, ast.AnnAssign, ast.Expr, ast.Return,
            ast.Continue, ast.Break, ast.Pass, ast.Raise, ast.Assert, ast.Global, ast.Nonlocal, ast.Delete, ast.Import,
            ast.ImportFrom, ast.AugAssign,
        )

        for node in ast.walk(tree):
            if isinstance(node, EXECUTABLE_NODES):
                loc_count += 1
                if isinstance(node, ast.If):
                    expr_count = sum(1 for child in ast.iter_child_nodes(node) if
                                     isinstance(child, ast.stmt) and isinstance(child, ast.Expr))
                    if expr_count == 2:
                        loc_count += 1

        return loc_count

    def calculate_maintainability_index(self) -> Union[float, dict]:
        try:
            maintainability_index = (
                171 - 5.2 * self.safe_log(self.halstead_volume)
                - 0.23 * self.cyclomatic_complexity
                - 16.2 * self.safe_log(self.lines_of_code)
            )
            return maintainability_index
        except ValueError as e:
            return {"Error": f"Error calculating Maintainability Index: {str(e)}"}

    def safe_log(self, value):
        # Check if the input value is positive before taking the logarithm
        if value > 0:
            return math.log(value)
        else:
            print("Safe Logged")
            return 0.0

    def calculate_class_metrics(self, tree: ast.AST) -> dict:
        class_metrics = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Calculate metrics for each class
                class_name = node.name
                class_metrics[class_name] = self.calculate_function_metrics(node)

        return class_metrics

    def calculate_function_metrics(self, tree: ast.AST) -> dict:
        function_metrics = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Calculate metrics for each function
                function_name = node.name

                # Extract the entire function source code
                function_code = astor.to_source(node)

                # Remove the function definition to get only the body of the function
                function_code_lines = function_code.split('\n')
                function_code_lines = function_code_lines[1:-1]  # Exclude the first and last lines
                function_code = '\n'.join(function_code_lines)

                # Remove leading tabs from every line
                function_code = textwrap.dedent(function_code)

                # Use a new instance of CodeMetricsAnalyzer to calculate the metrics for each function's code
                function_analyzer = CodeMetricsAnalyzer()
                metrics = function_analyzer.calculate_metrics(function_code)


                # Add the function code to the metrics dictionary
                if "Global Metrics" in metrics and "Lines of Code" in metrics["Global Metrics"]:
                    loc = metrics["Global Metrics"]["Lines of Code"]
                    if loc > 0:
                        function_code = astor.to_source(node).strip()
                        metrics["code"] = function_code

                function_metrics[function_name] = metrics

        return function_metrics

    def get_metrics_dict(self):
        metrics_dict = {
            "Maintainability Index": self.maintainability_index,
            "Halstead Volume": self.halstead_volume,
            "Cyclomatic Complexity": self.cyclomatic_complexity,
            "Lines of Code": self.lines_of_code
        }
        return metrics_dict


# Example usage:
code_analyzer = CodeMetricsAnalyzer()

# Example code (you can replace this with your actual code)
# Example code with various statements
example_code = """
# Single-line comment
class ExampleClass:
    def example_function(self, x, y):
        result = 0
        if x > 0:
            print("Positive")
            if y > 0:
                print("Both x and y are positive")
            else:
                print("Only x is positive")
        elif x < 0:
            print("Negative")
        else:
            print("Zero")

        for i in range(3):
            result += i

        while result < 10:
            result *= 2
            if result == 8:
                break

        try:
            raise ValueError("Example error")
        except ValueError as e:
            print(f"Caught an exception: {e}")
"""

# Create an instance of ASTWalker and perform traversals
#walker = ASTWalker()
#walker.traverse_depth_first(example_code)
def extract_parent_key_and_code(data, parent_key=None, result=[]):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == 'code' and parent_key:
                result.append((parent_key, value))
            elif isinstance(value, (dict, list)):
                extract_parent_key_and_code(value, key, result)
    elif isinstance(data, list):
        for item in data:
            extract_parent_key_and_code(item, parent_key, result)

    return result
result = code_analyzer.calculate_metrics_from_file("example.cpp")

# Convert the result to a JSON-formatted string with indentation
result_str = json.dumps(result, indent=5)
# Example usage:
# Assuming `result` is the JSON data you want to visualize
result_json = json.loads(result_str)

#print(extract_parent_key_and_code(result_json))

# Example usage:
# Assuming `result` is the JSON data you want to visualize
json_viewer = JsonViewerApp(result_json)
json_viewer.run()
