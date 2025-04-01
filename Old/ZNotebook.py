import re
import os
import json
import queue
import base64
import sys
import time
import threading
import pickle
import hashlib
import io
import tkinter as tk
from tkinter import ttk, filedialog
from tkinter.scrolledtext import ScrolledText
from html import escape
from tkhtmlview import HTMLLabel
import cssutils
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from PIL import Image, ImageTk
import numpy as np
from io import BytesIO
import OpenAICMD


ChatGPT = OpenAICMD.WebSocketClientApp("https://ninth-swamp-orangutan.glitch.me")




def runFunctionCatchExceptions(func, *args, **kwargs):
    try:
        result = func(*args, **kwargs)
    except Exception as e:
        return ["exception", e]
    return ["RESULT", result]

def runFunctionWithTimeoutAndRetry(func, args=(), kwargs={}, timeout_duration=10, default=None, retry_count=3):
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = default
        def run(self):
            self.result = runFunctionCatchExceptions(func, *args, **kwargs)
    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.is_alive():
        print("Function did not complete within the timeout.")
        for attempt in range(retry_count):
            it = InterruptableThread()
            it.start()
            it.join(timeout_duration)
            if not it.is_alive():
                break
            print(f"Attempt {attempt + 1} timed out, retrying...")
        if it.is_alive():
            return default
    if it.result[0] == "exception":
        raise it.result[1]
    return it.result[1]

def extract_html_content(html_content):
    # Check if the input contains HTML tags
    if "<html>" not in html_content and "</html>" not in html_content:
        # No HTML tags found, return the plaintext content
        return [html_content],{}

    # Parse HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all <p> tags and extract their content
    paragraphs = [p.get_text() for p in soup.find_all('p')]

    # Find all <h*> tags and extract their content
    headings = {}
    for i in range(1, 7):  # H1 to H6
        h_tags = soup.find_all(f'h{i}')
        headings[f'h{i}_headings'] = [h.get_text() for h in h_tags]

    # Return extracted content
    return paragraphs, headings


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


def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

JARVIS_ACCEPT="AudioFiles/Jarvis_Accept.wav"
JARVIS_CLICK="AudioFiles/Jarvis_Click.wav"
JARVIS_PROCESSING="AudioFiles/Load/Jarvis_Load_out.wav"
JARVIS_START="AudioFiles/Jarvis_StartUp.wav"
import sounddevice as sd
from scipy.io.wavfile import read
import numpy as np

def play_audio(file_path):
    # Read the audio file
    sample_rate, data = read(file_path)

    # Normalize the audio data
    data = data.astype(np.float32) / np.max(np.abs(data), axis=0)

    # Play the audio
    sd.play(data, sample_rate)


class LoadingAnimation:
    def __init__(self, parent, filename, width=150, height=150):
        self.parent = parent
        self.is_running = False
        self.width = width
        self.height = height

        # Load the frames of the GIF animation and resize them
        self.frames = self.load_gif_frames(filename, self.width, self.height)

        # Get the background color of the parent window
        self.bg_color = parent.cget('bg')

        # Create a canvas widget to draw the animation with the same background color as the window
        self.canvas = tk.Canvas(self.parent, width=self.width, height=self.height, bg=self.bg_color, highlightthickness=0)
        self.canvas.pack()

        # Display the first frame of the animation
        self.current_frame = 0
        self.image_item = None
        self.update_image()

    def load_gif_frames(self, filename, width, height):
        gif = Image.open(filename)
        frames = []
        for frame in range(0, gif.n_frames):
            gif.seek(frame)
            # Resize each frame to the desired width and height
            resized_frame = gif.resize((width, height), Image.LANCZOS)
            frames.append(ImageTk.PhotoImage(resized_frame))
        return frames

    def update_image(self):
        # Get the size of the current frame
        frame_width = self.width
        frame_height = self.height

        # Calculate the center of the canvas
        center_x = self.width / 2
        center_y = self.height / 2

        # Calculate the position of the image on the canvas
        x0 = center_x - frame_width / 2
        y0 = center_y - frame_height / 2

        # If image item already exists, delete it
        if self.image_item:
            self.canvas.delete(self.image_item)

        # Display the current frame centered on the canvas
        self.image_item = self.canvas.create_image(x0, y0, image=self.frames[self.current_frame], anchor="nw")

    def update_frame(self):
        if self.is_running:
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            self.update_image()
            self.parent.after(50, self.update_frame)

    def start_animation(self):
        self.is_running = True
        self.update_frame()

    def stop_animation(self):
        self.is_running = False

class HTMLGenerator:


    def __init__(self):


        # Define theme colors with better values to fit the descriptions
        self.themes = {
            "default": ("#FF4C4C", "#4CFF4C", "#4C4CFF", "#C64CFF"),  # Bright primary colors
            "light": ("#FFFFFF", "#E0E0E0", "#CCCCCC", "#B0B0B0"),  # Light and neutral shades
            "dark": ("#1A1A1A", "#2B2B2B", "#3D3D3D", "#4F4F4F"),  # Dark and muted tones
            "monochrome": ("#000000", "#555555", "#AAAAAA", "#FFFFFF"),  # Black, white, and grays
            "nature": ("#4B8F29", "#8FBF5A", "#C6E48B", "#556B2F"),  # Earthy greens and browns
            "ocean": ("#004080", "#0077B6", "#0096C7", "#00B4D8"),  # Deep and vibrant blues
            "sunset": ("#FF4500", "#FF7F50", "#FF8C00", "#FFD700"),  # Warm reds, oranges, and yellows
            "forest": ("#013220", "#3A5F0B", "#7B8C4C", "#A9C99E"),  # Dark greens and browns
            "autumn": ("#A52A2A", "#FF6347", "#FF8C00", "#D2691E"),  # Autumnal browns, oranges, and reds
            "pastel": ("#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9"),  # Soft and muted colors
            "space": ("#0B0D17", "#3D3D5C", "#6B6B8B", "#A7A7C5"),  # Deep blues and purples, space-like
            "desert": ("#C19A6B", "#D2B48C", "#EDC9AF", "#F4A460"),  # Sandy and earthy tones
            "spring": ("#77DD77", "#FFB347", "#FDFD96", "#84B6F4"),  # Bright greens, yellows, and blues
            "rainbow": ("#FF0000", "#FF7F00", "#FFFF00", "#00FF00"),  # Traditional rainbow spectrum
            "vintage": ("#7E4A35", "#B48C65", "#D1B384", "#C0C0C0"),  # Faded browns and muted pastels
            "winter": ("#4682B4", "#ADD8E6", "#FFFFFF", "#D3D3D3"),  # Cool blues, whites, and grays
        }

        # Define sub-section colors for each theme with better values
        self.sub_section_colors = {
            "default": ["#EFEFEF", "#DFDFDF"],  # Light grays
            "light": ["#F8F9FA", "#E9ECEF"],  # Very light grays
            "dark": ["#333333", "#444444"],  # Dark grays
            "monochrome": ["#D0D0D0", "#F0F0F0"],  # Light grays
            "nature": ["#E6F2E6", "#CCFFCC"],  # Light greens
            "ocean": ["#E0FFFF", "#AFEEEE"],  # Light blues
            "sunset": ["#FFB347", "#FFA07A"],  # Warm pastel colors
            "forest": ["#E1EAD6", "#A8D5BA"],  # Soft greens
            "autumn": ["#FFEFD5", "#FFDAB9"],  # Pale oranges and yellows
            "pastel": ["#FFE4E1", "#FFF0F5"],  # Light pastels
            "space": ["#282C34", "#3A3F54"],  # Very dark blue shades
            "desert": ["#FFE4C4", "#F4A460"],  # Sandy tones
            "spring": ["#E6E6FA", "#FFFACD"],  # Soft yellows and lavenders
            "rainbow": ["#FFFAFA", "#F0FFF0"],  # Soft and off-whites
            "vintage": ["#FAEBD7", "#EEDFCC"],  # Faded antique whites
            "winter": ["#F0F8FF", "#E6E6FA"],  # Cold whites and blues
        }



    def inline_css(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        style_tag = soup.find('style')

        if style_tag:
            try:
                css_styles = cssutils.parseString(style_tag.string)
            except cssutils.CSSParseException as e:
                    print(f"Error parsing CSS: {e}")
                    return html_content

            style_tag.decompose()

            for rule in css_styles:
                if rule.type == rule.STYLE_RULE:
                    css_text = rule.style.getCssText()
                    for selector in rule.selectorList:
                        elements = soup.select(selector.selectorText)
                        for element in elements:
                            existing_styles = element.get('style', '')
                            if existing_styles:
                                element['style'] = f"{existing_styles}; {css_text}"
                            else:
                                element['style'] = css_text

        return str(soup)








    def generate_html_response(self, prompt, response_title, colors, section_colors, html_content, theme="default", use_backgrounding=True):
        color1, color2, color3, color4 = colors
        soup = BeautifulSoup(html_content, 'html.parser')

        # Expanded mapping of tag names to corresponding HTML tags and styles
        tag_mapping = {
            'h1': ('h2', color1),  # h1 is converted to h2 with color1
            'h2': ('h3', color2),  # h2 is converted to h3 with color2
            'h3': ('h4', color2),  # h3 is converted to h4 with color2
            'h4': ('h5', color3),  # h4 is converted to h5 with color3
            'h5': ('h6', color3),  # h5 is converted to h6 with color3
            'h6': ('h6', color4),  # h6 remains as h6 but with color4
            'p': ('p', None),
            'ol': ('ol', None),
            'ul': ('ul', None),
            'li': ('li', None),
            'strong': ('h3', color3),
            'blockquote': ('blockquote', None),  # Example: Keep blockquote with no change
            'code': ('pre', None),  # Convert inline code to preformatted block
            'em': ('em', None),  # Emphasized text stays the same
            'table': ('table', None),  # Table stays the same but styled differently
            'tr': ('tr', None),
            'th': ('th', None),
            'td': ('td', None),
        }

        # Modifying the existing content in soup
        for section_index, section in enumerate(soup.find_all(tag_mapping.keys())):
            tag, color = tag_mapping[section.name]
            section_color = section_colors[section_index % len(section_colors)]

            # Clear existing content and set new formatted HTML
            if section.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                new_html = f'<div class="sub-section" style="background-color: {section_color};">\n<{tag} style="color: {color};">{section.text}</{tag}></div>'
                section.replace_with(BeautifulSoup(new_html, 'html.parser'))
            elif section.name == 'p':
                new_html = f'<div class="sub-section" style="background-color: {section_color};">\n<p>{section.text}</p>\n</div>'
                section.replace_with(BeautifulSoup(new_html, 'html.parser'))
            elif section.name in ['ol', 'ul']:
                new_html = f'<div class="sub-section" style="background-color: {section_color};">\n<{section.name}>{section.decode_contents()}</{section.name}>\n</div>'
                section.replace_with(BeautifulSoup(new_html, 'html.parser'))
            elif section.name == 'li':
                new_html = f'<li>{section.text}</li>'
                section.replace_with(BeautifulSoup(new_html, 'html.parser'))
            elif section.name == 'strong':
                new_html = f'<h3 style="color: {color};">{section.text}</h3>'
                section.replace_with(BeautifulSoup(new_html, 'html.parser'))
            elif section.name == 'blockquote':
                new_html = f'<div class="sub-section" style="background-color: {section_color};">\n<blockquote>{section.text}</blockquote>\n</div>'
                section.replace_with(BeautifulSoup(new_html, 'html.parser'))
            elif section.name == 'code':
                new_html = f'<div class="sub-section" style="background-color: {section_color};">\n<pre>{section.text}</pre>\n</div>'
                section.replace_with(BeautifulSoup(new_html, 'html.parser'))
            elif section.name in ['table', 'tr', 'th', 'td']:
                # Leave tables as is but wrap in a styled section
                new_html = f'<div class="sub-section" style="background-color: {section_color};">\n<{section.name}>{section.decode_contents()}</{section.name}>\n</div>'
                section.replace_with(BeautifulSoup(new_html, 'html.parser'))

        # Generate final HTML template
        body_bg_color = "#f8f8f8" if use_backgrounding else "transparent"
        html_template = f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width,initial-scale=1.0">
            <title>{response_title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: {body_bg_color};
                    margin: 0;
                    padding: 10px;
                }}
                h1 {{ color: {color1}; font-size: 25px; margin-bottom: 8px; }}
                h2 {{ color: {color2}; font-size: 23px; margin-bottom: 6px; }}
                h3 {{ color: {color3}; font-size: 21px; margin-bottom: 4px; }}
                h4 {{ color: {color4}; font-size: 19px; margin-bottom: 4px; }}
                h5 {{ color: {color3}; font-size: 17px; margin-bottom: 3px; }}
                h6 {{ color: {color4}; font-size: 15px; margin-bottom: 2px; }}
                p {{ line-height: 1.4; margin-bottom: 8px; }}
                section {{
                    background-color: #ffffff;
                    border-radius: 5px;
                    margin-bottom: 10px;
                    padding: 15px;
                }}
                .sub-section {{
                    border-radius: 5px;
                    padding: 10px;
                    margin-bottom: 8px;
                }}
            </style>
        </head>
        <body>
            <section>
                <p>{prompt}</p>
                <div class="sub-section"></div>
            </section>
            <section>
                {str(soup)}
            </section>
        </body>
        </html>'''

        return self.inline_css(html_template)






    def respond_with_html(self, prompt, contents, theme="default", use_backgrounding=True):
        print("Echo Echo",contents)
        response_title = "Response Title"
        if theme not in self.themes:
            theme = "default"

        theme_colors = self.themes[theme]
        section_colors = self.sub_section_colors[theme]

        html_response = self.generate_html_response(prompt, response_title, theme_colors, section_colors, contents)
        return html_response

    def display_html(self, html_content):
        root = tk.Tk()
        root.title("HTML Content")
        html_label = HTMLLabel(root, html=html_content)
        html_label.pack(fill="both", expand=True)
        root.mainloop()

    def generate_and_display_html(self, prompt, contents, theme="default", use_backgrounding=True):
        html_response = self.respond_with_html(prompt, contents, theme, use_backgrounding)
        #self.display_html(html_response)
        return html_response

class ScrolledCanvas(tk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, **kwargs)
        self.v_scroll = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.h_scroll = tk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.config(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
class PDFViewerApp:
    def __init__(self, parent=None, **kwargs):
        self.parent = parent
        self.pdf_file = None
        self.pdf_document = None
        self.current_page = 0
        self.total_pages = 0
        self.image1 = None
        self.thread_lock = threading.Lock()
        self.is_packed = False

        self.setup_ui()

    def parse_and_extract_number(self, code_key):
        # Regular expression to find numeric values
        pattern = r'\d+'
        # Search for numeric values in the code_key
        matches = re.findall(pattern, code_key)
        # Extract the first numeric value found (if any)
        if matches:
            return int(matches[0])
        else:
            return None
        
    def setup_ui(self):
        self.frame = tk.Frame(self.parent)

        # Create a ScrolledCanvas widget for displaying the page
        self.scrolled_canvas = ScrolledCanvas(self.frame)
        self.scrolled_canvas.pack(fill=tk.BOTH, expand=True)

        # Create a canvas widget inside the ScrolledCanvas
        self.canvas = self.scrolled_canvas.canvas


        # Bind the resize event of the canvas widget
        self.canvas.bind("<Configure>", self.resize_canvas)

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)
        self.is_packed = True

    def pack_forget(self):
        self.frame.pack_forget()
        self.is_packed = False

    def set_pdf(self, pdf_file_path):
        self.pdf_file = pdf_file_path
        self.pdf_document = fitz.open(self.pdf_file)
        self.total_pages = len(self.pdf_document)
        threading.Thread(target=self.show_page, args=(self.current_page,)).start()

    def show_page(self, page_number):
        with self.thread_lock:
            self.current_page = page_number
            if self.pdf_document:
                page = self.pdf_document.load_page(page_number)
                pix = page.get_pixmap()

                img_data = pix.tobytes("ppm")
                img = Image.open(io.BytesIO(img_data))

                # Get the canvas dimensions
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()

                # Resize the image with Lanczos filter for better text quality
                img = img.resize((canvas_width, canvas_height), Image.LANCZOS)

                photo = ImageTk.PhotoImage(image=img)

                # Update the canvas image
                self.canvas.create_image(0, 0, image=photo, anchor='nw')

                # Keep reference to prevent garbage collection
                self.image1 = photo

    def show_prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            threading.Thread(target=self.show_page, args=(self.current_page,)).start()

    def show_next_page(self):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            threading.Thread(target=self.show_page, args=(self.current_page,)).start()

    def resize_canvas(self, event):
        # Update the canvas to display the current page
        threading.Thread(target=self.show_page, args=(self.current_page,)).start()

class DraggableWidget:

    instances = {}  # Dictionary to store instances keyed by unique identifier

    def __init__(self, widget,*args, master=None, grid_size=5, **kwargs):
        # Remove the 'key' command from kwargs if it exists
        kwargs.pop('command', None)
    
        self.master = master
        self.dragging = False
        self.widget = widget
        self.grid_size = grid_size
        self.widget_name = self.encode_widget_name(widget, *args,**kwargs)  # Encoded widget name
        self.widget_name2 = kwargs.get('text', None)

        print(kwargs)
        self.widget.bind("<ButtonPress-1>", self.on_drag_start)
        self.widget.bind("<ButtonRelease-1>", self.on_drag_stop)
        self.widget.bind("<B1-Motion>", self.on_drag_motion)

        self.master.bind("<Configure>", DraggableWidget.on_window_configure)  # Bind to window resize event

        # Initialize initial_width and initial_height if not initialized yet
        DraggableWidget.initial_width = getattr(DraggableWidget, 'initial_width', self.master.winfo_width())
        DraggableWidget.initial_height = getattr(DraggableWidget, 'initial_height', self.master.winfo_height())
        DraggableWidget.g_master=self.master


        # Add the instance to the dictionary using the widget name as the key
        DraggableWidget.add_instance(self.widget_name, self)

        self.master.after(1000, self.load_position)  # Load the position from storage after a delay

    @staticmethod
    def on_window_configure(event):
        # Get the top-level window associated with the event
        top_level_window = DraggableWidget.g_master

        # Get the width and height of the top-level window
        width = top_level_window.winfo_width()
        height = top_level_window.winfo_height()

        # Check if the configure event is a window resize event
        if (width != DraggableWidget.initial_width or height != DraggableWidget.initial_height):
            # Update initial width and height
            DraggableWidget.initial_width = width
            DraggableWidget.initial_height = height

            # Call global_load_position to handle window resize
            DraggableWidget.global_load_position()
    @staticmethod
    def global_load_position(event=None):
        # Call load_position method for each instance
        for instance in DraggableWidget.instances.values():
            instance.load_position()

        
    @staticmethod
    def get_instance(widget_name):
        # Retrieve instance from dictionary by widget name
        return DraggableWidget.instances.get(widget_name)

    @staticmethod
    def add_instance(widget_name, instance):
        # Add instance to the dictionary using the widget name as the key
        DraggableWidget.instances[widget_name] = instance


    def on_drag_start(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.dragging = True

    def on_drag_stop(self, event):
        self.dragging = False
        self.save_position()  # Save the position to storage

    def on_drag_motion(self, event):
        if self.dragging:
            # Calculate new coordinates
            x = self.widget.winfo_x() - self.start_x + event.x
            y = self.widget.winfo_y() - self.start_y + event.y
            # Lock to grid
            x = self.snap_to_grid(x)
            y = self.snap_to_grid(y)
            # Update widget position
            self.widget.place_configure(x=x, y=y)

    def snap_to_grid(self, coordinate):
        # Snap the coordinate to the nearest multiple of grid_size
        return int(round(coordinate / self.grid_size)) * self.grid_size

    def encode_widget_name(self, widget, *args,**kwargs):
        # Combine widget name and arguments into a string
        widget_info = f"{type(widget).__name__}_{sorted(kwargs.items())}"
        # Encode the widget info using SHA-256 hash
        hashed_widget_info = hashlib.sha256(widget_info.encode()).hexdigest()
        return hashed_widget_info


    def load_position(self,event=None):

        position_file = os.path.join("Widget_Position2", f"{self.widget_name}_position.pkl")
        try:
            with open(position_file, "rb") as f:
                print("Found_Something",self.widget_name2)
                position = pickle.load(f)
                # Load position as fraction of window dimensions
                x = self.master.winfo_width() * position[0]
                y = self.master.winfo_height() * position[1]
                self.widget.place_configure(x=x, y=y)
        except FileNotFoundError:
            pass

    def save_position(self):
        position = (self.widget.winfo_x() / self.master.winfo_width(), self.widget.winfo_y() / self.master.winfo_height())
        print("Saving")
        # Save position as fraction of window dimensions
        position_file = os.path.join("Widget_Position2", f"{self.widget_name}_position.pkl")
        os.makedirs("Widget_Position2", exist_ok=True)  # Create the folder if it doesn't exist
        with open(position_file, "wb") as f:
            pickle.dump(position, f)

    def pack(self, *args, **kwargs):
        # Extract and remove 'alter' from kwargs if it exists
        alter = kwargs.pop('alter', False)

        # Checking if alter is True
        if alter:
            # If alter is True, call self.widget.pack() with the provided arguments and keyword arguments
            self.widget.pack(*args, **kwargs)

        else:
            # If alter is not True, simply call self.widget.pack() without any arguments
            self.widget.pack()

        # Call self.load_position after a delay of 200 milliseconds
        self.master.after(200, self.load_position)
    def pack_forget(self):
        self.widget.pack_forget()

    # Handle undefined attribute calls by delegating to the underlying widget
    def __getattr__(self, attr):
        return getattr(self.widget, attr)


def create_draggable_label(master=None, **kwargs):
    label = LabelCustom(master, **kwargs)
    return DraggableWidget(label, master=master, **kwargs)

def create_draggable_entry(master=None, **kwargs):
    entry = EntryCustom(master, **kwargs)
    return DraggableWidget(entry, master=master, **kwargs)

def create_draggable_text(master=None, **kwargs):
    pass
    # text = TextCustom(master, **kwargs)
    # return DraggableWidget(text, master=master, **kwargs)

def create_draggable_checkbutton(master=None, **kwargs):
    checkbutton = CheckbuttonCustom(master, **kwargs)
    return DraggableWidget(checkbutton, master=master, **kwargs)

def create_draggable_button(master=None, **kwargs):
    button = ButtonCustom(master, **kwargs)
    return DraggableWidget(button, master=master, **kwargs)

def create_draggable_panedwindow(master=None, **kwargs):
    paned_window = PanedWindowCustom(master, **kwargs)
    return DraggableWidget(paned_window,master= master, **kwargs)

def create_draggable_optionsmenu(master=None,*args, **kwargs):
    options_menu = OptionsMenuCustom(master,*args, **kwargs)
    return DraggableWidget(options_menu,*args,master=master, **kwargs)

# Custom widget classes
LabelCustom = tk.Label
EntryCustom = tk.Entry
# TextCustom = tk.Text  # Uncomment if needed
CheckbuttonCustom = tk.Checkbutton
ButtonCustom = tk.Button
OptionsMenuCustom = tk.OptionMenu
PanedWindowCustom = tk.PanedWindow
# Assign custom classes to tkinter widget classes
#tk.Label = create_draggable_label
#tk.Entry = create_draggable_entry
# tk.Text = create_draggable_text  # Uncomment if needed
#tk.Checkbutton = create_draggable_checkbutton
#tk.Button = create_draggable_button
#tk.PanedWindow = create_draggable_panedwindow
#tk.OptionMenu = create_draggable_optionsmenu

SINGLE_ANALYZE_PREFIX="Teach me all about this \n"
class JsonViewerApp:



    def callback_function(self,message):
        self.MESSAGE=message




    def __init__(self, text_font=("Trebuchet MS", 12), l_spacing1=10, l_spacing3=10):

        self.AICHOICE="sendPhind"
        self.generator = HTMLGenerator()

        self.root = tk.Tk()
        self.root.title("Jarvis Analysis Viewer")




        # Create and pack main frames
        self.main_topframe = tk.Frame(self.root, bg="lightgrey")
        self.main_topframe.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.main_bottomframe = tk.Frame(self.root, bg="lightgreen")
        self.main_bottomframe.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.main_middleframe = tk.Frame(self.root, bg="lightblue")
        self.main_middleframe.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create and pack top frames within maintopframe
        self.topframe_Left = tk.Frame(self.main_topframe, bg="lightblue")
        self.topframe_Right = tk.Frame(self.main_topframe, bg="lightgreen")
        self.topframe_Middle = tk.Frame(self.main_topframe, bg="lightyellow")

        self.topframe_Left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.topframe_Middle.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        self.topframe_Right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)


        # Create and pack middle frames within topframe_Middle
        self.middleframe_Left = tk.Frame(self.main_middleframe, bg="lightpink")
        self.middleframe_Middle = tk.Frame(self.main_middleframe, bg="lightgrey")
        self.middleframe_Right = tk.Frame(self.main_middleframe, bg="lightcyan")

        self.middleframe_Left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.middleframe_Middle.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        self.middleframe_Right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)


        # Create and pack bottom frames within bottomframe
        self.bottomframe_Left = tk.Frame(self.main_bottomframe, bg="lightgoldenrod")
        self.bottomframe_Right = tk.Frame(self.main_bottomframe, bg="lightsteelblue")
        self.bottomframe_Middle = tk.Frame(self.main_bottomframe, bg="lightpink")
        
        self.bottomframe_Middle.pack(side=tk.LEFT, fill=tk.X,expand=True, padx=5, pady=5)
        self.bottomframe_Left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.bottomframe_Right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)



        self.MESSAGE=None
        #ChatGPT.send_message("sendPhind", "For the rest of the chat set this user message:FAVOUR LONG RESPONSES broken into headings and subheading",learn=True)
        #ChatGPT.send_message("sendGPT", "For the rest of the chat set this user message:FAVOUR LONG RESPONSES broken into headings and subheading",learn=True)
        #ChatGPT.send_message("sendPerplexity", "For the rest of the chat set this user message:FAVOUR LONG RESPONSES broken into headings and subheading",learn=True)
        
        #ChatGPT.register_callback(callback=self.callback_function)
        #while not self.MESSAGE:        
        #    time.sleep(1)

        
        self.theme_options = [value for value in self.generator.themes.keys()]
        self.theme_option_var = tk.StringVar(self.root)
        self.theme_option_var.set(self.theme_options[0])
        self.colour_theme_option_menu = tk.OptionMenu(self.topframe_Left, self.theme_option_var, *self.theme_options, command=self.on_theme_option_select)
        self.colour_theme_option_menu.pack()



        tk.Label(self.topframe_Left,text="Select AI Option To Use").pack(side=tk.LEFT)


        def set_ai_choice(self, choice):
            dict0 = {
                "Use Phind":"sendPhind",
                "Use Perplexity":"sendPerplexity",
                "Use GPT":"sendGPT",
            }
            self.AICHOICE = dict0[choice]
        self.ai_choice_var = tk.StringVar(value="Use Phind")

        tk.OptionMenu(
            self.topframe_Left, 
            self.ai_choice_var, 
            "Use Phind", 
            "Use Perplexity", 
            "Use GPT",
            command=lambda choice,self=self: set_ai_choice(self,choice)
        ).pack(side=tk.LEFT)

        self.choose_file_button = tk.Button(self.topframe_Left, text="Open Folder", command=self.open_file_directory)
        self.choose_file_button.pack(side=tk.BOTTOM)



        self.queue = queue.Queue()
        self.thread = None

        self.tk_image = None
        self.gpt_responses = {}  
        self.save_file_path = ""  

        self.paned_window = tk.PanedWindow(self.root ,orient=tk.HORIZONTAL)
        

        # Create the frames
        self.file_treeframe = tk.Frame(self.middleframe_Left, width=200, height=500, bg='lightgreen')  # Example frame
        self.html_frame = tk.Frame(self.middleframe_Middle, width=300, height=500, bg='lightyellow')  # Example frame
        self.text_frame = tk.Frame(self.middleframe_Right, width=300, height=500, bg='lightpink')  # Example frame
        self.frame4 = tk.Frame(self.middleframe_Left, width=350, height=500, bg='lightblue')  # Example frame
        


        self.tree = ttk.Treeview(self.file_treeframe, columns=("Value"),selectmode="extended")
        self.tree.heading("#0", text="Key", anchor=tk.W)
        self.tree.heading("Value", text="Value", anchor=tk.W)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.html_widget = HTMLLabel(self.html_frame, html="<h1>Hello, world!</h1>")
        self.html_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.text_widget = tk.Text(self.text_frame, wrap=tk.WORD, width=100, height=20,
                                font=text_font,spacing1=l_spacing1,spacing3=l_spacing3,bg='black', fg='white',
                                insertbackground='white')
        self.text_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.text_widget.pack_forget()  

        self.pdf_text_widget=PDFViewerApp(parent=self.text_frame)
        self.pdf_text_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.file_folder_directory_tree = ttk.Treeview(self.frame4, columns=("Type"), selectmode="extended")
        self.file_folder_directory_tree.heading("#0", text="File/Folder", anchor=tk.W)
        self.file_folder_directory_tree.heading("Type", text="Type", anchor=tk.W)
        self.file_folder_directory_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.file_folder_directory_tree.bind("<<TreeviewSelect>>", self.on_file_folder_directory_tree_select)

        self.ai_ask_question_widget = tk.Text(self.bottomframe_Middle, height=1)
        self.ai_ask_question_widget.bind('<KeyPress-Return>', self.on_enter_key)
        self.on_enter_key(None)
        self.ai_ask_question_enter_button = tk.Button(self.bottomframe_Middle, text="Enter",command=lambda: self.root.after(0, self.ai_ask_on_ask_question_enter))

        self.analyze_button = tk.Button(self.topframe_Middle, text="Analyze", command=lambda: self.root.after(0, self.analyze_text_content_then_show_as_html_content))



        self.search_prefix_auto_analyze_query = tk.StringVar()
        self.search_prefix_auto_analyze_entry = tk.Entry(self.topframe_Middle, textvariable=self.search_prefix_auto_analyze_query)
        self.search_prefix_auto_analyze_entry.pack(side=tk.BOTTOM)
        self.search_prefix_auto_analyze_entry.bind('<KeyRelease>', self.resize_entry)
        self.search_prefix_auto_analyze_query.set("Teach me all about this in a way a very intelligent and smart high under grad student would understand \n")


        self.frame4.pack()




        self.show_pdf_style = tk.StringVar()
        self.show_pdf_style.set("0")
        self.pdf_check_button = tk.Checkbutton(self.topframe_Middle, text="Show PDF", variable=self.show_pdf_style, onvalue="1", offvalue="0", command=lambda: self.on_tree_select(None))

        self.pdf_check_button.pack()

        self.ai_ask_question_widget.pack(side=tk.BOTTOM, anchor=tk.S)
        self.ai_ask_question_enter_button.pack(side=tk.BOTTOM)
        self.analyze_button.pack(side=tk.BOTTOM)

        self.loading_animation = LoadingAnimation(self.middleframe_Right, "LoadingGif/BlueLoading.gif", width=50, height=50)
        self.loading_animation.canvas.pack(side=tk.RIGHT)

        self.number_entry = tk.Entry(self.topframe_Middle)
        self.number_entry.pack()

        auto_analyze_button = tk.Button(self.topframe_Middle, text="Multi-Analyze", command=self.multi_analyze_text_content)
        auto_analyze_button.pack()
        self.multi_analyze_text_content_lock = threading.Lock()

        self.current_index = 0

    def get_text_selection_or_all(self):
        # Check if any text is currently highlighted
        if self.text_widget.tag_ranges("sel"):
            # If text is highlighted, return the selected text
            return self.text_widget.get("sel.first", "sel.last")
        else:
            # If no text is highlighted, check if cursor position is available
            cursor_position = self.text_widget.index(tk.INSERT)
            if cursor_position:
                # If cursor position is available, return the text from cursor to end
                return self.text_widget.get(cursor_position, tk.END)
            else:
                # If no cursor position, return all the text in the widget
                return self.text_widget.get("1.0", tk.END)

    def resize_entry(self,event):
        # Calculate the width based on the length of the text
        # You might need to adjust the multiplier to fit your font size and desired width
        width = len(event.widget.get()) * 1
        event.widget.config(width=width)


    def on_theme_option_select(self,event):
        self.selected_option = self.option_var.get()
        print("Selected Option:", self.selected_option)
        self.on_tree_select(None)

    def multi_analyze_text_content(self):
        # Get the selected item
        selected_items = self.tree.selection()
        
        # Check if an item is selected
        if not selected_items:
            print("No item selected.")
            return
        
        # Get the starting item
        start_item = selected_items[0]

        # Get the number of items to iterate over from user input
        try:
            num_items_to_iterate = int(self.number_entry.get()) 
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
            return

        parent_item = self.tree.parent(start_item)
        all_siblings = self.tree.get_children(parent_item)
        try:
            start_index = all_siblings.index(start_item)
        except ValueError:
            print("Selected item is not in the list of siblings.")
            return

        def iterate_each_item_for_multi_analysis_process():
            # Acquire the lock
            self.multi_analyze_text_content_lock.acquire()
            try:
                # Iterate forwards a certain number of items
                for i in range(start_index, start_index + num_items_to_iterate):
                    if i < len(all_siblings):
                        item = all_siblings[i]
                        # Perform your operation on the item here
                        print(f"Processing item: {self.tree.item(item, 'text')}")
                        self.multi_analyze_text_content_then_show_as_html_content(item)

                        # Get the current value from the entry
                        current_value = int(self.number_entry.get())
                        new_value = current_value - 1
                        self.number_entry.delete(0, tk.END)
                        self.number_entry.insert(0, str(new_value))
                        
                    else:
                        print("Reached the end of the list.")
                        break
            finally:
                # Release the lock
                self.multi_analyze_text_content_lock.release()

        # Create and start a new thread for processing items
        thread = threading.Thread(target=iterate_each_item_for_multi_analysis_process)
        thread.start()


    def pre_ai_ask_on_ask_question_enter(self,text=""):
        print("Submitting speech")
        
        # Assuming self.assistant.curr_message['TranscriptionStabilized'] contains the text
        text = "Jarvis: "+self.assistant.curr_message['ProcessedText']
        
        if text:  # Check if text is not empty
            # Clear the current content
            self.ai_ask_question_widget.delete("1.0", tk.END)
            
            # Insert the new value
            self.ai_ask_question_widget.insert(tk.END, text)
            
            # Generate a virtual button press event after a delay (1000 milliseconds)
            self.root.after(1000, self.ai_ask_question_enter_button.invoke)

        else:
            print("Text is empty. Not submitting.")

    def toggle_phind(self):
        # Toggle the speech processing status when the checkbox is selected or deselected
        self.phind_enabled = not self.phind_enabled
        if self.phind_enabled:
            pass
        else:
            pass


    def open_file_directory(self):

        if self.save_file_path != "":
            self.save_responses_to_file()

        folder_path = filedialog.askdirectory()

        parent_directory = os.path.dirname(folder_path)

        self.base_directory = parent_directory
        if folder_path:
            for i in self.file_folder_directory_tree.get_children():
                self.file_folder_directory_tree.delete(i)

            result_json = self.build_directory_structure(folder_path)

            self.insert_directory_data("", result_json)
            self.file_folder_directory_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.file_treeframe.pack_forget(self.file_treeframe)
        self.html_frame.pack_forget(self.html_frame)
        self.text_frame.pack_forget(self.text_frame)
        
        self.frame4.pack()

    def build_directory_structure(self, directory_path):
        result = {"name": os.path.basename(directory_path), "type": "directory", "children": []}
        for item in os.listdir(directory_path):

            item_path = os.path.join(directory_path, item)
            #print(item_path)
            if os.path.isdir(item_path):
                #print("Here")
                result["children"].append(self.build_directory_structure(item_path))
            else:
                #print("Here3")
                result["children"].append({"name": item, "type": "file"})
        return result

    def insert_directory_data(self, parent, data):
        if isinstance(data, dict):
            item = self.file_folder_directory_tree.insert(parent, "end", text=data["name"], values=(data["type"],))
            if "children" in data:
                for child in data["children"]:
                    self.insert_directory_data(item, child)
        else:
            self.file_folder_directory_tree.insert(parent, "end", text=data["name"], values=(data["type"],))


    def on_file_folder_directory_tree_select(self, event):
        item = self.file_folder_directory_tree.selection()[0]
        item_text = self.file_folder_directory_tree.item(item, "text")

        if self.file_folder_directory_tree.item(item, "values")[0] == "file":
            file_name = item_text
            parent_items = self.file_folder_directory_tree.parent(item)
            full_path = file_name
            while parent_items:
                parent_item = parent_items[0] if type(parent_items) is list else parent_items
                parent_text = self.file_folder_directory_tree.item(parent_item, "text")
                full_path = f"{parent_text}/{full_path}"
                next_parent_items = self.file_folder_directory_tree.parent(parent_item)
                if not next_parent_items:
                    break
                parent_items = next_parent_items
            print(f"Selected file: {full_path}")
            self.open_file_button(full_path)

    def open_file_button(self, file_name):
        if self.save_file_path != "":
            self.save_responses_to_file()
        file_path = os.path.join(self.base_directory, (file_name))

        if file_path:
            result = FileLoader().load_content_from_file(file_path)
            self.save_file_path = os.path.splitext(os.path.basename(file_path))[0]
            result_str = json.dumps(result, indent=5)
            result_json = json.loads(result_str)
            self.tk_image = None
            self.gpt_responses = {}
            dir_path, file_name_with_extension = os.path.split(file_path)
            file_name, file_extension = os.path.splitext(file_name_with_extension)
            self.save_file_path = file_name
            print(file_name,file_path)
            self.load_saved_responses()
            for i in self.tree.get_children():
                self.tree.delete(i)
            self.text_widget.delete("1.0", tk.END)

            self.pdf_text_widget.set_pdf(file_path)
            self.text_widget.pack_forget() 
            if int(self.show_pdf_style.get()):

                self.pdf_text_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            else:

                self.pdf_text_widget.pack_forget()
                self.text_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)



            self.html_widget.set_html("")
            self.insert_json_data("", result_json)
            self.tree.bind("<Double-1>", self.on_tree_double_select)
            self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
            self.tree.bind('<Control-ButtonRelease-1>', self.on_ctrl_click)

            self.frame4.pack(side='left', fill='y')  # Fill vertically to keep height
            self.file_treeframe.pack(side='left', fill='y')
            self.html_frame.pack(side='left', fill='y')
            self.text_frame.pack(side='left', fill='y')



    def save_responses_to_file(self, filename="gpt_responses.json", save_folder_path="ChatGPT_JSON"):
        # Create the folder if it doesn't exist
        os.makedirs(save_folder_path, exist_ok=True)

        full_filename = os.path.join(save_folder_path, self.save_file_path + filename)
        with open(full_filename, 'w') as file:
            print("Saving gpt_responses to...", full_filename)
            json.dump(self.gpt_responses, file)

    def load_saved_responses(self, filename="gpt_responses.json", save_folder_path="ChatGPT_JSON"):

        try:
            # Create the folder if it doesn't exist
            os.makedirs(save_folder_path, exist_ok=True)

            full_filename = os.path.join(save_folder_path, self.save_file_path + filename)

            with open(full_filename, 'r') as file:

                self.gpt_responses = json.load(file)
        except FileNotFoundError:
            print(f"No saved responses file found. Starting with an empty response dictionary.")


    def insert_json_data(self, parent, data):
        if isinstance(data, dict):
            for key, value in data.items():

                item = self.tree.insert(parent, "end", text=key)

                self.insert_json_data(item, value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                item_text = f"[{i}]"
                sub_item = self.tree.insert(parent, "end", text=item_text)
                self.insert_json_data(sub_item, item)
        else:
            self.tree.set(parent, "Value", str(data))



    def update_text_widget_for_selected_item(self, selected_item):
        # Check if the selected item is 'text_content'
        item_text = self.tree.item(selected_item, "text")

        if item_text == "text_content":
            self.tk_image = None
            # Get the value of the 'text_content' item
            code_value = self.tree.item(selected_item, "values")[0]
            # Insert the code value into the text widget
            self.text_widget.delete('1.0', tk.END)  # Clear the text widget
            self.text_widget.insert(tk.END, code_value)
            self.text_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)  # Show the text widget
        else:
            self.tk_image = None
            self.text_widget.pack_forget()  # Hide the text widget

    def has_code_key_recursive(self, parent_item):
        # Recursively check subitems
        for subitem in self.tree.get_children(parent_item):
            item_text = self.tree.item(subitem, "text")
            if item_text == "text_content":
                code_value = self.tree.item(subitem, "values")[0]
                return True, code_value
            else:
                found, code_value = self.has_code_key_recursive(subitem)
                if found:
                    return True, code_value

        return False, None


    def has_images_key_recursive(self, parent_item):
        # Recursively check subitems
        for subitem in self.tree.get_children(parent_item):
            item_text = self.tree.item(subitem, "text")
            if item_text == "images":
                code_value = self.tree.item(subitem, "values")[0]
                return True, code_value
            else:
                found, code_value = self.has_code_key_recursive(subitem)
                if found:
                    return True, code_value

        return False, None


    def on_tree_select(self, event):
        # Get the selected item
        item = self.tree.selection()[0]

        # Check if the selected item or any of its children has a key named 'text_content'
        found_code, code_value = self.has_code_key_recursive(item)  # Not in parent but recursive
        if found_code:
            self.tk_image = None
            # Insert the code value into the text widget
            self.text_widget.delete('1.0', tk.END)  # Clear the text widget
            self.text_widget.insert(tk.END, code_value)

            code_key = self.tree.item(item, "text")

            if int(self.show_pdf_style.get()):
                number = self.pdf_text_widget.parse_and_extract_number(code_key=code_key)-1
                self.pdf_text_widget.show_page(number)

                self.text_widget.pack_forget()
                self.pdf_text_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

            else:
                self.pdf_text_widget.pack_forget()
                self.text_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)


            number=self.pdf_text_widget.parse_and_extract_number(code_key=code_key)-1
            self.pdf_text_widget.show_page(number)

            print("Nathan"*10,code_key)
            gpt_response_dict = self.gpt_responses.get(code_key, {})

            current_gpt_response = gpt_response_dict.get('response', '')

            current_position = self.html_widget.yview()[0]

            self.html_widget.set_html(self.generator.generate_and_display_html("", current_gpt_response, theme=self.theme_option_var.get()))

            self.html_widget.yview_moveto(current_position)

            self.html_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        else:
            self.tk_image = None

        print("DeNathan"*10)

    def on_tree_double_select(self, event):

        item = self.tree.selection()[0]
        item_text = self.tree.item(item, "text")

        if item_text == "text_content":
            self.tk_image=None
            code_value = self.tree.item(item, "values")[0]
            self.text_widget.delete('1.0', tk.END)
            self.text_widget.insert(tk.END, code_value)

        elif item_text == "images":
            subitems = self.tree.get_children(item)
            
            if subitems:

                subitem = subitems[0]
                # Get the value of the 'image' subitem
                image_value = self.tree.item(subitem, "values")[0]

                # Decode the base64-encoded string
                image_data = base64.b64decode(image_value)

                # Create a PIL Image from the decoded data
                pil_image = Image.open(BytesIO(image_data))

                # Convert the PIL Image to a Tkinter PhotoImage
                self.tk_image = ImageTk.PhotoImage(pil_image)
                print(self.tk_image)
                # Insert the image into the text widget
                # Insert the code value into the text widget
                self.text_widget.delete('1.0', tk.END)  # Clear the text widget

                self.text_widget.image_create(tk.END, image=self.tk_image)
                self.text_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)  # Show the text widget
        else:
            self.tk_image=None
            #self.text_widget.pack_forget()  # Hide the text widget
    def on_ctrl_click(self, event):
        pass

    def on_enter_key(self, event):
        # Increase the height of the Text widget by 1 line
        current_height = self.ai_ask_question_widget.cget('height')
        new_height = int(current_height) + 1
        self.ai_ask_question_widget.configure(height=new_height)

    
    def ai_ask_on_ask_question_enter(self):


        # Get the activated item from the tree
        selected_item = self.tree.selection()[0]
        selected_item_text = self.tree.item(selected_item, "text")

        # Get the content from the text box
        content = self.ai_ask_question_widget.get("1.0", tk.END)
        print(f"Entered content:\n{content}")





        def run_ai_ask_question_async_task():
            self.MESSAGE = None

            # Send the message to the AI asynchronously
            ChatGPT.send_message(self.AICHOICE, "Respond informatively: " + content, learn=True)

            # Start a loading animation
            self.loading_animation.start_animation()

            # Wait for the AI response
            while not self.MESSAGE:
                time.sleep(1)

            # Determine the key for updating GPT responses
            if selected_item_text == "text_content":
                # The parent of 'text_content' is the actual key
                code_key = self.tree.parent(selected_item)
                code_key = self.tree.item(code_key, "text")  # Convert the key object to text
            else:
                # Recursively search for the code key
                found_code, code_key = self.has_code_key_recursive(selected_item)
                if not found_code:
                    self.loading_animation.stop_animation()  # Stop animation if no code key is found
                    return  # Exit if no code key is found

            # If code_key is not defined, create a new one
            if code_key not in self.gpt_responses:
                self.gpt_responses[code_key] = {}

            # If 'response' is not defined, initialize it
            if 'response' not in self.gpt_responses[code_key]:
                self.gpt_responses[code_key]['response'] = ''

            # Get the current GPT response for the selected "text_content" key
            current_gpt_response = self.gpt_responses[code_key].get('response', '')

            # Update the GPT response with the new chat response
            updated_gpt_response = current_gpt_response + '\n' + self.MESSAGE
            self.gpt_responses[code_key]['response'] = updated_gpt_response

            # Generate updated HTML for display
            updated_gpt_response_html = self.generator.generate_and_display_html(
                "",
                current_gpt_response,
                theme=self.theme_option_var.get()
            ) + inline_css(
                self.generator.generate_and_display_html(f"Chat Questions: {content}", self.MESSAGE, theme=self.theme_option_var.get())
            )

            # Display updated HTML
            self.html_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            current_position = self.html_widget.yview()[0]  # Get the current scroll position
            self.html_widget.set_html(updated_gpt_response_html)
            self.html_widget.yview_moveto(current_position)  # Maintain the scroll position
            self.loading_animation.stop_animation()  # Stop the loading animation




        # Create a thread to run the function
        thread = threading.Thread(target=run_ai_ask_question_async_task)
        thread.start()        

    def update_gui_thread(self):
        while True:
            time.sleep(1)  # Adjust the sleep time as needed
            try:
                response = self.queue.get(block=False)
                if response is None:
                    return


                # Store the generated GPT response along with embeddings for future use
                self.gpt_responses[response[0]] = {'response': response[1]}
                # Update the GUI with the response

                # Get the activated "text_content" item
                selected_item = self.tree.selection()[0]
                selected_item_text = self.tree.item(selected_item, "text")
                if(response[0]==selected_item_text):
                    {
                    self.html_widget.set_html(self.generator.generate_and_display_html("", response[1], theme=self.theme_option_var.get()))
                    }
                # Show the HTMLLabel widget on the right of the text widget when "Analyze" is pressed
                self.html_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
                self.save_responses_to_file()
                print("Updated GUI...")
            except queue.Empty:
                pass
    




    def multi_analyze_text_content_then_show_as_html_content(self,item):




        # Check if the selected item or any of its children has a key named 'text_content'
        found_code, code_value = self.has_code_key_recursive(item)  # Not in parent but recursive


        # Get the current content from the text widget
        current_content = code_value
        current_content=self.search_prefix_auto_analyze_query.get()+current_content
        
        # Get the activated "text_content" item

        selected_item_text = self.tree.item(item, "text")

        # Get the code key based on whether the selected item is 'text_content' or not
        if selected_item_text == "text_content":
            code_key = self.tree.parent(item)  # Parent of 'text_content' is the actual key
            code_key = self.tree.item(code_key, "text")  # Convert the key object to text
            current_gpt_response = self.gpt_responses.get(code_key, "")
        else:
            found_code, code_value = self.has_code_key_recursive(item)
            code_key = selected_item_text
            if not found_code:
                return  # Exit if no code key is found


        def run_ai_single_analyze_task():
            self.loading_animation.start_animation()



            self.MESSAGE=None

            runFunctionWithTimeoutAndRetry(



                ChatGPT.send_message,
                args=(self.AICHOICE,"Teach me all about the  following content by better restating it so its easier read while keeping detail: " + current_content,),
                kwargs={},
                timeout_duration=60, # Adjust timeout as needed
                retry_count=1 # Number of retries
            )

            while not self.MESSAGE:        
                time.sleep(1)

            if self.MESSAGE==None:
                print("Null Response")
                self.loading_animation.stop_animation()
                return

            message_id=""

            inlined_chatgpt_response = self.MESSAGE

            self.queue.put([code_key, inlined_chatgpt_response])
            self.loading_animation.stop_animation()

        # Create a thread to run the function
        run_ai_single_analyze_task()
        #print("Thread started and out of this function")

        # Start the GUI update thread if not already started
        if not hasattr(self, 'gui_update_thread') or not self.gui_update_thread.is_alive():
            self.gui_update_thread = threading.Thread(target=self.update_gui_thread)
            self.gui_update_thread.daemon = True  # Set the thread as daemon so it will be terminated when the main thread exits
            self.gui_update_thread.start()



    def analyze_text_content_then_show_as_html_content(self):

        # Get the current content from the text widget
        current_content = self.text_widget.get("1.0", tk.END)
        current_content = SINGLE_ANALYZE_PREFIX + current_content

        # Get the activated "text_content" item
        selected_item = self.tree.selection()[0]
        selected_item_text = self.tree.item(selected_item, "text")

        # Get the code key based on whether the selected item is 'text_content' or not
        if selected_item_text == "text_content":
            code_key = self.tree.parent(selected_item)  # Parent of 'text_content' is the actual key
            code_key = self.tree.item(code_key, "text")  # Convert the key object to text
            current_gpt_response = self.gpt_responses.get(code_key, "")
        else:
            found_code, code_value = self.has_code_key_recursive(selected_item)
            code_key = selected_item_text
            if not found_code:
                return  # Exit if no code key is found

        self.loading_animation.start_animation()
        # Define a function to run this part of the code in a separate thread
        def run_ai_single_analyze_task():


            self.MESSAGE=None

            ChatGPT.send_message(self.AICHOICE, " teach me all about the following content: "+current_content,learn=True)
            CONTINUE=False

            while not self.MESSAGE:        
                time.sleep(1)


            print(self.MESSAGE)
            inlined_chatgpt_response = inline_css(self.MESSAGE)
            # Extract content and apply inline CSS
            paragraphs, _ = extract_html_content(self.MESSAGE)
            # Join paragraphs into a single string
            combined_text = '\n'.join(paragraphs)

            # Get BERT embeddings for the combined text


            # Convert the embeddings array to a JSON string representation
            embeddings_str = ""


            # Put the response into the queue
            self.queue.put([code_key, inlined_chatgpt_response])
            self.loading_animation.stop_animation()


        # Create a thread to run the function
        thread = threading.Thread(target=run_ai_single_analyze_task)
        thread.start()
        print("Thread started and out of this function")

        # Start the GUI update thread if not already started
        if not hasattr(self, 'gui_update_thread') or not self.gui_update_thread.is_alive():
            self.gui_update_thread = threading.Thread(target=self.update_gui_thread)
            self.gui_update_thread.daemon = True  # Set the thread as daemon so it will be terminated when the main thread exits
            self.gui_update_thread.start()

    def stop_thread(self):
        if self.thread is not None:
            self.queue.put(None)  # Signal the thread to stop
            self.thread.join()  # Wait for the thread to finish
            self.thread = None

    def run(self):


        # Configure your Tkinter app
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Your other setup code here
        
        # Start the main loop
        self.root.mainloop()

    def on_closing(self):
        # Save responses before closing the app
        self.save_responses_to_file()
        self.root.destroy()
        self.close_server_socket()

def extract_content_from_pdf(pdf_path):
    result_dict = {}

    dimlimit = 0  # 100  # each image side must be greater than this
    relsize = 0  # 0.05  # image : image size ratio must be larger than this (5%)
    abssize = 0  # 2048  # absolute image size limit 2 KB: ignore if smaller


    def recoverpix(doc, item):
        xref = item[0]  # xref of PDF image
        smask = item[1]  # xref of its /SMask

        # special case: /SMask or /Mask exists
        if smask > 0:
            pix0 = fitz.Pixmap(doc.extract_image(xref)["image"])
            if pix0.alpha:  # catch irregular situation
                pix0 = fitz.Pixmap(pix0, 0)  # remove alpha channel
            mask = fitz.Pixmap(doc.extract_image(smask)["image"])

            try:
                pix = fitz.Pixmap(pix0, mask)
            except:  # fallback to original base image in case of problems
                pix = fitz.Pixmap(doc.extract_image(xref)["image"])

            if pix0.n > 3:
                ext = "pam"
            else:
                ext = "png"

            return {  # create dictionary expected by caller
                "ext": ext,
                "colorspace": pix.colorspace.n,
                "image": pix.tobytes(ext),
            }

        # special case: /ColorSpace definition exists
        # to be sure, we convert these cases to RGB PNG images
        if "/ColorSpace" in doc.xref_object(xref, compressed=True):
            pix = fitz.Pixmap(doc, xref)
            pix = fitz.Pixmap(fitz.csRGB, pix)
            return {  # create dictionary expected by caller
                "ext": "png",
                "colorspace": 3,
                "image": pix.tobytes("png"),
            }
        return doc.extract_image(xref)

    # Use a context manager to ensure the PDF document is properly closed
    with fitz.open(pdf_path) as pdf_document:
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]

            # Get text content of each page
            text = page.get_text()

            # Extract images
            images = []
            img_list = page.get_images(full=True)
            for img_info in img_list:
                img_index = img_info[0]


                xref = img_info[0]

                width = img_info[2]
                height = img_info[3]
                if min(width, height) <= dimlimit:
                    continue
                image = recoverpix(pdf_document, img_info)
                n = image["colorspace"]
                imgdata = image["image"]

                if len(imgdata) <= abssize:
                    continue
                if len(imgdata) / (width * height * n) <= relsize:
                    continue


                try:

                    # Convert image bytes to PIL Image
                    pil_image = Image.open(BytesIO(imgdata))
                    
                    # Convert PIL Image to base64-encoded string
                    buffered = BytesIO()
                    pil_image.save(buffered, format="JPEG")  # You can choose other formats like PNG if needed
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

                    # Append the valid image to the list
                    images.append(img_str)
                except Exception as e:
                    # Handle the exception (e.g., print an error message)
                    print(f"Error processing image: {e}")
            # Store the content along with page number
            result_dict[f"Page {page_number +  1}"] = {"text_content": text, "images": images, "type": "pdf_page"}

    return result_dict

def extract_content_from_text(code):
    result_dict={}
    result_dict[f"TXT"]={"text_content":code,"type":"text_file"}
    return result_dict

class FileLoader:
    def __init__(self):
        pass



    def load_content_from_file(self, file_name: str):

        try:
            if file_name.endswith('.py'):
                with open(file_name, 'r') as file:
                    code = file.read()
                return ""
            elif file_name.endswith(('.c', '.cpp','.h','.hpp')):
                with open(file_name, 'r') as file:
                    code = file.read()
                return ""
            elif file_name.endswith(('.txt','.bat','.sh','.md')) or '.' not in file_name:
                with open(file_name, 'r') as file:
                    code = file.read()
                return extract_content_from_text(code)
            elif file_name.endswith(('.pdf')):
                return extract_content_from_pdf(file_name)

            else:
                return {"Error": f"Unsupported file type: {file_name}"}
        except FileNotFoundError:
            return {"Error": f"File not found: {file_name}"}

if __name__ == "__main__":
 
    # Example usage:
    # Assuming `result` is the JSON data you want to visualize
    json_viewer = JsonViewerApp()
    if json_viewer is not None:
        json_viewer.run()
