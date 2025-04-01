
import ast
import math

from typing import Any, Union
import astor
import re
import queue
import os
import textwrap
import json
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
# Cause Process.start re imports everything in alternative cpu core and that basically reruns global code
import ChatGPT
from html import escape
from tkhtmlview import HTMLLabel
from tkinter import Tk, Button, filedialog
import cssutils
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from PIL import Image, ImageTk
import base64
import os
import threading
import time

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

from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Instantiate the Sentence Transformer model
model = SentenceTransformer('bert-base-nli-mean-tokens')

def get_bert_embeddings(passages):
    """
    Get BERT embeddings for a list of passages or a single passage.
    
    Args:
        passages (str or list of str): Passage(s) to encode into embeddings.
        
    Returns:
        embeddings (numpy array or list of numpy arrays): BERT embeddings for each passage.
    """
    if isinstance(passages, str):  # Check if passages is a single string
        passages = [passages]  # Convert to a list with one element

    embeddings = model.encode(passages)
    return embeddings

def calculate_cosine_similarity(embeddings1, embeddings2):
    """
    Calculate cosine similarity between two sets of embeddings.
    
    Args:
        embeddings1 (numpy array): Embeddings for the first set of passages.
        embeddings2 (numpy array): Embeddings for the second set of passages.
        
    Returns:
        similarity_scores (numpy array): Cosine similarity scores between corresponding pairs of embeddings.
    """
    similarity_scores = cosine_similarity(embeddings1, embeddings2)
    return similarity_scores

def calculate_similarity_between_passages(passage1, passage2):
    """
    Calculate similarity between two passages.
    
    Args:
        passage1 (str): First passage.
        passage2 (str): Second passage.
        
    Returns:
        similarity_score (float): Cosine similarity score between the two passages.
    """
    # Get BERT embeddings for the passages
    embeddings = get_bert_embeddings([passage1, passage2])

    # Calculate cosine similarity between the embeddings
    similarity_score = calculate_cosine_similarity(embeddings[0:1], embeddings[1:2])

    return similarity_score[0][0]


import time
from queue import Queue
from threading import Thread
import sys
import multiprocessing as mp


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
"""
class SophiaAssistant:
    def __init__(self):
        self.recorder = AudioToTextRecorder(
            spinner=False,
            model='small.en',
            language='en',
            on_wakeword_detected=lambda: ( play_audio(JARVIS_START),print("Wake Word detected"),self.start_callback()),
            on_recording_stop=lambda: ( play_audio(JARVIS_PROCESSING), self.recording_stopped()),
            on_wakeword_timeout=lambda:(play_audio(JARVIS_PROCESSING),print("Deactivated cause no audio")),
            on_recording_start=lambda: print("Speak, we're listening now"),
            enable_realtime_transcription=True,
            on_realtime_transcription_update=self.realtime_transcribed_update,
            on_realtime_transcription_stabilized=self.realtime_transcribed_update_stabilized,
            silero_sensitivity=0.4,
            silero_use_onnx=True,
            webrtc_sensitivity=2,
            post_speech_silence_duration=1.5,
            min_length_of_recording=0,
            min_gap_between_recordings=0,
            realtime_processing_pause=0.2,
            realtime_model_type='tiny.en',
            wake_words='jarvis'
        )

        self.is_turned_on = False
        self.message_queues = {
            'RecordingStart': Queue(),
            'RecordingStop': Queue(),
            'TranscriptionUpdate': Queue(),
            'TranscriptionStabilized': Queue(),
            'ProcessedText': Queue(),
        }

        self.curr_message = {
            'RecordingStart': '',
            'RecordingStop': '',
            'TranscriptionUpdate': '',
            'TranscriptionStabilized': '',
            'ProcessedText': '',
        }

        self.loop_thread = Thread(target=self._turn_on_loop)
        self.loop_thread.start()  # Start the thread
        # Callback functions
        self.start_callback = None
        self.stop_callback = None
        self.transcription_update_callback = None
        self.transcription_stabilized_callback = None
        self.processed_text_callback = None
        self.current_transcription_update = ""
        self.current_transcription_stabilized = ""

    def recording_started(self, *args, **kwargs):
        text = args[0] if args else ""
        self.message_queues['RecordingStart'].put(f"Recording started: {text}")
        self.curr_message['RecordingStart'] = f"Recording started: {text}"
        if self.start_callback:
            self.start_callback(text)

    def realtime_transcribed_update(self, *args, **kwargs):
        text = args[0] if args else ""
        if self.is_turned_on:
            self.current_transcription_update = f"Real-time transcription update: {text}"
            self.curr_message['TranscriptionUpdate'] = self.current_transcription_update
            # If callback is set, update it
            if self.transcription_update_callback:
                self.transcription_update_callback(self.current_transcription_update)

    def realtime_transcribed_update_stabilized(self, *args, **kwargs):
        text = args[0] if args else ""
        if self.is_turned_on:
            self.current_transcription_stabilized = f"Real-time transcription stabilized: {text}"
            self.curr_message['TranscriptionStabilized'] = self.current_transcription_stabilized
            # If callback is set, update it
            if self.transcription_stabilized_callback:
                self.transcription_stabilized_callback(self.current_transcription_stabilized)

    def recording_stopped(self, *args, **kwargs):
        text = args[0] if args else ""

        # If turned on, put the current updates into the queue
        if self.is_turned_on:
            if self.current_transcription_update:
                self.message_queues['TranscriptionUpdate'].put(self.current_transcription_update)
                self.curr_message['TranscriptionUpdate'] = self.current_transcription_update
                self.current_transcription_update = ""
            if self.current_transcription_stabilized:
                self.message_queues['TranscriptionStabilized'].put(self.current_transcription_stabilized)
                self.curr_message['TranscriptionStabilized'] = self.current_transcription_stabilized
                self.current_transcription_stabilized = ""

        # Put the recording stopped message into the queue
        self.message_queues['RecordingStop'].put(f"Recording stopped: {text}")
        self.curr_message['RecordingStop'] = f"{text}"

        # If stop callback is set, update it
        if self.stop_callback:
            self.stop_callback()

    def process_sentence(self, *args, **kwargs):
        text = args[0] if args else ""
        self.message_queues['ProcessedText'].put(f"Processed text: {text}")
        self.curr_message['ProcessedText']=f"{text}"
        if self.processed_text_callback:
            self.processed_text_callback(text)

    def _turn_on_loop(self):
        while True:
            if self.is_turned_on:
                self.recorder.text(self.process_sentence)
            time.sleep(0.1)  # Use time.sleep instead of asyncio.sleep


    def turn_on(self):
        self.is_turned_on = True
        return self  # Return the instance to allow method chaining

    def turn_off(self):
        self.is_turned_on = False
        return self  # Return the instance to allow method chaining

    def force_turn_off(self):
        self.is_turn_on = False
        self.recorder.stop()
        return self

    def set_start_callback(self, callback):
        self.start_callback = callback
        return self  # Return the instance to allow method chaining

    def set_stop_callback(self, callback):
        self.stop_callback = callback
        return self  # Return the instance to allow method chaining

    def set_transcription_update_callback(self, callback):
        self.transcription_update_callback = callback
        return self  # Return the instance to allow method chaining

    def set_transcription_stabilized_callback(self, callback):
        self.transcription_stabilized_callback = callback
        return self  # Return the instance to allow method chaining

    def set_processed_text_callback(self, callback):
        self.processed_text_callback = callback
        return self  # Return the instance to allow method chaining

    def get_messages(self, message_type):
        if message_type in self.message_queues:
            messages = list(self.message_queues[message_type].queue)
            self.message_queues[message_type].queue.clear()
            return messages
        else:
            return []
"""
import socket
from PIL import Image, ImageTk

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
            resized_frame = gif.resize((width, height), Image.ANTIALIAS)
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







import markdown

class HTMLGenerator:
    def __init__(self):
        self.themes = {
            "default": ("red", "green", "blue", "purple"),
            "light": ("#333333", "#007ACC", "#009688", "#FF5722"),
            "dark": ("#FFFFFF", "#FFD600", "#FF4081", "#4CAF50"),
            "monochrome": ("#212121", "#795548", "#607D8B", "#FF5722"),
            "nature": ("#263238", "#43A047", "#EF6C00", "#009688"),
            "ocean": ("#37474F", "#FFD600", "#0288D1", "#FFA000"),
            "sunset": ("#263238", "#FF7043", "#EF5350", "#FFC107"),
            "forest": ("#37474F", "#4CAF50", "#689F38", "#795548"),
            "autumn": ("#3E2723", "#FF6F00", "#8D6E63", "#FFAB91"),
            "pastel": ("#424242", "#7986CB", "#AED581", "#FF8A65"),
            "space": ("#FFFFFF", "#FFD600", "#B39DDB", "#4CAF50"),
            "desert": ("#5D4037", "#FF8F00", "#FFA000", "#4E342E"),
            "spring": ("#37474F", "#43A047", "#4CAF50", "#FFC107"),
            "rainbow": ("#000000", "#FF0000", "#FF7F00", "#FFFF00"),
            "vintage": ("#3E2723", "#795548", "#8D6E63", "#FFAB91"),
            "winter": ("#263238", "#039BE5", "#03A9F4", "#4CAF50")
        }
        self.sub_section_colors = {
            "default": ["#f2f2f2", "#e6e6e6"],
            "light": ["#f0f8ff", "#e0f7fa"],
            "dark": ["#ffe0b2", "#ffcc80"],
            "monochrome": ["#f5f5f5", "#e0e0e0"],
            "nature": ["#c8e6c9", "#a5d6a7"],
            "ocean": ["#e1f5fe", "#b3e5fc"],
            "sunset": ["#ffccbc", "#ffab91"],
            "forest": ["#dcedc8", "#c5e1a5"],
            "autumn": ["#ffecb3", "#ffe082"],
            "pastel": ["#f8bbd0", "#f48fb1"],
            "space": ["#b2ebf2", "#80deea"],
            "desert": ["#ffecb3", "#ffe082"],
            "spring": ["#c8e6c9", "#a5d6a7"],
            "rainbow": ["#f5f5f5", "#e0e0e0"],
            "vintage": ["#ffe0b2", "#ffcc80"],
            "winter": ["#e1f5fe", "#b3e5fc"]
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

    def generate_html_response(self,prompt, response_title, colors, section_colors, html_content, theme="default", use_backgrounding=True):

        color1, color2, color3, color4 = colors
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract headings, paragraphs, ordered and unordered lists, and strong tags
        sections = soup.find_all(['h3', 'h4', 'p', 'ol', 'ul', 'li', 'strong'])
        content_html = ""
        #in_section = False  # Flag to track whether we are currently inside a section

        for section_index, section in enumerate(sections):
            section_color = section_colors[section_index % len(section_colors)]

            if section.name == 'h3':
                content_html += f'<div class="sub-section" style="background-color: {section_color};">\n<h2>{section.text}</h2>\n'
                #in_section = True
            elif section.name == 'h4':
                content_html += f'<div class="sub-section" style="background-color: {section_color};">\n<h3>{section.text}</h3>\n'
                #in_section = True
            elif section.name == 'p':
                if section.parent.name=='li':
                    pass
                elif section.find('strong'):
                    pass
                else:
                    #if not in_section:
                    content_html += f'<div class="sub-section" style="background-color: {section_color};">\n'
                    #    in_section = True
                    content_html += f'<p>{section.text}</p>\n'
            elif section.name in ['ol', 'ul']:
                #if not in_section:
                content_html += f'<div class="sub-section" style="background-color: {section_color};">\n'
                #    in_section = True
                content_html += f'<{section.name}>\n'
            elif section.name == 'li':
                # Check if the li contains a strong tag
                strong_tag = section.find('strong')
                if strong_tag:
                    pass
                    #skip
                    # If yes, include the strong tag content in the list item
                    #content_html += f'<li><strong>{strong_tag.text}</strong>{section.text.replace(strong_tag.text, "")}</li>\n'
                else:
                    content_html += f'<li>{section.text}</li>\n'
            elif section.name == 'strong':
                # Treat strong tag as heading
                if section.parent.name == 'p':
                    content_html += f'<h3>{section.text}</h3>'
                    strong_text = section.text
                    # If strong tag is inside a list item, add the remaining text in the list item as a paragraph
                    strong_text = section.text
                    li_content = section.parent.text.replace(strong_text, '', 1).strip()
                    content_html += f'<p>{li_content}</p>\n'

                else:
                    # Treat it as a standalone section
                    content_html += f'<div class="sub-section" style="background-color: {section_color};">\n<h3>{section.text}</h3>\n'
                    if section.parent.name == 'li':
                        # If strong tag is inside a list item, add the remaining text in the list item as a paragraph
                        strong_text = section.text
                        li_content = section.parent.text.replace(strong_text, '', 1).strip()
                        content_html += f'<p>{li_content}</p>\n'

        body_bg_color = "#f8f8f8" if use_backgrounding else "transparent"
        html_template = f'<!DOCTYPE html> \
    <html lang="en"> \
    <head> \
        <meta charset="UTF-8"> \
        <meta name="viewport" content="width=device-width,initial-scale=1.0"> \
        <title>{response_title}</title> \
        <style> \
            body {{ \
                font-family: Arial, sans-serif; \
                background-color: {body_bg_color}; \
                margin: 0; \
                padding: 10px; \
            }} \
            h1 {{ \
                color: {color1}; \
                font-size: 20px; \
                margin-bottom: 8px; \
            }} \
            h2 {{ \
                color: {color2}; \
                font-size: 18px; \
                margin-bottom: 6px; \
            }} \
            h3 {{ \
                color: {color3}; \
                font-size: 16px; \
                margin-bottom: 4px; \
            }} \
            h4 {{ \
                color: {color4}; \
                font-size: 14px; \
                margin-bottom: 4px; \
            }} \
            p {{ \
                line-height: 1.4; \
                margin-bottom: 8px; \
            }} \
            section {{ \
                background-color: #ffffff; \
                border-radius: 5px; \
                margin-bottom: 10px; \
                padding: 15px; \
            }} \
            .sub-section {{ \
                border-radius: 5px; \
                padding: 10px; \
                margin-bottom: 8px; \
            }} \
        </style> \
    </head> \
    <body> \
        <section>  \
            <p>{prompt}</p> \
            <div class="sub-section"> \
            </div> \
        </section> \
        <section> \
            {content_html} \
        </section> \
    </body> \
    </html>'
        return self.inline_css(html_template)

    def respond_with_html(self, prompt, contents, theme="default", use_backgrounding=True):
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
        html_response = self.respond_with_html(prompt, markdown.markdown(contents), theme, use_backgrounding)
        #self.display_html(html_response)
        return html_response



import io
from tkinter import filedialog
import threading
from tkinter.scrolledtext import ScrolledText



from tkinter.scrolledtext import ScrolledText

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


import base64
import pygame
import requests
from io import BytesIO
import tkinter as tk
from tkinter import ttk

import tkinter as tk
import os
import pickle
import hashlib
import tkinter as tk
import os
import pickle
import hashlib

class DraggableWidget:

    instances = {}  # Dictionary to store instances keyed by unique identifier

    def __init__(self, widget, master=None, grid_size=15, **kwargs):
        # Remove the 'key' command from kwargs if it exists
        kwargs.pop('command', None)
    
        self.master = master
        self.dragging = False
        self.widget = widget
        self.grid_size = grid_size
        self.widget_name = self.encode_widget_name(widget, **kwargs)  # Encoded widget name
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

    def encode_widget_name(self, widget, **kwargs):
        # Combine widget name and arguments into a string
        widget_info = f"{type(widget).__name__}_{sorted(kwargs.items())}"
        # Encode the widget info using SHA-256 hash
        hashed_widget_info = hashlib.sha256(widget_info.encode()).hexdigest()
        return hashed_widget_info


    def load_position(self,event=None):

        position_file = os.path.join("Widget_Position", f"{self.widget_name}_position.pkl")
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
        position_file = os.path.join("Widget_Position", f"{self.widget_name}_position.pkl")
        os.makedirs("Widget_Position", exist_ok=True)  # Create the folder if it doesn't exist
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
    return DraggableWidget(label, master, **kwargs)

def create_draggable_entry(master=None, **kwargs):
    entry = EntryCustom(master, **kwargs)
    return DraggableWidget(entry, master, **kwargs)

def create_draggable_text(master=None, **kwargs):
    pass
    # text = TextCustom(master, **kwargs)
    # return DraggableWidget(text, master, **kwargs)

def create_draggable_checkbutton(master=None, **kwargs):
    checkbutton = CheckbuttonCustom(master, **kwargs)
    return DraggableWidget(checkbutton, master, **kwargs)

def create_draggable_button(master=None, **kwargs):
    button = ButtonCustom(master, **kwargs)
    return DraggableWidget(button, master, **kwargs)

def create_draggable_panedwindow(master=None, **kwargs):
    paned_window = PanedWindowCustom(master, **kwargs)
    return DraggableWidget(paned_window, master, **kwargs)

def create_draggable_optionsmenu(master=None, **kwargs):
    options_menu = None#OptionsMenuCustom(master, **kwargs)
    return DraggableWidget(options_menu, master, **kwargs)

# Custom widget classes
LabelCustom = tk.Label
EntryCustom = tk.Entry
# TextCustom = tk.Text  # Uncomment if needed
CheckbuttonCustom = tk.Checkbutton
ButtonCustom = tk.Button

PanedWindowCustom = tk.PanedWindow
# Assign custom classes to tkinter widget classes
tk.Label = create_draggable_label
tk.Entry = create_draggable_entry
# tk.Text = create_draggable_text  # Uncomment if needed
tk.Checkbutton = create_draggable_checkbutton
tk.Button = create_draggable_button
tk.PanedWindow = create_draggable_panedwindow

class JsonViewerApp:

    def __init__(self, text_font=("Trebuchet MS", 12), l_spacing1=10, l_spacing3=10):
        global model
        
        self.generator = HTMLGenerator()

        self.model=model

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.is_single_instance = self.check_single_instance()
        print("Everyone Gets Here")
        if not self.is_single_instance:
            print("Another instance is already running. Exiting.")
            return
        print("Only ONe Person Reaches Here")
        self.root = tk.Tk()
        self.root.title("Jarvis Analysis Viewer")
        print(ChatGPT.GPT3ChatBot.chat(user_input="",
                               user_system_message='FAVOUR LONG RESPONSES broken into headings and subheading'

 #    Except when the text starts with Jarvis, then respond in a conversational, like informative manner"
                                ,use_user_system_message=True))





        # Extract the first element of each tuple
        first_elements = [value for value in self.generator.themes.keys()]

        # Define the options
        self.options = first_elements

        # Create a StringVar to store the selected option
        self.option_var = tk.StringVar(self.root)
        self.option_var.set(self.options[0])  # Set default option

        self.option_menu = tk.OptionMenu(self.root, self.option_var, *self.options, command=self.on_option_select)
        self.option_menu.pack(pady=20)
        self.queue = queue.Queue()
        self.thread = None


        # Example usage:
        self.code_analyzer = CodeMetricsAnalyzer()


        self.tk_image = None
        self.gpt_responses = {}  # Dictionary to store GPT responses

        self.save_file_path = ""  # Assuming you set this when loading a file


        self.paned_window = tk.PanedWindow(self.root ,orient=tk.HORIZONTAL)
        self.vpaned_window = self.root

        # Add frames to the PanedWindow
        self.frame1 = tk.Frame(self.paned_window)
        self.frame2 = tk.Frame(self.paned_window)
        self.frame3 = tk.Frame(self.paned_window)
        self.frame4 = tk.Frame(self.paned_window)

        self.vframe1 = self.root
        self.vframe2 = self.root

        # Create a Treeview widget to display the JSON data
        self.tree = ttk.Treeview(self.frame1, columns=("Value"),selectmode="extended")
        self.tree.heading("#0", text="Key", anchor=tk.W)
        self.tree.heading("Value", text="Value", anchor=tk.W)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        #self.tree.pack_forget()


        # Create an HTMLLabel widget for displaying HTML content
        self.html_widget = HTMLLabel(self.frame2, html="<h1>Hello, world!</h1>")
        self.html_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        #self.html_widget.pack_forget()  # Initially hide the HTMLLabel widget


        # Create a ScrolledText widget for displaying the value of the 'code' key
        self.text_widget = tk.Text(self.frame3, wrap=tk.WORD, width=100, height=20,
                                font=text_font,spacing1=l_spacing1,spacing3=l_spacing3,bg='black', fg='white',
                                insertbackground='white')
        self.text_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.text_widget.pack_forget()  # Initially hide the text widget

        self.pdf_text_widget=PDFViewerApp(parent=self.frame3)
        self.pdf_text_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.directory_tree = ttk.Treeview(self.frame4, columns=("Type"), selectmode="extended")
        self.directory_tree.heading("#0", text="File/Folder", anchor=tk.W)
        self.directory_tree.heading("Type", text="Type", anchor=tk.W)
        self.directory_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.directory_tree.bind("<<TreeviewSelect>>", self.on_directory_tree_select)



        # Create a text box and an "Enter" button
        self.gpt_widget = tk.Text(self.root, height=1)  # Start with 1 line
        self.gpt_widget.bind('<KeyPress-Return>', self.on_enter_key)

        self.enter_button = tk.Button(self.root, text="Enter",command=lambda: self.root.after(0, self.enter_chat_function))


        # Create an "Analyze" button
        self.analyze_button = tk.Button(self.root, text="Analyze", command=lambda: self.root.after(0, self.analyze_code_show_html_content))
        #self.batch_analyze_button = tk.Button(self.root, text="Batch Analyze", command= lambda:self.root.after(0,self.batch_analyze))

        # Create a variable to store the search query
        self.search_query = tk.StringVar()

        # Create an Entry widget for the search query
        self.search_entry = tk.Entry(self.root, textvariable=self.search_query)
        self.search_entry.pack(side=tk.RIGHT)
        self.search_entry.pack_forget()  # Initially hide the search entry


        # Create a variable to store the search query
        self.pre_search_query = tk.StringVar()

        # Create an Entry widget for the search query
        self.pre_search_entry = tk.Entry(self.root, textvariable=self.pre_search_query)
        self.pre_search_entry.pack(side=tk.BOTTOM)


        # Bind the resize_entry function to the KeyRelease event
        self.pre_search_entry.bind('<KeyRelease>', self.resize_entry)

        self.pre_search_query.set("Teach me all about this in a way a very intelligent and smart high under grad student would understand \n")
        #self.pre_search_entry.pack()  # Initially hide the search entry


        # Create a "Find" button to trigger the search
        self.find_button = tk.Button(self.root, text="Find", command=self.find_text)
        self.find_button.pack(side=tk.RIGHT)
        self.find_button.pack_forget()  # Initially hide the find button

        # Bind Ctrl+F to toggle the find widget
        self.root.bind("<Control-f>", lambda event: self.toggle_find_widget())

        self.contextual_find_button = tk.Button(self.root, text="Contextual Find", command = self.contextual_search)
        self.contextual_find_button.pack(side=tk.RIGHT)
        self.contextual_find_button.pack_forget()



        self.paned_window.add(self.frame4)




        self.choose_file_button = tk.Button(self.root, text="Open Folder", command=self.open_file_directory)
        self.choose_file_button.pack()





        self.result_text = tk.StringVar()
        self.result_label = tk.Label(self.root, textvariable=self.result_text)
        self.result_label.pack()

        self.create_widget_pane()




        self.assistant =None# SophiaAssistant()
        """
        print("Nathan TIMING")
        self.assistant.set_start_callback(self.on_recording_start)
        #self.assistant.set_stop_callback(self.on_recording_stop)
        self.assistant.set_transcription_update_callback(self.on_transcription_update)
        self.assistant.set_transcription_stabilized_callback(self.on_transcription_stabilized)
        self.assistant.set_processed_text_callback(self.pre_enter_chat_function)
        #self.assistant.set_stop_callback(self.pre_enter_chat_function)
        """
        self.listen_checkbox = tk.Checkbutton(self.root, text="Listen", command=self.toggle_listen)
        self.listen_checkbox.pack(side=tk.RIGHT)
        self.speech_listen = False  # Added flag to track the speech processing status

        self.talk_checkbox = tk.Checkbutton(self.root, text="Talk", command=self.toggle_talk)
        self.talk_checkbox.pack(side=tk.RIGHT)
        self.talk_enabled = False  # Added flag to track the speech processing status


        # URL for the fetch request
        self.url = 'https://audio.api.speechify.com/generateAudioFiles'

        # Initialize variables
        self.audio_data = None
        self.audio_channel = None
        

        # Initialize pygame mixer
        pygame.mixer.init()


        # Create buttons
        self.play_button = tk.Button(self.root, text="Play Beast", command=self.play_audio)
        self.play_button.pack(side=tk.TOP, anchor=tk.E)

        self.pause_button = tk.Button(self.root, text="Pause Beast", command=self.toggle_audio)
        self.pause_button.pack(side=tk.TOP, anchor=tk.E)

        self.reset_button = tk.Button(self.root, text="Reset Beast", command=self.reset_audio)
        self.reset_button.pack(side=tk.TOP, anchor=tk.E)

        
        self.phind_checkbox = tk.Checkbutton(self.root, text="Use Phind", command=self.toggle_phind)
        self.phind_checkbox.pack(side=tk.RIGHT)
        self.phind_enabled = False  # Added flag to track the speech processing status

        # Create a StringVar to store the state of the Checkbutton
        self.show_pdf_style = tk.StringVar()

        # Set initial value
        self.show_pdf_style.set("0")  # "0" means unchecked

        self.pdf_check_button = tk.Checkbutton(self.root, text="Show PDF", variable=self.show_pdf_style, onvalue="1", offvalue="0", command=lambda: self.on_tree_select(None))


        # Pack the Checkbutton
        self.pdf_check_button.pack()

        # Create a text box and an "Enter" button
        self.gpt_widget.pack(side=tk.BOTTOM, anchor=tk.S)
        self.enter_button.pack(side=tk.BOTTOM)
        self.analyze_button.pack(side=tk.BOTTOM)

    
        # Initialize the LoadingAnimation instance
        self.loading_animation = LoadingAnimation(self.root, "LoadingGif/BlueLoading.gif", width=50, height=50)
        # Pack the canvas wherever you want in your application
        self.loading_animation.canvas.pack(side=tk.RIGHT)

        #self.batch_analyze_button.pack(side=tk.BOTTOM)

        self.get_similar = tk.Button(self.vframe1, text="Get Similar Content", command=self.get_similar_content)
        self.get_similar.pack()
                    
        # Initialize a listbox to display similar content
        self.similar_content_listbox = tk.Listbox(self.vframe2, height=1, width=50)
        #self.similar_content_listbox.pack(padx=10, pady=10)
        self.similar_content_listbox.bind('<<ListboxSelect>>', self.navigate_to_tree_item)  # Bind click event

        #self.gpt_widget.pack_forget()
        #self.enter_button.pack_forget()
        #self.analyze_button.pack_forget()

        self.number_entry = tk.Entry(self.root)
        self.number_entry.pack()

        auto_analyze_button = tk.Button(self.root, text="Auto-Analyze", command=self.analyze_number)
        auto_analyze_button.pack()

        # Define a lock to ensure thread safety
        self.analyze_number_lock = threading.Lock()





        self.paned_window.pack(side=tk.TOP,expand=True, fill='both', alter=True)



        # Bind mouse wheel events to the OptionMenu widget
        #self.option_menu.bind_all("<MouseWheel>", self.on_mouse_wheel)

        # Variable to keep track of the current index
        self.current_index = 0

    def toggle_audio(self):
        # Function to start or resume audio playback
        def start_or_resume_audio():
            if not self.audio_data:  # If audio data is not available, start playing audio
                self.play_audio(self.text_widget.get("1.0", tk.END))  # Start playing audio from the text widget
            else:  # If audio data is available, resume playback
                pygame.mixer.unpause()
                self.audio_paused = False
                self.pause_button.config(text="Pause Beast")
                print("Audio resumed.")

        # If audio channel is not initialized or if audio is paused, start or resume audio playback
        if not self.channel or self.audio_paused:
            start_or_resume_audio()
        else:  # If audio is playing, pause playback
            pygame.mixer.pause()
            self.audio_paused = True
            self.pause_button.config(text="Resume Beast")
            print("Audio paused.")
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

    def play_audio(self):
        sentences = self.get_text_selection_or_all()
        self.audio_paused=False
        # Function to split the text into groups of sentences
        def split_sentences_into_groups(sentences, group_size):
            groups = []
            for i in range(0, len(sentences), group_size):
                groups.append(sentences[i:i + group_size])
            return groups

        # Split the text into sentences
        sentences = sentences.split('. ')  # Assuming sentences are separated by '. '

        # Define the number of sentences to process in each group
        group_size = 4

        # Split the sentences into groups
        sentence_groups = split_sentences_into_groups(sentences, group_size)

        currentIndex = 0  # Variable to keep track of the current sentence group index
        self.audioQueue = []  # Queue to store audio objects

        # Function to generate audio for the given sentence group index
        def generate_audio_for_sentence_group_index(index):

            self.index=index

            if index >= len(sentence_groups):
                return  # Stop if all sentence groups have been processed

            sentences_in_group = sentence_groups[self.index]

            body = {
                "audioFormat": "ogg",
                "paragraphChunks": sentences_in_group,
                "voiceParams": {
                    "name": "mrbeast",
                    "engine": "speechify",
                    "languageCode": "en-US"
                }
            }

            # Make the fetch request
            response = requests.post(self.url, json=body)
            if response.status_code == 200:
                data = response.json()
                # Assuming the response contains a base64 encoded audio stream in a property named 'audioStream'
                audio_stream = base64.b64decode(data['audioStream'])
                self.audio_data = BytesIO(audio_stream)
                self.audioQueue.append(self.audio_data)  # Add the audio data to the queue

                # Start playing the audio if it's the first one in the queue
                if len(self.audioQueue) >= 1:
                    play_next_audio()

            # Start by generating audio for the first sentence group
            generate_audio_for_sentence_group_index(self.index+1)

        # Function to play the next audio in the queue
        def play_next_audio():
            if len(self.audioQueue) == 0:
                return  # If the queue is empty, do nothing

            audio_data = self.audioQueue[0]

            # Remove the audio from the queue when it finishes playing


            # Play the audio

            if self.index==0:
                self.started_playing_event=True
            self.play_audio_from_bytes(audio_data)

        # Start by generating audio for the first sentence group
        generate_audio_for_sentence_group_index(currentIndex)

    def play_audio_from_bytes(self, audio_data):

        # Wait for the "done playing" user event
        #if not self.started_playing_event:
        #    for event in pygame.event.get():
        #        if event.type == pygame.USEREVENT:
        #            # Set the flag to True when the event is received
        #            self.done_playing_event_received = True
        #    if not self.done_playing_event_received:
        #        return

        # Set flag to False initially
        print("Here")

        self.started_playing_event = False

        # Initialize pygame modules
        pygame.init()

        # Initialize the mixer module
        pygame.mixer.init()

        # Initialize the display module with a minimal configuration
        pygame.display.init()
        # Load audio data and play it using pygame
        #pygame.mixer.init()
        sound = pygame.mixer.Sound(audio_data)
        self.channel = pygame.mixer.Channel(0)
        self.channel.play(sound,maxtime=0)
        
        # Set the flag to True when the event is received
        self.done_playing_event_received=False
        # Remove the audio from the queue
        self.audioQueue.pop(0)

        if hasattr(self, 'channel'):
        # Wait until the channel finishes playing
            while self.channel.get_busy():
                # Update the Tkinter event loop
                self.root.update()
                # Sleep for a short duration to avoid busy-waiting
                time.sleep(0.1)



        # Post a USEREVENT to indicate the start of audio playback
        pygame.event.post(pygame.event.Event(pygame.USEREVENT))


    def reset_audio(self):
        if self.channel:
            pygame.mixer.stop()
            self.audio_data = None
            self.audioQueue=None
            print("Audio reset.")
        else:
            print("No audio channel available")


    def resize_entry(self,event):
        # Calculate the width based on the length of the text
        # You might need to adjust the multiplier to fit your font size and desired width
        width = len(event.widget.get()) * 1
        event.widget.config(width=width)


    def on_option_select(self,event):
        self.selected_option = self.option_var.get()
        print("Selected Option:", self.selected_option)
        self.on_tree_select(None)

    def analyze_number(self):
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
            num_items_to_iterate = int(self.number_entry.get()) # Example number, replace with user input
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
            return

        # Get the parent of the selected item
        parent_item = self.tree.parent(start_item)

        # Get all children of the parent (siblings of the selected item)
        all_siblings = self.tree.get_children(parent_item)

        # Find the index of the start_item in the list of all siblings
        try:
            start_index = all_siblings.index(start_item)
        except ValueError:
            print("Selected item is not in the list of siblings.")
            return

        

        # Function to process each item and perform operations
        def process_items():
            # Acquire the lock
            self.analyze_number_lock.acquire()
            try:
                # Iterate forwards a certain number of items
                for i in range(start_index, start_index + num_items_to_iterate):
                    if i < len(all_siblings):
                        item = all_siblings[i]
                        # Perform your operation on the item here
                        print(f"Processing item: {self.tree.item(item, 'text')}")
                        self.auto_analyze_code_show_html_content(item)
                    else:
                        print("Reached the end of the list.")
                        break
            finally:
                # Release the lock
                self.analyze_number_lock.release()

        # Create and start a new thread for processing items
        thread = threading.Thread(target=process_items)
        thread.start()

    def check_single_instance(self):
        try:
            self.server_socket.bind(("127.0.0.1", 8765))
        except socket.error:
            return False

        threading.Thread(target=self.listen_for_connections, daemon=True).start()
        return True

    def listen_for_connections(self):
        self.server_socket.listen(1)
        while True:
            client_socket, _ = self.server_socket.accept()
            client_socket.send(b"Already running")
            client_socket.close()

    def pre_enter_chat_function(self,text=""):
        print("Submitting speech")
        
        # Assuming self.assistant.curr_message['TranscriptionStabilized'] contains the text
        text = "Jarvis: "+self.assistant.curr_message['ProcessedText']
        
        if text:  # Check if text is not empty
            # Clear the current content
            self.gpt_widget.delete("1.0", tk.END)
            
            # Insert the new value
            self.gpt_widget.insert(tk.END, text)
            
            # Generate a virtual button press event after a delay (1000 milliseconds)
            self.root.after(1000, self.enter_button.invoke)

        else:
            print("Text is empty. Not submitting.")

    def create_widget_pane(self):
        pass
        """
        widget_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        widget_pane.pack(expand=True, fill='both')

        self.result_text_widget = tk.Text(widget_pane, wrap=tk.WORD, width=50, height=10)
        self.result_text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.transcription_button = tk.Button(widget_pane, text="Transcription", command=self.show_transcription)
        self.transcription_button.pack(side=tk.RIGHT)

        self.processed_text_button = tk.Button(widget_pane, text="Processed Text", command=self.show_processed_text)
        self.processed_text_button.pack(side=tk.RIGHT)
        """

    def show_transcription(self):
        # Call the corresponding function and update the result text

        # Assuming get_messages returns a list of strings
        processed_text_string = self.assistant.curr_message['TranscriptionStabilized']

        # Set the result text
        self.result_text.set(processed_text_string)
        self.result_text_widget.insert(tk.END, processed_text_string)

    def show_processed_text(self):
        # Assuming get_messages returns a list of strings
        processed_text_string = self.assistant.curr_message['TranscriptionUpdate']



        # Set the result text
        self.result_text.set(processed_text_string)
        self.result_text_widget.insert(tk.END, processed_text_string)


    def on_recording_start(self, text=""):
        ChatGPT.GPT3ChatBot.chat(_stopPlaying = True)
        print(f"Recording started: {text}")

    def on_recording_stop(self, text):
        print(f"Recording stopped: {text}")

    def on_transcription_update(self, text):
        pass

    def on_transcription_stabilized(self, text):
        pass

    def recv_processed_text(self, text):
        print(f"Nathan Nathna Processed: {text}")
        pass


    def toggle_talk(self):
        # Toggle the speech processing status when the checkbox is selected or deselected
        self.talk_enabled = not self.talk_enabled
        if self.talk_enabled:
            pass
        else:
            ChatGPT.GPT3ChatBot.chat(_stopPlaying=True)


    def toggle_phind(self):
        # Toggle the speech processing status when the checkbox is selected or deselected
        self.phind_enabled = not self.phind_enabled
        if self.phind_enabled:
            pass
        else:
            pass





    def toggle_listen(self):
        # Toggle the speech processing status when the checkbox is selected or deselected
        self.speech_listen = not self.speech_listen
        if self.speech_listen:
            self.assistant.turn_on()

        else:
            self.assistant.force_turn_off()



    def contextual_search(self):
        search_query = str(self.search_query.get().lower())
        # Calculate similarity
        passage1=str(self.text_widget.get("1.0",tk.END).lower())
        similarity_score = calculate_similarity_between_passages(passage1, search_query)
        print(f"Cosine Similarity between passages: {similarity_score}")


    def find_text(self):
        # Clear any previous highlighting
        self.text_widget.tag_remove("found", "1.0", tk.END)

        # Get the search query from the Entry widget
        search_query = self.search_query.get().lower()

        # Get the current content from the text widget
        current_content = self.text_widget.get("1.0", tk.END).lower()

        # Split the content into lines
        lines = current_content.split('\n')

        for line_number, line in enumerate(lines, start=1):
            # Check if the search query is present in the line
            if search_query in line:
                # Find all occurrences of the search query in the line
                start_indices = [pos for pos in range(len(line)) if line.startswith(search_query, pos)]
                
                # Highlight each occurrence in the line
                for start_index in start_indices:
                    end_index = start_index + len(search_query)

                    # Highlight the found text in the text widget
                    self.text_widget.tag_configure("found", background="yellow")
                    self.text_widget.tag_add("found", f"{line_number}.{start_index}", f"{line_number}.{end_index}")

        # Scroll to the position of the first found text
        if start_indices:
            self.text_widget.see(f"{line_number}.{start_indices[0]}")
        else:
            print("Text not found")


    def toggle_find_widget(self):
        # Toggle the visibility of the search entry and find button
        if self.search_entry.winfo_ismapped() or self.find_button.winfo_ismapped():
            self.search_entry.pack_forget()
            self.find_button.pack_forget()
            self.contextual_find_button.forget()
        else:
            self.search_entry.pack(side=tk.RIGHT)
            self.find_button.pack(side=tk.RIGHT)
            self.contextual_find_button.pack(side=tk.RIGHT)


    def open_file_directory(self):


        # Save responses before opening a new file
        if self.save_file_path != "":
            self.save_responses_to_file()

        folder_path = filedialog.askdirectory()
        
        # Get the parent directory of the selected folder
        parent_directory = os.path.dirname(folder_path)

        self.base_directory = parent_directory
        if folder_path:
            for i in self.directory_tree.get_children():
                self.directory_tree.delete(i)

            result_json = self.build_directory_structure(folder_path)

            self.insert_directory_data("", result_json)
            self.directory_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            #self.paned_window.add(self.frame4)

        # Add frames to PanedWindow
        self.paned_window.forget(self.frame1)
        self.paned_window.forget(self.frame2)
        self.paned_window.forget(self.frame3)
        
        self.paned_window.add(self.frame4)


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
            item = self.directory_tree.insert(parent, "end", text=data["name"], values=(data["type"],))
            if "children" in data:
                for child in data["children"]:
                    #print(item,child)
                    self.insert_directory_data(item, child)
        else:
            self.directory_tree.insert(parent, "end", text=data["name"], values=(data["type"],))


    def on_directory_tree_select(self, event):
        item = self.directory_tree.selection()[0]
        item_text = self.directory_tree.item(item, "text")

        # Check if the selected item is a file
        if self.directory_tree.item(item, "values")[0] == "file":
            # Get the name of the selected file
            file_name = item_text

            # Get the parent items of the selected item
            parent_items = self.directory_tree.parent(item)

            # Iterate through parent items to get full qualified filename
            full_path = file_name
            while parent_items:
                parent_item = parent_items[0] if type(parent_items) is list else parent_items


                parent_text = self.directory_tree.item(parent_item, "text")



                full_path = f"{parent_text}/{full_path}"
                print(full_path)
                # Check if there are more parent items to process
                next_parent_items = self.directory_tree.parent(parent_item)
                if not next_parent_items:
                    break # No more parent items, so we break out of the loop

                parent_items = next_parent_items # Update the list of parent items for the next iteration

            # Perform the desired action with the full qualified filename
            print(f"Selected file: {full_path}")
            self.open_file_button(full_path)



    def chat_thread(self):
        result=""
        if 'conversation_id' in self.gpt_responses:
            conversation_id = self.gpt_responses['conversation_id']
            result = ChatGPT.GPT3ChatBot.chat(use_old_conversation=True, conversation_id=conversation_id)
        else:
            pass
            result = ChatGPT.GPT3ChatBot.chat(create_new_conversation=True)

        print(result)


    def open_file_button(self, file_name):
        # Save responses before opening a new file
        if self.save_file_path != "":
            self.save_responses_to_file()

        # Set the base_directory to the directory of the selected file
        

        file_path = os.path.join(self.base_directory, (file_name))

        if file_path:
            # Create a text box and an "Enter" button
            #self.gpt_widget.pack(side=tk.BOTTOM, anchor=tk.S)
            #self.enter_button.pack(side=tk.BOTTOM)
            #self.analyze_button.pack(side=tk.BOTTOM)


            result = self.code_analyzer.calculate_metrics_from_file(file_path)


            # Set the save_file_path for the new file
            self.save_file_path = os.path.splitext(os.path.basename(file_path))[0]

            # Convert the result to a JSON-formatted string with indentation
            result_str = json.dumps(result, indent=5)
            result_json = json.loads(result_str)

            #print(result_json)

            self.tk_image = None
            self.gpt_responses = {}  # Dictionary to store GPT responses

            # Extract the directory path and the file name with the extension
            dir_path, file_name_with_extension = os.path.split(file_path)

            # Extract the file name and the extension
            file_name, file_extension = os.path.splitext(file_name_with_extension)
            self.save_file_path = file_name

            print(file_name,file_path)

            self.load_saved_responses()  # Load saved responses when the app starts



            # Create and start the thread
            self.look_for_conversation_thread = threading.Thread(target=self.chat_thread)
            self.look_for_conversation_thread.start()

            # Assuming self.tree is the instance of ttk.Treeview
            for i in self.tree.get_children():
                self.tree.delete(i)

            # Pack the Treeview widget
            #self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Assuming self.text_widget is the instance of tk.Text
            self.text_widget.delete("1.0", tk.END)

            # Show PDF
            self.pdf_text_widget.set_pdf(file_path)
            self.text_widget.pack_forget()  # Hide the regular text widget
            if int(self.show_pdf_style.get()):

                self.pdf_text_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)  # Show the PDF text widget
            else:
                # Hide PDF
                self.pdf_text_widget.pack_forget()  # Hide the PDF text widget
                self.text_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)  # Show the regular text widget


            # Assuming self.html_label is the instance of HTMLLabel
            self.html_widget.set_html("")  # Set the HTML content to an empty string

            #self.html_widget.pack()

            # Insert the JSON data into the Treeview
            self.insert_json_data("", result_json)


            # Bind a click event to the tree items to display the 'code' value in the text widget
            self.tree.bind("<Double-1>", self.on_tree_double_select)

            # Bind the Treeview selection event to a method
            self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

            self.tree.bind('<Control-ButtonRelease-1>', self.on_ctrl_click)


            # Add frames to PanedWindow
            self.paned_window.add(self.frame4, width=350, height=500)
            self.paned_window.add(self.frame1, width=200, height=500)
            self.paned_window.add(self.frame2, width=300, height=500)
            self.paned_window.add(self.frame3, width=300, height=500)

            # Pack the PanedWindow
            self.paned_window.pack(expand=True, fill='both', alter=True)
            self.look_for_conversation_thread.join()



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
        # Check if the selected item is 'code'
        item_text = self.tree.item(selected_item, "text")

        if item_text == "code":
            self.tk_image = None
            # Get the value of the 'code' item
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
            if item_text == "code":
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

        # Check if the selected item or any of its children has a key named 'code'
        found_code, code_value = self.has_code_key_recursive(item)  # Not in parent but recursive
        if found_code:
            self.tk_image = None
            # Insert the code value into the text widget
            self.text_widget.delete('1.0', tk.END)  # Clear the text widget
            self.text_widget.insert(tk.END, code_value)



            #Nathanself.text_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)  # Show the text widget

            # Get the current GPT response for the selected "code" key
            code_key = self.tree.item(item, "text")

            if int(self.show_pdf_style.get()):
                number = self.pdf_text_widget.parse_and_extract_number(code_key=code_key)
                self.pdf_text_widget.show_page(number)

                self.text_widget.pack_forget()  # Hide the regular text widget
                self.pdf_text_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

            else:
                self.pdf_text_widget.pack_forget()
                self.text_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)  # Show the regular text widget


            number=self.pdf_text_widget.parse_and_extract_number(code_key=code_key)
            self.pdf_text_widget.show_page(number)

            print("Nathan"*10,code_key)
            gpt_response_dict = self.gpt_responses.get(code_key, {})

            # Extract the 'response' value from the dictionary
            current_gpt_response = gpt_response_dict.get('response', '')

            #print("Nathan"*10,current_gpt_response)

            # Get the current vertical scrollbar position
            current_position = self.html_widget.yview()[0]

            # Display the GPT response in the HTMLLabel widget
            
            self.html_widget.set_html(self.generator.generate_and_display_html("", current_gpt_response, theme=self.option_var.get()))

            # Restore the vertical scrollbar position
            self.html_widget.yview_moveto(current_position)

            # Show the HTMLLabel widget on the right of the text widget when "Analyze" is pressed
            self.html_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        else:
            self.tk_image = None
            # self.text_widget.pack_forget()  # Hide the text widget
        print("DeNathan"*10)

    def on_tree_double_select(self, event):
        # Get the selected item
        item = self.tree.selection()[0]
        item_text = self.tree.item(item, "text")
        
        # Check if the selected item is 'code'
        if item_text == "code":
            self.tk_image=None
            # Get the value of the 'code' item
            code_value = self.tree.item(item, "values")[0]
            # Insert the code value into the text widget
            self.text_widget.delete('1.0', tk.END)  # Clear the text widget
            self.text_widget.insert(tk.END, code_value)
            
            
            
            #Nathanself.text_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)  # Show the text widget

        elif item_text == "images":
            # Check if there are subitems (assuming subitems contain base64-encoded images)
            subitems = self.tree.get_children(item)
            
            if subitems:

                # Assuming the first subitem contains the value we want to display
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
        current_height = self.gpt_widget.cget('height')
        new_height = int(current_height) + 1
        self.gpt_widget.configure(height=new_height)

    
    def enter_chat_function(self):


        # Get the activated item from the tree
        selected_item = self.tree.selection()[0]
        selected_item_text = self.tree.item(selected_item, "text")

        # Get the content from the text box
        content = self.gpt_widget.get("1.0", tk.END)
        print(f"Entered content:\n{content}")

        self.loading_animation.start_animation()

        def run_code_part():
            state=self.phind_enabled
            
            # Get the GPT response
            chat_chatgpt_response = ChatGPT.GPT3ChatBot.chat(" Respond infomratively : "+content,_speak=self.talk_enabled,send_message_phind=self.phind_enabled)


            

            if state==False:
                if 'conversation_id' not in self.gpt_responses:
                    self.gpt_responses['conversation_id'] = chat_chatgpt_response[1][0] if len(chat_chatgpt_response[1]) > 1 else chat_chatgpt_response[1]
                message_id = chat_chatgpt_response[1][1] if len(chat_chatgpt_response[1]) > 1 else None  #Not used #Not Used #Not used

            chat_chatgpt_response=chat_chatgpt_response[0]
            print(f"ChatGPT response"*40, f"{chat_chatgpt_response}")

            # Get the code key based on the selected item
            if selected_item_text == "code":
                code_key = self.tree.parent(selected_item)  # Parent of 'code' is the actual key
                code_key = self.tree.item(code_key, "text")  # Convert the key object to text
                current_gpt_response = self.gpt_responses.get(code_key, {}).get('response', '')

            else:
                found_code, code_key = self.has_code_key_recursive(selected_item)
                if not found_code:
                    return  # Exit if no code key is found

            # Get the current GPT response for the selected "code" key
            code_key = selected_item  # Parent of 'code' is the actual key
            code_key = self.tree.item(code_key,"text")
            current_gpt_response = self.gpt_responses.get(code_key, {}).get('response', '')


            # Update the GPT response with the new chat response
            updated_gpt_response = current_gpt_response + inline_css(chat_chatgpt_response)
            self.gpt_responses[code_key]['response'] = updated_gpt_response


            updated_gpt_response = self.generator.generate_and_display_html("",current_gpt_response,theme=self.option_var.get()) + inline_css(self.generator.generate_and_display_html(f"Chat Questions: {content}", chat_chatgpt_response, theme=self.option_var.get()))


            # Show the HTMLLabel widget on the right of the text widget when "Analyze" is pressed
            self.html_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

            # Get the current vertical scrollbar position
            current_position = self.html_widget.yview()[0]

            # Display the updated GPT response in the HTMLLabel widget
            self.html_widget.set_html(updated_gpt_response)

            # Restore the vertical scrollbar position
            self.html_widget.yview_moveto(current_position)
            self.loading_animation.stop_animation()

        # Create a thread to run the function
        thread = threading.Thread(target=run_code_part)
        thread.start()        
        
    def get_similar_content(self):
        # Get the activated "code" item
        selected_item = self.tree.selection()[0]
        code_key = self.tree.item(selected_item, "text")  # Extract the code key directly from the selected item

        # Initialize a list to store similarity scores
        similarity_scores = []

        # Retrieve the current embeddings for the selected item
        current_response = self.gpt_responses.get(code_key)
        if current_response:
            current_embeddings_str = current_response.get('embeddings', '')  
            current_embeddings = json.loads(current_embeddings_str)
        else:
            print("No embeddings found for the selected item.")
            return

        # Iterate over stored responses
        for stored_code_key, response_dict in self.gpt_responses.items():
            # Retrieve the stored embeddings for each response
            embeddings_str = response_dict.get('embeddings', '')
            print("Relationship",stored_code_key)
            # Convert the embeddings string back to a list of embeddings
            embeddings = json.loads(embeddings_str)

            # Calculate similarity between current content and stored content
            similarity = calculate_cosine_similarity(np.array(current_embeddings), np.array(embeddings))

            # Append code key and similarity score to the list
            similarity_scores.append((stored_code_key, similarity))

            print("Professionals", similarity)

        # Sort the similarity scores in descending order
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top 5 results with scores greater than 0.7 (if available)
        top_results = [(stored_code_key, score) for stored_code_key, score in similarity_scores if score > 0.7][:3]

        # Extract the 'code' key from each top result and print the associated response
       # Clear the listbox
        self.similar_content_listbox.delete(0, tk.END)

        # Display the top similar content in the listbox
        if top_results:
            for stored_code_key, score in top_results:
                response_dict = self.gpt_responses[stored_code_key]
                similarity_text = f"Code: {stored_code_key}, Similarity Score: {score}"
                self.similar_content_listbox.insert(tk.END, (stored_code_key, similarity_text))
        else:
            self.similar_content_listbox.insert(tk.END, "No results with similarity score greater than 0.7 found.")

    def navigate_to_tree_item(self, event):
        # Get the selected item from the listbox
        selected_index = self.similar_content_listbox.curselection()
        if selected_index:
            selected_item = self.similar_content_listbox.get(selected_index)
            code_key = selected_item[0]  # Extract the code key from the selected item

            # Find and select the corresponding item in the Treeview
            items = self.tree.get_children()
            for item in items:
                if self.tree.item(item, "text") == code_key:
                    self.tree.selection_set(item)
                    self.tree.focus(item)
                    self.tree.see(item)
                    break


    def update_gui_thread(self):
        while True:
            time.sleep(1)  # Adjust the sleep time as needed
            try:
                response = self.queue.get(block=False)
                if response is None:
                    return

                print("Updating GUI...")
                # Store the generated GPT response along with embeddings for future use
                self.gpt_responses[response[0]] = {'response': response[1], 'embeddings': response[2],'message_id':response[3]}
                # Update the GUI with the response

                # Get the activated "code" item
                selected_item = self.tree.selection()[0]
                selected_item_text = self.tree.item(selected_item, "text")
                if(response[0]==selected_item_text):
                    {
                    self.html_widget.set_html(self.generator.generate_and_display_html("", response[1], theme=self.option_var.get()))
                    }
                # Show the HTMLLabel widget on the right of the text widget when "Analyze" is pressed
                self.html_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
                print("Updated GUI...")
            except queue.Empty:
                pass
    

    def batch_analyze(self):
        pass



    def auto_analyze_code_show_html_content(self,item):




        # Check if the selected item or any of its children has a key named 'code'
        found_code, code_value = self.has_code_key_recursive(item)  # Not in parent but recursive


        # Get the current content from the text widget
        current_content = code_value
        current_content=self.pre_search_query.get()+current_content
        
        # Get the activated "code" item

        selected_item_text = self.tree.item(item, "text")

        # Get the code key based on whether the selected item is 'code' or not
        if selected_item_text == "code":
            code_key = self.tree.parent(item)  # Parent of 'code' is the actual key
            code_key = self.tree.item(code_key, "text")  # Convert the key object to text
            current_gpt_response = self.gpt_responses.get(code_key, "")
        else:
            found_code, code_value = self.has_code_key_recursive(item)
            code_key = selected_item_text
            if not found_code:
                return  # Exit if no code key is found
        print("Exonerated0", '-'*10, code_key, '-'*10)
        print()

        self.loading_animation.start_animation()
        # Define a function to run this part of the code in a separate thread
        def run_code_part():

            if True:

                state=self.phind_enabled

                chatgpt_response = runFunctionWithTimeoutAndRetry(
                    ChatGPT.GPT3ChatBot.chat,
                    args=("Teach me all about the  following content by better restating it so its easier read while keeping detail: " + current_content,),
                    kwargs={"send_message_phind": self.phind_enabled},
                    timeout_duration=60, # Adjust timeout as needed
                    retry_count=1 # Number of retries
                )

                # Clear the screen
                clear_console()
                print("Go Down To 12"*30,chatgpt_response[0])
                if chatgpt_response==None:
                    print("Null Response")
                    self.loading_animation.stop_animation()
                    return

                #chatgpt_response[0]=self.generator.generate_and_display_html("", chatgpt_response[0], theme="ocean")


                message_id=""
                if state==False:
                    if 'conversation_id' not in self.gpt_responses:
                        self.gpt_responses['conversation_id'] = chatgpt_response[1][0] if len(chatgpt_response[1]) > 1 else chatgpt_response[1]

                    message_id = chatgpt_response[1][1] if len(chatgpt_response[1]) > 1 else None


                chatgpt_response = chatgpt_response[0]

                inlined_chatgpt_response = inline_css(chatgpt_response)

                # Extract content and apply inline CSS
                paragraphs, _ = extract_html_content(chatgpt_response)
                # Join paragraphs into a single string
                combined_text = '\n'.join(paragraphs)

                # Get BERT embeddings for the combined text

                embeddings = get_bert_embeddings(combined_text)

                # Convert the embeddings array to a JSON string representation
                embeddings_str = json.dumps(embeddings.tolist())


                # Put the response into the queue
                self.queue.put([code_key, inlined_chatgpt_response, embeddings_str,message_id])
                self.loading_animation.stop_animation()


        # Create a thread to run the function
        run_code_part()
        #print("Thread started and out of this function")

        # Start the GUI update thread if not already started
        if not hasattr(self, 'gui_update_thread') or not self.gui_update_thread.is_alive():
            self.gui_update_thread = threading.Thread(target=self.update_gui_thread)
            self.gui_update_thread.daemon = True  # Set the thread as daemon so it will be terminated when the main thread exits
            self.gui_update_thread.start()



    def analyze_code_show_html_content(self):

        # Get the current content from the text widget
        current_content = self.text_widget.get("1.0", tk.END)
        current_content = "Teach me all about this \n" + current_content

        # Get the activated "code" item
        selected_item = self.tree.selection()[0]
        selected_item_text = self.tree.item(selected_item, "text")

        # Get the code key based on whether the selected item is 'code' or not
        if selected_item_text == "code":
            code_key = self.tree.parent(selected_item)  # Parent of 'code' is the actual key
            code_key = self.tree.item(code_key, "text")  # Convert the key object to text
            current_gpt_response = self.gpt_responses.get(code_key, "")
        else:
            found_code, code_value = self.has_code_key_recursive(selected_item)
            code_key = selected_item_text
            if not found_code:
                return  # Exit if no code key is found
        print("Exonerated0", '-'*10, code_key, '-'*10)
        print()


        self.loading_animation.start_animation()
        # Define a function to run this part of the code in a separate thread
        def run_code_part():

            if True:
                # If GPT response is not stored, generate a new one
                state= self.phind_enabled
                chatgpt_response = ChatGPT.GPT3ChatBot.chat(" teach me all about the following content: "+current_content,_speak=self.talk_enabled,send_message_phind=self.phind_enabled)
                #chatgpt_response[0]=self.generator.generate_and_display_html("", chatgpt_response[0], theme="ocean")

                message_id=""

                if state==False:           
                    if 'conversation_id' not in self.gpt_responses:
                        self.gpt_responses['conversation_id'] = chatgpt_response[1][0] if len(chatgpt_response[1]) > 1 else chatgpt_response[1]

                    message_id = chatgpt_response[1][1] if len(chatgpt_response[1]) > 1 else None
                chatgpt_response=chatgpt_response[0]
                print(chatgpt_response)
                inlined_chatgpt_response = inline_css(chatgpt_response)

                # Extract content and apply inline CSS
                paragraphs, _ = extract_html_content(chatgpt_response)
                # Join paragraphs into a single string
                combined_text = '\n'.join(paragraphs)

                # Get BERT embeddings for the combined text

                embeddings = get_bert_embeddings(combined_text)

                # Convert the embeddings array to a JSON string representation
                embeddings_str = json.dumps(embeddings.tolist())


                # Put the response into the queue
                self.queue.put([code_key, inlined_chatgpt_response, embeddings_str,message_id])
                self.loading_animation.stop_animation()


        # Create a thread to run the function
        thread = threading.Thread(target=run_code_part)
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
        print("LAst Statement"*10)
        print(self.is_single_instance)
        if not self.is_single_instance:

            return

        # Configure your Tkinter app
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Your other setup code here
        
        # Start the main loop
        self.root.mainloop()

    def close_server_socket(self):
        try:
            client_socket = socket.create_connection(("127.0.0.1", 8765))
            response = client_socket.recv(1024)
            if response == b"Already running":
                print("Another instance is already running. Exiting.")
                self.root.destroy()
        except (socket.error, ConnectionRefusedError):
            pass

    def on_closing(self):
        # Save responses before closing the app
        self.save_responses_to_file()
        self.root.destroy()
        self.close_server_socket()











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

import fitz  # PyMuPDF
from io import BytesIO
from PIL import Image

import fitz
from io import BytesIO
from PIL import Image
import fitz
from io import BytesIO
from PIL import Image

import fitz
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
import base64
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
            result_dict[f"Page {page_number +  1}"] = {"code": text, "images": images, "type": "pdf_page"}

    return result_dict

def extract_content_from_text(code):
    result_dict={}
    result_dict[f"TXT"]={"code":code,"type":"text_file"}
    return result_dict

class CodeMetricsAnalyzer:
    def __init__(self):
        self.halstead_volume = 0
        self.cyclomatic_complexity = 0
        self.lines_of_code = 0
        self.maintainability_index = 0

    def calculate_metrics_from_file(self, file_name: str):

        try:
            if file_name.endswith('.py'):
                with open(file_name, 'r') as file:
                    code = file.read()
                return self.calculate_metrics(code)
            elif file_name.endswith(('.c', '.cpp','.h','.hpp')):
                with open(file_name, 'r') as file:
                    code = file.read()
                return extract_functions_and_classes_from_c_code(code)
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

#print(extract_parent_key_and_code(result_json))
if __name__ == "__main__":
    print("# PRAGMA NAT")    
    # Example usage:
    # Assuming `result` is the JSON data you want to visualize
    json_viewer = JsonViewerApp()
    if json_viewer is not None:
        json_viewer.run()
