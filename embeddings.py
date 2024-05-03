# imports
from openai import OpenAI # for calling the OpenAI API
import pandas as pd  # for DataFrames to store article sections and embeddings
import tkinter as tk
from tkinter import messagebox


def show_message_box(message):
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Show message box
    messagebox.showinfo("Message", message)

client = OpenAI(api_key="OPENAI_API_KEY")

# Step 1: Read unstructured text data from a text file
DATA_FILE_PATH = r"PATH_TO_TEXT_FILE"  # Path to your text file

def read_text_data(file_path: str) -> str:
    """Read unstructured text data from a text file."""
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text

# Read unstructured text data
unstructured_text_data = read_text_data(DATA_FILE_PATH)

# Step 2: Process text data into sections or chunks
# Here, you need to define a function to split your text data into sections or chunks
# This function will depend on the specific structure of your text data

def split_text_into_sections(text: str) -> list[str]:
    """
    Split text data into sections or chunks.

    This function should implement your logic to split the text data
    into meaningful sections or chunks.
    """
    # Implement your logic here to split the text data
    # For demonstration, let's split the text into chunks of 1000 characters each
    chunk_size = 1000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    return chunks


# Split text data into sections or chunks
text_sections = split_text_into_sections(unstructured_text_data)

# Step 3: Compute embeddings for each section or chunk
# You can use the same code as provided in the example to compute embeddings

# Define embedding model and batch size
EMBEDDING_MODEL = "text-embedding-ada-002"
BATCH_SIZE = 1000

# Compute embeddings
embeddings = []
for batch_start in range(0, len(text_sections)):
    batch = text_sections[batch_start]

    response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
    for i, be in enumerate(response.data):
        assert i == be.index
    batch_embeddings = [e.embedding for e in response.data]
    embeddings.extend(batch_embeddings)
    
# Step 4: Store document chunks and embeddings
# Here, you can store the processed data along with the embeddings in a CSV file
# Modify the code provided in the example to save your data

# Save processed data and embeddings to a CSV file
SAVE_PATH = r"PATH_TO_EMBEDDINGS_FILE"

# Create a DataFrame with text sections and embeddings
df = pd.DataFrame({"text": text_sections, "embedding": embeddings})

# Save DataFrame to a CSV file
df.to_csv(SAVE_PATH, index=False)

