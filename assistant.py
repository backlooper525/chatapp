import ast  # for converting embeddings saved as strings back to arrays
from openai import OpenAI # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
from scipy.spatial.distance import euclidean
import os, sys

import tkinter as tk
from tkinter.scrolledtext import ScrolledText

sys.stdout.reconfigure(encoding='utf-8')


# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = ''

client = OpenAI(api_key="YOUR_API_KEY")


# download pre-chunked text and pre-computed embeddings
embeddings_path = r"PATH_TO_EMBEDDINGS_FILE"


df = pd.read_csv(embeddings_path)

# convert embeddings from CSV str type back to list type
df['embedding'] = df['embedding'].apply(ast.literal_eval)
# the dataframe has two columns: "text" and "embedding"


# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    #relatedness_fn=lambda x, y: euclidean(x, y),
    top_n: int = 20
) -> tuple[list[str], list[float]]:
      
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding

    strings_and_relatednesses = [
        (relatedness_fn(query_embedding, row["embedding"]), row["text"])
        for i, row in df.iterrows()
    ]
    
    strings_and_relatednesses.sort(key=lambda x: x[0], reverse=True)
    
    
    
    df5 = pd.DataFrame(strings_and_relatednesses, columns=['Relatedness', 'String'])
    #print(df5)
    df5.to_csv("df5.txt", sep='\t', index=False, header=True)
    #sys.exit()
    
    
  
    relatednesses, strings = zip(*strings_and_relatednesses)
    return relatednesses[:top_n], strings[:top_n]


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    model = selected_option.get()
    
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    relatednesses, strings = strings_ranked_by_relatedness(query, df)

    introduction = 'Use the below text to answer the subsequent question. And use ONLY that text'
    #introduction = 'Use the below text to answer the subsequent question. If the answer cannot be found in the text, write "I could not find an answer."'
    
    
    question = f"\n\nQuestion: {query}\n\n"
    message = introduction
    
    for string in strings:
        next_article = f'{string}\n\n'
        if (num_tokens(message + next_article + question, model=model) > token_budget):
            break
        else:
            message += next_article
    return message + question


def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = True,
) -> str:
    
    model = selected_option.get()
    
    if checkbox_var.get() == 0: #2ra kasuta faili
        message = query
    else:
        """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
        message = query_message(query, df, model=model, token_budget=token_budget)
    
    if print_message:
        print(message.encode('utf-8'))
    

    messages = [
        {"role": "system", "content": "You answer questions"},
        {"role": "user", "content": message},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    
    return response_message


def start_program(kysimus):
    global GPT_MODEL
    GPT_MODEL = selected_option.get()
    
    response = ask(kysimus)
    output_text.insert(tk.END, f"{kysimus}\n{response}\n\n", "regular")
    symbol_entry.delete(0, tk.END)
     

def on_close(window):
    save_window_size(window, 'chatapp_settings.txt')
    window.destroy()

def load_window_size(root, filename):
    default_size = "800x600"
    if os.path.isfile(filename):
        with open(filename, 'r') as file:
            geometry = file.read()
        root.geometry(geometry)
    else:
        root.geometry(default_size)

def save_window_size(root, filename):
    geometry = root.geometry()
    with open(filename, 'w') as file:
        file.write(geometry)



fsize=10
# Peaaken
window = tk.Tk()
window.title("ChatApp")
load_window_size(window, 'chatapp_settings.txt')
# Tume v2rvilahendus
window.configure(bg='gray10')

button_frame = tk.Frame(window, bg='gray10')
button_frame.pack(padx=10, pady=10)

symbol_label = tk.Label(button_frame, text="Question", bg='gray10', fg='white')
symbol_label.grid(row=0, column=0, padx=(0, 5))

# Sisestusv2li
symbol_entry = tk.Entry(button_frame, bg='gray30', fg='white', width=100)
symbol_entry.grid(row=0, column=1, padx=(0, 5))

start_button = tk.Button(button_frame, text="Ask", command=lambda: start_program(symbol_entry.get()), bg='gray10', fg='white')
start_button.grid(row=0, column=2)

# Enter klahv k2ivitab start_program
symbol_entry.bind('<Return>', lambda event: start_program(symbol_entry.get()))

# Output aken
output_text = ScrolledText(window, bg='gray20', fg='white')
output_text.pack(fill=tk.BOTH, expand=True)  # Initial configuration
output_text.tag_configure("regular", font=("TkDefaultFont", fsize, ""))   #saab kasutada funktsioonides


# Dropdown menu
options = ["gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-4-turbo"]  # Example options
selected_option = tk.StringVar()
selected_option.set(options[0])  # Set default option
dropdown_menu = tk.OptionMenu(button_frame, selected_option, *options)
dropdown_menu.config(bg='gray10', fg='white')  # Adjust background color and text color
dropdown_menu.grid(row=0, column=3, padx=(0, 5))


# Checkbox
checkbox_var = tk.BooleanVar()
checkbox = tk.Checkbutton(button_frame, text="Use file", variable=checkbox_var, bg='gray10', fg='red')
checkbox.grid(row=0, column=4)



window.protocol("WM_DELETE_WINDOW", lambda: on_close(window))

window.mainloop()


