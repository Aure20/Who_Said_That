# Import module  
from tkinter import *
import tkinter as tk
from quotes import Quotes
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

  
class MyApp:
    def __init__(self, master):

        self.quotes = Quotes('embeddings-300.csv', 'base_nn.pkl', 'https://tfhub.dev/google/universal-sentence-encoder/4')
        self.master = master
        master.title("Who Said That?")
        master.geometry("1071x604")

        self.bg = PhotoImage(file="background.png")
        self.canvas = Canvas(master)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, image=self.bg, anchor="nw") 

        self.text_input = Entry(master, font = ('Candara 20 bold'))
        self.text_input.bind('<Return>', self.on_button_click)
        entry_width = 250 
        y_entry = 200
        self.canvas.create_window(1071//2, y_entry, window=self.text_input,width=entry_width)

        self.search_button = Button(master, text="Search quote", command=self.on_button_click,font = ('Helvetica 12 bold'))
        self.canvas.create_window(1071//2+entry_width//2+70, y_entry, window=self.search_button)

        # Create the dropdown menu
        mode_options = ['base', 'exact','weighted', 'corrected']
        self.mode_var = tk.StringVar(root)
        self.mode_var.set(mode_options[0])
        self.mode_var.trace_add('write', self.on_change_of_mode) #Everytime it changes we need to change the index as well

        self.result_index = tk.IntVar(root)
        self.result_index.set(1)

        self.next_button = Button(master, text="→", command=self.next_batch,font = ('Helvetica 12 bold'))
        self.canvas.create_window(800, 375, window=self.next_button)
        self.prev_button = Button(master, text="←", command=self.prev_batch,font = ('Helvetica 12 bold'))
        self.canvas.create_window(271, 375, window=self.prev_button)

        self.dropdown = OptionMenu(root, self.mode_var, *mode_options)
        self.dropdown.config(font=('Helvetica 12 bold'))
        self.canvas.create_window(350,200, window=self.dropdown)

        self.displayed_quotes = []

    @staticmethod   
    def process_quotes(results):
        authors = [results[i].split(',')[0].strip('(\'').rstrip('\'').strip('\"').rstrip('\"') for i in range(len(results))]
        quotes = [','.join(results[i].split(',')[1:]).strip().rstrip(')\"').strip('\'').rstrip('\"').strip('\"') for i in range(len(results))]
        out_dict = {quote : author  for author,quote  in zip(authors,quotes)}
        return out_dict
    
    def next_batch(self):
        i = self.result_index.get()
        mode = self.mode_var.get()
        if i > 3 and mode != 'exact': #Avoid going over 20 elements
            self.result_index.set(4)
        else:
            self.result_index.set(i+1)
        self.on_button_click()
    
    def prev_batch(self):
        i = self.result_index.get()
        if i < 2: #Avoid going below 0
            self.result_index.set(1)
        else:
            self.result_index.set(i-1)
        self.on_button_click()
    
    def on_change_of_mode(self,*args):
        self.result_index.set(1)
        for quote in self.displayed_quotes:
            self.canvas.delete(quote)
        self.displayed_quotes = []
            
    def on_button_click(self, *args):
        entered_text = self.text_input.get()
        mode = self.mode_var.get()
        i = self.result_index.get()

        #Get 20 nearest quotes
        results = self.quotes.get_nearest(entered_text,mode=mode)

        #Check that 
        if i*5 > len(results) and mode == 'exact':
            i = len(results)//5+1
            self.result_index.set(i)

        #Return a dictionary from the ndarray and apply some processing 
        results = self.process_quotes(results.flatten()[(i-1)*5:i*5])

        #Delete previous quotes from the screen
        for quote in self.displayed_quotes:
            self.canvas.delete(quote)
        self.displayed_quotes = []

        # Add new results to the listbox
        start_spot = 225
        offset = 0 
        for j,(quote,author) in enumerate(results.items()):
            #Limit length of quote 
            if len(quote) > 65:
                if len(quote) > 132:
                    quote = quote[:128] + ' ...'
                quote = quote[:65] + '-\n' + quote[65:132] 
            
            author_text = self.canvas.create_text(1071//2-220,start_spot + j*40 + offset, text=author+':',font = ('Helvetica 13 bold'),anchor='nw', fill='red')
            self.displayed_quotes.append(author_text)
           

            quote_text = self.canvas.create_text(1071//2-220,start_spot + j*40 + offset+20,text=quote,font = ('Helvetica 11'),anchor='nw')
            self.displayed_quotes.append(quote_text)
            
            #If quote is on 2 lines need to make necessary space
            if len(quote) > 66:
                offset += 20
        
        #Add index of page
        page_text = self.canvas.create_text(1071//2,start_spot + j*40 + offset+40,text=f'Page:{i}',font = ('Helvetica 11 bold'))
        self.displayed_quotes.append(page_text)
    
if __name__ == "__main__":
    root = Tk()
    app = MyApp(root)
    root.mainloop()

    