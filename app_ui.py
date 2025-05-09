import gradio as gr
from main_agent import process_query

def ask_agent(question):
    try:
        return process_query(question)
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# local agent")
    gr.Markdown("good questions leads to more insights than great answers")
    
    with gr.Row():
        inp = gr.Textbox(label="ask and ye shall receive:", lines=2, placeholder="Type your question here...")
        submit_btn = gr.Button("Submit", variant="primary")
    
    out = gr.Textbox(label="Agent Reply", lines=10)
    
    # Connect both the submit button and Enter key to the same function
    submit_btn.click(ask_agent, inputs=inp, outputs=out)
    inp.submit(ask_agent, inputs=inp, outputs=out)

demo.launch()