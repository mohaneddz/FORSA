import gradio as gr
import requests
import json
import os

API_URL = "http://127.0.0.1:8000/api/chat/"

def process_request(input_type, text_input, file_input):
    payload = {}
    
    if input_type == "Text Input":
        if not text_input:
            return {"error": "Please enter text."}, None
        try:

            # Let's try to parse it as JSON first.
            try:
                payload = json.loads(text_input)
            except json.JSONDecodeError:
                # If not JSON, wrap it as a single question
                payload = {
                    "equipe": "GhostTruth",
                    "question": {
                        "Manual_Input": {
                            "Q1": text_input
                        }
                    }
                }
        except Exception as e:
            return {"error": f"Error processing text input: {str(e)}"}, None

    elif input_type == "File Input":
        if not file_input:
            return {"error": "Please upload a file."}, None
        try:
            # file_input is a temp file path
            with open(file_input.name, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except Exception as e:
            return {"error": f"Error processing file: {str(e)}"}, None
    
    # Send to API
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result_json = response.json()
        
        # Save to a temporary file for download
        output_filename = "response.json"
        with open(output_filename, "w", encoding='utf-8') as f:
            json.dump(result_json, f, indent=2, ensure_ascii=False)
            
        return result_json, output_filename
    except requests.exceptions.RequestException as e:
        return {"error": f"API Error: {str(e)}", "detail": f"Make sure the FastAPI server is running at {API_URL}"}, None
    except Exception as e:
        return {"error": str(e)}, None

with gr.Blocks(title="Heavy RAG Interface") as demo:
    gr.Markdown("# Algerie Telecom RAG System")
    gr.Markdown("Select input method to ask questions.")
    
    with gr.Row():
        input_type = gr.Radio(["Text Input", "File Input"], label="Input Method", value="Text Input")
    
    with gr.Row(visible=True) as text_row:
        text_input = gr.Textbox(lines=10, label="Paste JSON or Question", placeholder='{"equipe": "...", "question": {...}} or just "What is Idoom?"')
    
    with gr.Row(visible=False) as file_row:
        file_input = gr.File(label="Upload JSON File", file_types=[".json"])

    # Toggle visibility based on radio selection
    def toggle_input(choice):
        if choice == "Text Input":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    input_type.change(fn=toggle_input, inputs=input_type, outputs=[text_row, file_row])
    
    submit_btn = gr.Button("Submit", variant="primary")
    
    with gr.Row():
        result_output = gr.JSON(label="Response JSON")
        download_btn = gr.File(label="Download Answer")

    submit_btn.click(
        fn=process_request,
        inputs=[input_type, text_input, file_input],
        outputs=[result_output, download_btn]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860)
