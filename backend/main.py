from flask import Flask, request, jsonify
import openai
import anthropic
import logging
import re
from dotenv import load_dotenv
import json
import google.generativeai as genai 
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
load_dotenv()

claude_api_key = os.getenv("CLAUDE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

logging.basicConfig(level=logging.INFO)

def main_llm(prompt):
    
    list_code = [
    "code", "snippet", "script", "function", "class", "method",
    "example", "sample", "program", "algorithm", "solve", "loop", "recursion", "html", "css", "javascript", "python", "java", "run"]
    list_touch = ["feeling", "sad", "mad"]
    list_image = ["image", "generate"]
    list_logic = ["m", "km", "crosses"]
    for i in list_code:
        if i in prompt.lower():
            return "Coding related task"
    for i in list_touch:
        if i in prompt.lower():
            return "Human touch related task"
    for i in list_image:
        if i in prompt.lower():
            return "Image processing related question"
    for i in list_logic:
        if i in prompt.lower():
            return "Logical thinking related question"
        
    return "Human touch related task"

def claude_llm(messages):
    client = anthropic.Anthropic(api_key=claude_api_key)

    system_message = f"""
    You are a highly intelligent AI assistant named Claude. Your main goal is to assist with code generation and general inquiries.
    Default to Python for code examples unless another language is requested.
    
    For coding-related questions, format your answer clearly with:
    - A brief introduction
    - The code in proper code blocks with language specification (```python, ```javascript, etc.)
    - A conclusion with explanation

    For general questions, provide complete and helpful responses.
    """

    try:
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=800,
            temperature=0.7,
            system=system_message,
            messages=[
                {"role": "user", "content": messages}
            ]
        )

        reply = response.content[0].text.strip()
        
        return {
            "response": reply,
            "model": "claude"
        }

    except Exception as e:
        logging.error(f"Claude API error: {str(e)}")
        return {"Error": f"An unexpected error occurred: {e}"}

def openai_llm(messages):
    list_image = ["image", "generate"]
    for i in list_image:
        if i in messages.lower():
            response = gemini_llm(messages)
            return response
    client = openai.OpenAI(api_key=openai_api_key)
    
    system_message = f"""
    You are an AI assistant who responds like a friendly companion. Your goal is to keep the conversation engaging and natural.
    - If the user is looking for an answer to a task, provide a structured and complete response.
    - If the user wants to chat, respond in a warm, interactive way, extending the conversation when possible.
    - Maintain a smooth, natural flow—never let the interaction feel dry.
    - Adapt to the user's tone and mood, ensuring an enjoyable experience.

    Given user input:
    {messages}

    Respond in one of these formats and strictly as raw JSON:
    - If there's a task to complete:
      {{
          "task": "short description of the task",
          "response": "solution or completion of the task"
      }}
    - If it's a casual interaction output:
          "engaging and natural response"
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": system_message}, {"role": "user", "content": messages}],
            max_tokens=500,
            temperature=0.7
        )
        content = response.choices[0].message.content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return json.loads(json.dumps({"output": content}))

    except Exception as e:
        return json.loads(json.dumps({"Error": f"An error occurred: {e}"}))

def gemini_llm(messages):
    # Dall-E Image-Handle.
    try:
        genai.configure(api_key=gemini_api_key)
        
        client = openai.OpenAI(api_key=openai_api_key)
        
        model = genai.GenerativeModel('gemini-1.5-pro')
        gemini_response = model.generate_content(f"""
        Take the following prompt. The prompt may contain a command on generating image. You have to expand it into a rich, detailed image prompt suitable for DALL·E.
        Include descriptions of scenery, style, atmosphere, lighting, and artistic tone.

        Prompt: {messages}
        """)
        extended_prompt = gemini_response.text.strip()

        try:
            dalle_response = client.images.generate(
                model="dall-e-3",
                prompt=extended_prompt,
                n=1,
                size="1024x1024"
            )
            image_url = dalle_response.data[0].url

            return {
                "Image URL": image_url,
            }
        except Exception as image_error:
            logging.error(f"DALL-E image generation error: {str(image_error)}")
            return {
                "Error": f"Failed to generate image: {str(image_error)}"
            }

    except Exception as e:
        logging.error(f"Gemini error: {str(e)}")
        return {"Error": str(e), "Response": "I encountered an error processing your image request."}

def clean_json_text(text):
    return re.sub(r"^```json|```$", "", text.strip(), flags=re.MULTILINE).strip()

def gemini_llama_llm(prompt):
    """Handle logical thinking using Gemini"""
    try:
        genai.configure(api_key=gemini_api_key)

        model = genai.GenerativeModel("gemini-1.5-pro")

        system_instruction = """
        You are an AI specialized in logical reasoning, critical thinking, and problem-solving.
        Carefully analyze the given prompt and provide a structured response in this format:

        {
            "Explanation": "Briefly explain what the question is asking",
            "Formula or Technique": "Mention the formula, rule, or logical principle used",
            "Completion Steps": "Detailed step-by-step process to reach the answer",
            "Final Answer": "The correct logical conclusion",
        }

        Respond ONLY with raw JSON. Do NOT use code blocks or markdown.
        """

        convo = model.start_chat(history=[])
        convo.send_message(system_instruction)
        response = convo.send_message(prompt)

        cleaned = clean_json_text(response.text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logging.error("Failed to parse response JSON.")
            return {"Error": "Gemini returned an invalid JSON format.", "Raw": response.text}

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return {"Error": f"Unexpected error: {str(e)}"}

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('prompt', '')
    requested_model = data.get('model', '')
    
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400
    
    logging.info(f"Received request with input: {user_input}")
    
    model_used = None
    
    if requested_model:
        if requested_model == "claude" or requested_model == "deepseek":
            result = claude_llm(user_input)
            model_used = requested_model
        elif requested_model == "chatgpt":
            result = openai_llm(user_input)
            model_used = "chatgpt"
        elif requested_model == "gemini":
            result = gemini_llm(user_input)
            model_used = "gemini + Dall-E"
        elif requested_model == "gemini-llama":
            result = gemini_llama_llm(user_input)
            model_used = "llama"
        else:
            task_type = main_llm(user_input)
            
            if task_type == "Coding related task":
                result = claude_llm(user_input)
                model_used = "claude"
            elif task_type == "Human touch related task":
                result = openai_llm(user_input)
                model_used = "chatgpt"
            elif task_type == "Image processing related question":
                result = gemini_llm(user_input)
                model_used = "gemini + Dall-E"
            elif task_type == "Logical thinking related question":
                result = gemini_llama_llm(user_input)
                model_used = "llama"
            else:
                result = claude_llm(user_input)
                model_used = "claude"
    
    response_text = ""
    if isinstance(result, dict):
        if "task" in result and "response" in result:
            response_text = result["response"]
        elif "Output" in result:
            response_text = result["Output"]
        elif "Response" in result:
            response_text = result["Response"]
        elif "Code" in result:
            response_text = f"{result.get('Intro', '')}\n\n```\\cn{result['Code']}\n```\n\n{result.get('Conclusion', '')}"
        elif "Image URL" in result:
            image_url = result.get("Image URL", "")
            response_text = f"I've created an image based on your prompt.\n\nImage URL: {image_url}"

        elif "Final Answer" in result:
            response_text = f"{result.get('Explanation', '')}\n\n{result.get('Completion Steps', '')}\n\nFinal Answer: {result.get('Final Answer', '')}"
        else:
            response_text = json.dumps(result, indent=2)
    else:
        response_text = str(result)
    
    response = {
        'response': response_text,
        'model': model_used
    }
    
    return jsonify(response)

@app.route('/api/predict', methods=['POST'])
def predict():
    return chat()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://ai-router-three.vercel.app/"])


if _name_ == '_main_':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
