import openai
import base64
import traceback

openai.api_type = "azure"
openai.api_base = "https://YOUR_AZURE_OPENAI_ENDPOINT.openai.azure.com"
openai.api_version = "2023-09-15-preview"  # Adjust to the version you're using
openai.api_key = "YOUR_AZURE_OPENAI_API_KEY"

def load_prompt_template(file_path: str) -> str:
    """Load the prompt template from a .txt file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def read_image_as_base64(image_path: str) -> str:
    """Convert an image file to a base64-encoded string."""
    with open(image_path, 'rb') as img:
        encoded_str = base64.b64encode(img.read()).decode('utf-8')
    return encoded_str

def evaluate_with_gpt4v(
    question_type: str,
    question: str,
    reference_answer: str,
    model_output: str,
    format_instructions: str = "",
    image_path: str = "",
    difficulty_level: str = "Medium"
) -> str:
    """
    Evaluate the model's output against a reference answer, considering 
    correctness AND format instructions (e.g., single word, short answer),
    plus image context if using GPT-4V.
    """
    # 1) Load the template from file
    template_text = load_prompt_template("evaluation_prompt.txt")
    
    # 2) Prepare the image_info
    if image_path:
        base64_image = read_image_as_base64(image_path)
        image_info = f"Base64 Image Data:\n{base64_image}"
    else:
        image_info = "No image provided."
    
    # 3) Replace placeholders
    prompt_text = template_text \
        .replace("{{question_type}}", question_type) \
        .replace("{{difficulty_level}}", difficulty_level) \
        .replace("{{format_instructions}}", format_instructions) \
        .replace("{{question}}", question) \
        .replace("{{image_info}}", image_info) \
        .replace("{{reference_answer}}", reference_answer) \
        .replace("{{model_output}}", model_output)
    
    # Separate system and user content by splitting on "System:" and "User:" markers
    system_marker = "System:"
    user_marker = "User:"
    
    parts_system = prompt_text.split(system_marker, 1)
    if len(parts_system) < 2:
        raise ValueError("Template missing 'System:' section.")
    parts_user = parts_system[1].split(user_marker, 1)
    if len(parts_user) < 2:
        raise ValueError("Template missing 'User:' section.")
    
    system_content = parts_user[0].strip()
    user_content = parts_user[1].strip()
    
    # 4) Call the Azure OpenAI ChatCompletion (GPT-4V or GPT-4)
    try:
        response = openai.ChatCompletion.create(
            engine="YOUR_DEPLOYMENT_NAME",  # e.g., "gpt-4-vision"
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            max_tokens=500,
            temperature=0.0
        )
        answer = response.choices[0].message["content"].strip()
        return answer
    
    except Exception as e:
        # In case of error, show message & stack trace
        print("Error during ChatCompletion:", str(e))
        traceback.print_exc()
        return ""

if __name__ == "__main__":
    # Example usage
    q_type = "Open-ended"
    q_text = "What animal is considered a human's best friend? Answer with a single word."
    ref_answer = "Dog"
    model_ans = "A dog."
    
    # The question specifically says "Answer with a single word."
    # We'll pass that to format_instructions so the evaluator can check compliance.
    format_instruction = "Answer with a single word."
    
    result = evaluate_with_gpt4v(
        question_type=q_type,
        question=q_text,
        reference_answer=ref_answer,
        model_output=model_ans,
        format_instructions=format_instruction,
        image_path="",  # No image this time
        difficulty_level="Easy"
    )
    
    print("\n=== EVALUATION RESULT ===")
    print(result)
