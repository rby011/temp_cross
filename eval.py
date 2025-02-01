import openai
import base64

# 1. Azure OpenAI Configuration
openai.api_type = "azure"
openai.api_base = "https://YOUR_AZURE_OPENAI_ENDPOINT.openai.azure.com"
openai.api_version = "2023-09-15-preview"  # Example, version may vary
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
    image_path: str = "",
    difficulty_level: str = "Medium"
) -> str:
    """
    Uses GPT-4V (Vision) to evaluate whether the model's output is correct 
    compared to the reference answer, considering both text and image context.
    """
    # 1) Load the template
    template_text = load_prompt_template("evaluation_prompt.txt")
    
    # 2) Prepare the image_info placeholder
    image_info = ""
    if image_path:
        # If you want to pass an actual image to GPT-4V, you could do:
        # - Base64-encode the image and embed it in the prompt
        # - Or attach it as a separate 'file' parameter, depending on API availability
        # For demonstration, we store base64 data in "image_info"
        base64_image = read_image_as_base64(image_path)
        image_info = f"Base64 Image Data:\n{base64_image}"
    else:
        image_info = "No image provided."
    
    # 3) Replace placeholders in the template
    prompt_for_user = template_text.replace("{{question_type}}", question_type)
    prompt_for_user = prompt_for_user.replace("{{difficulty_level}}", difficulty_level)
    prompt_for_user = prompt_for_user.replace("{{question}}", question)
    prompt_for_user = prompt_for_user.replace("{{image_info}}", image_info)
    prompt_for_user = prompt_for_user.replace("{{reference_answer}}", reference_answer)
    prompt_for_user = prompt_for_user.replace("{{model_output}}", model_output)
    
    # Split the template into system and user messages
    # We assume the template is structured such that "System:" and "User:" 
    # separate system vs. user content. We'll parse it manually here.
    # A simpler approach could be having two separate .txt files, 
    # but for demonstration we parse one file.
    
    # We look for "System:\n" and "User:\n" as markers:
    system_marker = "System:"
    user_marker = "User:"
    
    # We'll split once on "System:" (discarding it), then split on "User:"
    # In a robust solution, you'd parse carefully, but let's keep it simple:
    # 1) Split the entire text by "System:"
    parts_after_system = prompt_for_user.split(system_marker, 1)
    if len(parts_after_system) < 2:
        raise ValueError("Template missing 'System:' section.")
    
    # 2) The substring after "System:" may contain "User:"
    # We'll split that part by "User:" to isolate system content vs user content
    parts_after_user = parts_after_system[1].split(user_marker, 1)
    if len(parts_after_user) < 2:
        raise ValueError("Template missing 'User:' section.")
    
    system_content = parts_after_user[0].strip()
    user_content = parts_after_user[1].strip()
    
    # 4) Call GPT-4V API
    response = openai.ChatCompletion.create(
        engine="YOUR_DEPLOYMENT_NAME",  # e.g., "gpt-4-vision"
        messages=[
            {"role": "system", "content": system_content},
            # For GPT-4V with an image, some environments may require a special parameter.
            # If not supported, we embed base64 in the user content:
            {"role": "user", "content": user_content},
        ],
        # If the API supports file attachments for images, you'd pass them as well.
        # e.g., files=[("image", open(image_path, "rb"))]  # hypothetical usage
        max_tokens=500,
        temperature=0.0
    )
    
    # 5) Return the text response
    return response.choices[0].message["content"].strip()


if __name__ == "__main__":
    # Example usage:
    q_type = "Image-based"
    q_text = "What animal is shown in the attached image?"
    ref_answer = "It is a cat."
    model_out = "I see a cat, specifically a small domestic cat."
    
    # Suppose you have an image "cat_photo.jpg" in the same directory:
    image_path_example = "cat_photo.jpg"
    
    result = evaluate_with_gpt4v(
        question_type=q_type,
        question=q_text,
        reference_answer=ref_answer,
        model_output=model_out,
        image_path=image_path_example,
        difficulty_level="Easy"
    )
    
    print("\n=== EVALUATION RESULT ===")
    print(result)
