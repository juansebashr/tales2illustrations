"""
Main script to generate images from PDF files using DALL-E and OpenAI's GPT-4.
"""

# Imports
import os

from openai import OpenAI
import PyPDF2
from kiwi_booster.gcp_utils.secrets import access_secret_version


openai_api_key = access_secret_version(
    secret_id="openai-key", version_id=1, project_id="autonomy-286821"
)

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
)


DEFAULT_MODEL = "gpt-4-1106-preview"


def read_pdf(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ""
        for page in range(reader.numPages):
            text += reader.getPage(page).extractText()
    return text


def summarize_text(text):
    prompt = f"""
    Summarize the following text inside triple quotes into a natural divisions of its chapters and sections:
    '''{text}'''
    """
    messages = [{"role": "user", "content": prompt}]    
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        #temperature=0,  # Degree of randomness of the model
    )
    return response.choices[0].message["content"], response.usage["total_tokens"]


def generate_dalle_prompts(summary, style):
    # This function should convert the summary into DALL-E prompts.
    # The implementation depends on how you want to parse and use the summary.
    # Example:
    prompts = []
    for division in summary.split(
        "\n\n"
    ):  # Assuming each division is separated by two newlines
        prompt = f"Create an image in the style of {style} that represents: {division}"
        prompts.append(prompt)
    return prompts

def generate_images(prompts):
    for prompt in prompts:
        response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
        )

        image_url = response.data[0].url


def main(pdf_path, image_style):
    text = read_pdf(pdf_path)
    summary = summarize_text(text)
    dalle_prompts = generate_dalle_prompts(summary, image_style)

    for i, prompt in enumerate(dalle_prompts):
        print(f"Prompt {i+1}: {prompt}")


if __name__ == "__main__":
    PDF_FILE_PATH = "path/to/your/pdf/file.pdf"
    IMAGE_STYLE = "your-desired-image-style"

    main(PDF_FILE_PATH, IMAGE_STYLE)
