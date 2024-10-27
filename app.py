from flask import Flask, render_template, request, redirect, url_for
from openai import OpenAI
from transformers import BertForSequenceClassification, BertTokenizer
import torch

app = Flask(__name__)

# Load the saved model and tokenizer
model_save_path = "/Users/srinidhia/Library/CloudStorage/OneDrive-RMITUniversity/Semester 3/Case Studies/Project/Project/saved_model"
model = BertForSequenceClassification.from_pretrained(model_save_path)
tokenizer = BertTokenizer.from_pretrained(model_save_path)

# If you have a GPU available, move the model to GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

# Set the model to evaluation mode
model.eval()

# Initialize API key and OpenAI client
api_key = "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"
client = OpenAI(api_key=api_key)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    user_text = request.form['user_text']

    # BERT Model Classification
    encoded_dict = tokenizer.encode_plus(
        user_text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoded_dict['input_ids'].to(device)
    attention_mask = encoded_dict['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]

    # Label the prediction result
    bert_classification = 'Demeaning' if prediction == 1 else 'Non-Demeaning'

    # Pass the original input and BERT classification to the result.html page
    return render_template('result.html', 
                           original_input=user_text, 
                           bert_classification=bert_classification)


@app.route('/equaleyes', methods=['POST'])
def equaleyes():
    user_text = request.form['user_text']
    bert_classification = request.form['bert_classification']

    # If the classification is "Demeaning", make the GPT API call
    if bert_classification == 'Demeaning':
        user_input_with_suffix = f"{user_text}, change this statement with no demeaning or stereotyping content. Note: Provide the content only with no explanation."
        
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",  # Adjust this if you're using another model
                messages=[
                    {"role": "user", "content": user_input_with_suffix}
                ]
            )
            generated_text = completion.choices[0].message.content
        except Exception as e:
            generated_text = f"An error occurred: {str(e)}"
    else:
        # If "Non-Demeaning", simply show the message
        generated_text = "The text doesn't contain any demeaning or stereotype words."

    # Pass the results to the result.html page to display them
    return render_template('result.html', 
                           original_input=user_text, 
                           alternative_text=generated_text, 
                           bert_classification=bert_classification)


if __name__ == '__main__':
    app.run(debug=True)
