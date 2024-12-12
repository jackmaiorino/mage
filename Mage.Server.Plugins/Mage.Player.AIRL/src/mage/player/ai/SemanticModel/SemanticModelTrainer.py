from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset, concatenate_datasets
import re
import logging
import sys
import torch  # Keep this for device checking

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Function to preprocess the card dataset
def preprocess_card_function(examples):
    fields = examples['text'].split('|')
    # Extract the Type-line and Card Text
    type_line = fields[5] if len(fields) > 5 else ""
    card_text = fields[8] if len(fields) > 8 else ""
    # Process the card text to handle mana costs
    processed_card_text = preprocess_ability_costs(card_text)
    # Combine them for the model input
    combined_text = type_line + " " + processed_card_text
    return {'text': combined_text}

# Function to preprocess the rules dataset
def preprocess_rules_function(examples):
    # Remove rule numbers and clean up the text
    cleaned_text = re.sub(r'^\d+\.\d+[a-z]?\s*', '', examples['text'])
    return {'text': cleaned_text}

def preprocess_ability_costs(text):
    # Combine all adjacent mana costs into a single cost
    text = re.sub(r'(\{(?:\d+|[WUBRGT])\})+', lambda m: 'COST_' + ''.join(m.group(0).replace('{', '').replace('}', '')), text)
    return text

# Add this function before using it
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512
    )

logger.info("Starting script execution")

try:
    logger.info("Loading datasets...")
    cards_dataset = load_dataset('text', data_files={'train': '/Users/jack/Dev/mage/Utils/mtg-cards-data.txt'})
    logger.info(f"Cards dataset loaded: {len(cards_dataset['train'])} entries")
    
    rules_dataset = load_dataset('text', data_files={'train': '/Users/jack/Dev/mage/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/SemanticModel/MagicCompRules20241108.txt'})
    logger.info(f"Rules dataset loaded: {len(rules_dataset['train'])} entries")

    logger.info("Preprocessing datasets...")
    processed_cards = cards_dataset.map(preprocess_card_function, batched=False)
    logger.info("Cards preprocessing complete")
    
    processed_rules = rules_dataset.map(preprocess_rules_function, batched=False)
    logger.info("Rules preprocessing complete")

    logger.info("Combining datasets...")
    combined_dataset = concatenate_datasets([processed_cards['train'], processed_rules['train']])
    logger.info(f"Combined dataset size: {len(combined_dataset)} entries")

    logger.info("Loading BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    logger.info("BERT model loaded")

    logger.info("Tokenizing dataset...")
    tokenized_datasets = combined_dataset.map(tokenize_function, batched=True)
    logger.info("Tokenization complete")

    logger.info("Setting up training...")
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    logger.info("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    trainer.train()
    logger.info("Training complete!")

    # Assuming 'model' is your trained BERT model
    dummy_input = torch.randint(0, 1000, (1, 512))  # Example input, adjust as needed
    torch.onnx.export(
        model,
        dummy_input,
        "bert_model.onnx",
        input_names=["input_ids"],
        output_names=["output"],
        dynamic_axes={"input_ids": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )

except Exception as e:
    logger.error(f"An error occurred: {str(e)}", exc_info=True)
    raise