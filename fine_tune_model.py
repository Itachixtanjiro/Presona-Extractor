import os
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from src.preprocessing import read_docx, read_pdf, clean_text, _regex_sent_tokenize
from src.config import EMBEDDING_MODEL
from src.logger import log

# --- CONFIGURATION ---
# Directory containing your domain-specific documents for training
TRAINING_DATA_DIR = "rules_problem_statement" 
# Where to save the newly fine-tuned model
FINE_TUNED_MODEL_PATH = f"./models/finetuned-{EMBEDDING_MODEL.split('/')[-1]}"
# Training parameters
TRAIN_BATCH_SIZE = 16
TRAIN_EPOCHS = 1

def create_training_examples(docs_dir: str) -> list[InputExample]:
    """
    Reads all documents from a directory and creates positive sentence pairs
    for unsupervised fine-tuning.
    """
    log.info(f"Creating training examples from directory: {docs_dir}")
    training_examples = []
    
    for filename in os.listdir(docs_dir):
        file_path = os.path.join(docs_dir, filename)
        text = ""
        try:
            if filename.endswith(".docx"):
                text = read_docx(file_path)
            elif filename.endswith(".pdf"):
                text = read_pdf(file_path)
            else:
                continue # Skip other files
                
            cleaned_text = clean_text(text)
            sentences = _regex_sent_tokenize(cleaned_text)
            
            # Create positive pairs from adjacent sentences
            for i in range(len(sentences) - 1):
                # InputExample format for this loss is just a list of two similar sentences
                example = InputExample(texts=[sentences[i], sentences[i+1]])
                training_examples.append(example)
                
        except Exception as e:
            log.error(f"Failed to process file {filename}: {e}")
            
    log.info(f"Created {len(training_examples)} training examples.")
    return training_examples

def run_fine_tuning():
    """
    Main function to run the fine-tuning process.
    """
    log.info("--- Starting Model Fine-Tuning ---")
    
    # 1. Load a pre-trained model
    log.info(f"Loading base model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # 2. Create training data
    train_examples = create_training_examples(TRAINING_DATA_DIR)
    if not train_examples:
        log.error("No training examples were created. Aborting fine-tuning.")
        return
        
    # 3. Set up the data loader and loss function
    # The dataloader shuffles and batches the training examples
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=TRAIN_BATCH_SIZE)
    # MultipleNegativesRankingLoss is excellent for this type of unsupervised training
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # 4. Run the training
    log.info(f"Starting training for {TRAIN_EPOCHS} epoch(s)...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=TRAIN_EPOCHS,
        show_progress_bar=True,
        output_path=FINE_TUNED_MODEL_PATH,
        # Increase warmup steps for better stability
        warmup_steps=int(len(train_dataloader) * 0.1) 
    )
    
    log.info(f"--- Fine-Tuning Complete ---")
    log.info(f"Model saved to: {FINE_TUNED_MODEL_PATH}")

if __name__ == "__main__":
    run_fine_tuning()