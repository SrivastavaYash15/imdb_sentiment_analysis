import comet_ml
comet_ml.init(project_name= "imdb_reviwes")
pre_trained_model = "distilbert-base-uncased"
seed = 20

#loading the data
from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
dataset = load_dataset('imdb')

#tekensizing the data
tokenizer = AutoTokenizer.from_pretrained(pre_trained_model) #automatically loads the tokenier for that specific model
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

print(tokenized_datasets["train"][0])
#data colator = used for padding the data
from transformers import DataCollatorWithPadding
data_colator = DataCollatorWithPadding(tokenizer =tokenizer)

#creating sample datasets
tokenized_datasets = tokenized_datasets["train"].train_test_split(test_size=0.2, seed=seed)

train_dataset = tokenized_datasets["train"].shuffle(seed=seed).select(range(200))
eval_dataset = tokenized_datasets["test"].shuffle(seed=seed).select(range(200))


#set up the transformer model

from transformers import AutoModelForSequenceClassification  #using the AutoModel from hugging face
model = AutoModelForSequenceClassification.from_pretrained(pre_trained_model, torch_dtype = "auto", num_labels =2) #num_labels =2 for binary classification


#evalution 
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
def get_example(index): #to access the text from the listlike dataset objects
    return eval_dataset[index]["text"]

def compute_metrics(pred):
    experiment = comet_ml.get_global_experiment()
    
    label = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(label, preds, average = "macro")
    accracy = accuracy_score(label, preds)
    
    if experiment:
        epoch = int(experiment.curr_epoch) if experiment.curr_epoch else 0 #if the experiment is running
        experiment.set_epoch(epoch)
        experiment.log_confusion_matrix(
            y_true =label,
            y_predicted=preds,
            file_name = f"confusion_matrix_{epoch}.json",
            labels = ["negative", "positive"],
            index_to_example_function= get_example    
        )
    for i in range(20):
        experiment.log_text(get_example(i), metadata = {"label": labels[i].item()})
        
    return {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}


from transformers import TrainingArguments
import os

os.environ["COMET_MODE"] = "ONLINE"  
os.environ["COMET_LOG_ASSETS"] = "TRUE"  # Set COMET_LOG_ASSETS to "TRUE"





training_args = TrainingArguments(
    output_dir='./results',          # Directory to save checkpoints and logs
    num_train_epochs=3,
    do_train = True,
    do_eval =True,
    evaluation_strategy = "epoch",
    eval_steps = 25,
    save_total_limit=3,
    save_steps = 25,
    per_device_train_batch_size=8 
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_colator,
    tokenizer=tokenizer,
)
trainer.train()
