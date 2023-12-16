import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
import numpy as np
import pandas as pd
from transformers import TrainingArguments, Trainer
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification,VideoMAEConfig
import torchvision
import pytorch_lightning as pl
import torch
from PIL import Image
import PIL
import evaluate
import datetime
from sklearn.model_selection import train_test_split

seed = 41
torch.manual_seed(seed)
model_ckpt = "MCG-NJU/videomae-base" 
batch_size = 8 
cv = pd.read_csv("frame_label")
label = cv["Label"].unique()
trainset, validset = train_test_split(cv, test_size=0.2,shuffle=True)

class CustomDataset(torch.utils.data.Dataset): 
  def __init__(self,frames,trans):
    self.df = frames
    self.label = frames['Label']
    self.trans = trans
  def __len__(self): 
    return len(self.df['Label'])
  def __getitem__(self, idx): 
    frame = self.df.iloc[idx]
    images = []
    # label = frame['Label']
    # label = label2id[label]
    for _ in range(10):
      try:
        images.append(self.trans(Image.open(frame[f"frames{_+1}"])))
      except PIL.UnidentifiedImageError:
        images.append(self.trans(Image.open("/home/yangcw/video/Data1/error.jpg")))
    return {"video":torch.stack([x for x in images]),"label":1}
  
transform_train = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
                                                  torchvision.transforms.CenterCrop(224),
                                                            torchvision.transforms.RandomHorizontalFlip(),
                                                            torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
                                                                ])
transform_val = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
                                                            torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
                                                                ])
# train_dataset = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
# val_dataset = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True)
train_dataset = CustomDataset(trainset,transform_train)
val_dataset =CustomDataset(validset,transform_val)
# label2id = {label: i for i, label in enumerate(label)}
# id2label = {i: label for label, i in label2id.items()}
label2id = {'neg':1,"pos":0}
id2label = {1:'neg',0:'pos'}
configuration = VideoMAEConfig(num_frames = 10)
image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
model = VideoMAEForVideoClassification.from_pretrained(
    model_ckpt,
    # config = configuration,
    num_frames = 10,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    
)

model_name = model_ckpt.split("/")[-1]
new_model_name = f"{model_name}-finetuned-action-subset"
num_epochs = 4


args = TrainingArguments(
    # new_model_name,
    output_dir="./bin",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    # push_to_hub=True,
    max_steps=int(cv.iloc[-1]['id']+1 // batch_size) * num_epochs,
    logging_dir=f'./logs/{datetime.datetime.now().time()}',  
    seed= seed,
    
)

metric = evaluate.load("accuracy")

# def collate_fn(examples):
#     """The collation function to be used by `Trainer` to prepare data batches."""
#     # permute to (num_frames, num_channels, height, width)
#     pixel_values = torch.stack(
#         [example["video"].permute(1, 0, 2, 3) for example in examples]
#     )
#     labels = torch.tensor([example["label"] for example in examples])
#     return {"pixel_values": pixel_values, "labels": labels}
def collate_fn(examples):
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example["video"] for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
  
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions."""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)
trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn)
train_results = trainer.train()
trainer.save_model()
test_results = trainer.evaluate(val_dataset)
trainer.log_metrics("test", test_results)
trainer.save_metrics("test", test_results)
trainer.save_state()
