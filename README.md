## Binding Of Isaac Item Dataset
Images scraped (using BeautifulSoup) from the BOI Fandom Wiki: https://bindingofisaacrebirth.fandom.com/wiki/Items with 200 image augmentations for each item.

### Format of data
Each item is a 32x32 PNG, stored in a separate directory, with 200 images, each image has random augmentations. (Augmentations done because of limited data available).
There are 720 items in total

### What augmentations?
```python
data_transforms = transforms.Compose([
        transforms.Resize(30), 
        transforms.RandomPerspective(p=0.75), 
        transforms.ToTensor(),
        transforms.Pad(1), # pad images because there's gonna be random pixels around the item (unlikely to be by itself)
        transforms.GaussianBlur(1, sigma=(0.1, 0.3)),
        transforms.ToPILImage(),
    ])
```

### Example Use (In PyTorch)
```python
ACTIVATED_COLLECTIBLE_DATA_DIR = "/data/activated-collectible-icons"
PASSIVE_COLLECTIBLE_DATA_DIR = "/data/passive-collectible-icons"

# feel free to change these ratios
TRAIN_DATA_RATIO = 0.8 # 80% = training
VALIDATE_DATA_RATIO = 0.08 # 8% = validation
TEST_DATA_RATIO = 0.12 # 12% = testing
RATIOS = [TRAIN_DATA_RATIO, VALIDATE_DATA_RATIO, TEST_DATA_RATIO]

weights = ...
model_transforms = weights.transforms() 

activated_collectible_dataset = ImageFolder(ACTIVATED_COLLECTIBLE_DATA_DIR, model_transforms)
passive_collectible_dataset = ImageFolder(PASSIVE_COLLECTIBLE_DATA_DIR, model_transforms)
collectible_dataset = ConcatDataset([activated_collectible_dataset, passive_collectible_dataset]) # we're only training a single model, so combine datasets
train_set, validate_set, test_set = random_split(collectible_dataset, RATIOS)
collectible_datasets = {"train": train_set, "validate": validate_set, "test": test_set}

collectible_class_names = activated_collectible_dataset.classes + passive_collectible_dataset.classes

collectible_dataloader = DataLoader(collectible_dataset, batch_size=32, shuffle=True)
```


### Why am I doing this?
I'm using it for a personal project of mine, so thought I might as well share it with others :)

## Questions / Contact?
Create an Issue if you want to make any requests :))
