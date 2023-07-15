## Binding Of Isaac Item Dataset
Images scraped (using BeautifulSoup) from the BOI Fandom Wiki: https://bindingofisaacrebirth.fandom.com/wiki/Items#Passive_Collectibles with 200 image augmentations for each item.

### Format of data
Each item is a 32z32 PNG, stored in a separate directory, with 200 images, each image has random augmentations. (Augmentations done because of limited data available).
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

### Why am I doing this?
I'm using it for a personal project of mine, so thought I might as well share it with others :)

## Questions / Contact?
Send an Issue if you want to make any requests :))
