## Binding Of Isaac Item Dataset
Images scraped (using BeautifulSoup) from the BOI Fandom Wiki: https://bindingofisaacrebirth.fandom.com/wiki/Items#Passive_Collectibles with 200 image augmentations for each item.

### Format of data
Each item is a 40x40 PNG, stored in a separate directory, with 200 images, each image has random augmentations. (Augmentations done because of limited data available).

### What augmentations?
```python
data_transforms = transforms.Compose([
        transforms.Resize((32, 32)), 
        transforms.RandomPerspective(p=0.85), 
        transforms.ToTensor(),
        transforms.Pad(4), # pad images because there's gonna be random pixels around the item (unlikely to be by itself)
        transforms.GaussianBlur(1, sigma=(0.1, 1.0)),
        transforms.Lambda(lambda x: x * 255.0),
        transforms.Lambda(lambda x: x.byte()),
        transforms.ToPILImage(),
        transforms.Lambda(color_jitter_channels),
    ])
```

### Why am I doing this?
I'm using it for a personal project of mine, so thought I might as well share it with others :)

## Questions / Contact?
Send an Issue if you want to make any requests :))
