from PIL import Image

def load_icon_box(path, box_size=(200, 200)):
    img = Image.open(path).convert("RGBA")
    canvas = Image.new("RGBA", box_size, (255, 255, 255, 0))
    img.thumbnail(box_size, Image.Resampling.LANCZOS)
    x = (box_size[0] - img.width) // 2
    y = (box_size[1] - img.height) // 2
    canvas.paste(img, (x, y), img)
    return canvas
