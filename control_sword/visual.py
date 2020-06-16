from PIL import ImageDraw, Image


def draw_circle_on_image(draw, x: int, y: int, fill, radius=4):
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)


def mark_sword_on_image(img: Image, sword_coordinates: [int, int, int, int], fill=(255, 0, 255)):
    draw = ImageDraw.Draw(img)
    draw.line((sword_coordinates[0], sword_coordinates[1], sword_coordinates[2], sword_coordinates[3]),
              fill=fill, width=3)
    draw_circle_on_image(draw, sword_coordinates[0], sword_coordinates[1], (255, 0, 0))
    draw_circle_on_image(draw, sword_coordinates[2], sword_coordinates[3], (0, 0, 255))
    del draw

# os.getcwd()
