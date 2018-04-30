from PIL import Image, ImageDraw, ImageFont

IMAGE_FILE = '/home/shinobu/Downloads/sample.jpg'

def resize():
  image_file = Image.open(IMAGE_FILE, 'r')
  resized_image = image_file.resize((100, 100))
  resized_image.save(
    '/tmp/resized_image0.jpg',
    quality = 100,
    optimize = True)

  image_file.thumbnail((100, 100), Image.ANTIALIAS)
  image_file.save(
    '/tmp/resized_image1.jpg',
    quality = 100,
    optimize = True)

def text_to_image():
  text_canvas = Image.new(
    'RGB',
    (80, 40),
    (255, 255, 255))
  draw = ImageDraw.Draw(text_canvas)
  font_type = ImageFont.truetype(
    '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
    15)
  draw.text(
    (10, 10),
    'abc',
    font = font_type,
    fill = '#000')
  text_canvas.save(
    '/tmp/text_image.jpg',
    'jpeg',
    quality = 100,
    optimize = True)

if __name__ == '__main__':
  resize()
  text_to_image()
