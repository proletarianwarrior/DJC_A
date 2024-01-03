from PIL import Image

image = Image.open("data/mask/mask.jpg")

# 定义新的宽度和高度
new_width = 1000
new_height = 1000

# 放大图像
enlarged_image = image.resize((new_width, new_height))
enlarged_image = enlarged_image.convert("RGB")
enlarged_image.save("data/mask/enlarged_image.jpg")

